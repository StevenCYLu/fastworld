"""Static-shape, CUDA-graph runtime for LongLive-2.0-5B (BF16, T2V, guidance 1.0, multi-shot sink).

Same kernels, same values as the official eager pipeline (`pipeline/causal_diffusion_inference.py`):

* KV cache: one fixed buffer per layer, [1, 28160, 24, 128] (= local_attn_size 32 frames x 880 tokens), laid out
  EXACTLY like the official cache (sink | rolled history | new chunk). The official code reads the cache indices with
  `.item()` before every forward, clones the whole cache (2 x 173 MB per layer per forward), rolls/inserts into the
  clone, attends over clone[:, :local_end], then applies the same roll/insert to the real cache. Here the index
  arithmetic (incl. sink / multi-shot pinning / shot_clean_recache) is replayed on the host with Python ints
  (`_plan`), the roll is applied once per chunk (chunked in-place left shift), and every forward writes its new K/V
  straight into the slot [L-7040, L) and attends over buf[:, :L]. Since local_attn_size == cache size, the attended
  window is always buf[:, :local_end] (prepend-sink/pinned branches are unreachable), so only 4 shapes exist:
  L in {7040, 14080, 21120, 28160}.
* Everything chunk-dependent enters via static buffers: noise chunk, RoPE cos/sin tables (incl. multi-shot temporal
  offset), cross-attention K/V of the chunk's prompt (recomputed only when the prompt changes; the official code
  recomputes text_embedding + 30 x (k,v) projections on every forward but only uses them for this).
  Constants: timestep sinusoids, cu_seqlens, UniPC device constants (rhos), all built once outside capture.
* flash_attention wrapper (builds cu_seqlens on the CPU + H2D per call) -> flash_attn_varlen_func with device cu_seqlens.
* One CUDA graph per bucket L for a whole chunk: 4 denoising forwards + 4 UniPC steps + the clean context
  (timestep 0) forward that writes the cache. Prompt switches only change static inputs (RoPE offset, cross K/V) and
  the host-side roll; a `shot_clean_recache` scene cut (context forward at a different L) runs the same static code
  without a graph.
* The wrapper's `_convert_flow_pred_to_x0` (fp64, result discarded by the pipeline) is not executed.
"""
import math
import torch
from fastworld import graph
from fastworld.models.longlive2.model import FPB, TOK
from wan_5b.modules.model import sinusoidal_embedding_1d
from wan_5b.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from utils.adaln_triton import adaln_modulate_triton
from utils.rope_triton import rope_apply_triton, _split_complex_to_cos_sin
from utils.position_embedding_utils import compute_temporal_freqs, select_temporal_offset_for_sample
import flash_attn

N = FPB * TOK          # 7040 tokens per chunk


class StaticUniPC(FlowUniPCMultistepScheduler):
    """FlowUniPCMultistepScheduler.step with the per-call `torch.tensor(..., device=cuda)` constants (H2D copies, not
    capturable) and the CUDA linalg.solve (host sync) computed once per step index and cached. Same ops otherwise."""

    def setup(self):
        self._c = {}

    def reset(self):
        self.model_outputs = [None] * self.config.solver_order
        self.timestep_list = [None] * self.config.solver_order
        self.lower_order_nums = 0
        self.last_sample = None
        self._step_index = 0          # == index_for_timestep(timesteps[0]) (timesteps are distinct)

    def _const(self, key, fn):
        if key not in self._c:
            self._c[key] = fn()
        return self._c[key]

    def multistep_uni_p_bh_update(self, model_output, sample=None, order=None, **kw):
        model_output_list = self.model_outputs
        m0 = model_output_list[-1]
        x = sample
        sigma_t, sigma_s0 = self.sigmas[self.step_index + 1], self.sigmas[self.step_index]
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)
        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
        h = lambda_t - lambda_s0
        device = sample.device
        rks, D1s = [], []
        for i in range(1, order):
            si = self.step_index - i
            mi = model_output_list[-(i + 1)]
            alpha_si, sigma_si = self._sigma_to_alpha_sigma_t(self.sigmas[si])
            lambda_si = torch.log(alpha_si) - torch.log(sigma_si)
            rk = (lambda_si - lambda_s0) / h
            rks.append(rk)
            D1s.append((mi - m0) / rk)
        hh = -h if self.predict_x0 else h
        h_phi_1 = torch.expm1(hh)
        B_h = torch.expm1(hh)           # solver_type bh2
        assert self.config.solver_type == "bh2" and order <= 2
        if len(D1s) > 0:
            D1s = torch.stack(D1s, dim=1)
            rhos_p = self._const(("p", self.step_index, x.dtype),
                                 lambda: torch.tensor([0.5], dtype=x.dtype, device=device))
        else:
            D1s = None
        assert self.predict_x0
        x_t_ = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0
        if D1s is not None:
            pred_res = torch.einsum("k,bkc...->bc...", rhos_p, D1s)
        else:
            pred_res = 0
        x_t = x_t_ - alpha_t * B_h * pred_res
        return x_t.to(x.dtype)

    def multistep_uni_c_bh_update(self, this_model_output, last_sample=None, this_sample=None, order=None, **kw):
        model_output_list = self.model_outputs
        m0 = model_output_list[-1]
        x = last_sample
        x_t = this_sample
        model_t = this_model_output
        sigma_t, sigma_s0 = self.sigmas[self.step_index], self.sigmas[self.step_index - 1]
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)
        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
        h = lambda_t - lambda_s0
        device = this_sample.device
        rks, D1s = [], []
        for i in range(1, order):
            si = self.step_index - (i + 1)
            mi = model_output_list[-(i + 1)]
            alpha_si, sigma_si = self._sigma_to_alpha_sigma_t(self.sigmas[si])
            lambda_si = torch.log(alpha_si) - torch.log(sigma_si)
            rk = (lambda_si - lambda_s0) / h
            rks.append(rk)
            D1s.append((mi - m0) / rk)
        rks.append(1.0)
        hh = -h if self.predict_x0 else h
        h_phi_1 = torch.expm1(hh)
        h_phi_k = h_phi_1 / hh - 1
        factorial_i = 1
        B_h = torch.expm1(hh)
        R_host, b_host = [], []
        for i in range(1, order + 1):
            R_host.append(i - 1)
            b_host.append(h_phi_k * factorial_i / B_h)
            factorial_i *= i + 1
            h_phi_k = h_phi_k / hh - 1 / factorial_i
        D1s = torch.stack(D1s, dim=1) if len(D1s) > 0 else None
        if order == 1:
            rhos_c = self._const(("c", self.step_index, order, x.dtype),
                                 lambda: torch.tensor([0.5], dtype=x.dtype, device=device))
        else:
            def solve():   # exactly the official construction (device rks/R/b, CUDA linalg.solve)
                rks_d = torch.tensor(rks, device=device)
                R = torch.stack([torch.pow(rks_d, p) for p in R_host])
                b = torch.tensor(b_host, device=device)
                return torch.linalg.solve(R, b).to(device).to(x.dtype)
            rhos_c = self._const(("c", self.step_index, order, x.dtype), solve)
        x_t_ = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0
        if D1s is not None:
            corr_res = torch.einsum("k,bkc...->bc...", rhos_c[:-1], D1s)
        else:
            corr_res = 0
        D1_t = model_t - m0
        x_t = x_t_ - alpha_t * B_h * (corr_res + rhos_c[-1] * D1_t)
        return x_t.to(x.dtype)


class StaticLL:
    def __init__(self, pipe):
        self.pipe = pipe
        net = self.net = pipe._dit_model
        dev = self.dev = net.patch_embedding.weight.device
        bf = torch.bfloat16
        if net.freqs.device != dev:
            net.freqs = net.freqs.to(dev)
        self.blocks = list(net.blocks)
        sa0 = self.blocks[0].self_attn
        self.H, self.D = sa0.num_heads, sa0.head_dim
        self.CS = pipe.local_attn_size * pipe.frame_seq_length          # 28160 cache tokens (== max_attention_size)
        self.sink = pipe.sink_size * pipe.frame_seq_length               # 7040
        self.gsink = pipe.global_sink_size * pipe.frame_seq_length       # 7040 (multi_shot_sink)
        self.multi_shot_sink, self.recache = pipe.multi_shot_sink, pipe.shot_clean_recache
        self.phi, self.prefix = pipe.multi_shot_rope_offset, pipe.scene_cut_prefix
        assert pipe.guidance_scale == 1.0 and not pipe.quantize_kv and not pipe.use_relative_rope
        assert pipe.frame_seq_length == TOK and pipe.num_frame_per_block == FPB
        # ---- state: per-layer KV buffers (same layout as the official cache) + host-side indices
        self.K = [torch.zeros(1, self.CS, self.H, self.D, device=dev, dtype=bf) for _ in self.blocks]
        self.V = [torch.zeros_like(t) for t in self.K]
        self.reset()
        # ---- static inputs
        self.x_in = torch.zeros(1, FPB, 48, 44, 80, device=dev, dtype=bf)
        self.cos = torch.zeros(N, self.D // 2, device=dev, dtype=torch.float32)
        self.sin = torch.zeros_like(self.cos)
        self.ck = [torch.zeros(1, 512, self.H, self.D, device=dev, dtype=bf) for _ in self.blocks]
        self.cv = [torch.zeros_like(t) for t in self.ck]
        self.out = torch.zeros(1, FPB, 48, 44, 80, device=dev, dtype=bf)
        self.cur_prompt = None
        # ---- constants: timesteps exactly as the official loop builds them, and their sinusoids
        sch = FlowUniPCMultistepScheduler(num_train_timesteps=pipe.num_train_timesteps, shift=1, use_dynamic_shifting=False)
        sch.set_timesteps(pipe.sampling_steps, device=dev, shift=pipe.shift)
        self.sch = StaticUniPC(num_train_timesteps=pipe.num_train_timesteps, shift=1, use_dynamic_shifting=False)
        self.sch.set_timesteps(pipe.sampling_steps, device=dev, shift=pipe.shift)
        self.sch.setup()
        self.n_steps = len(sch.timesteps)
        tts = [t * torch.ones([1, FPB], device=dev, dtype=torch.float32) for t in sch.timesteps]
        tts.append(tts[-1] * 0)                       # context forward: `timestep * 0`
        self.t_steps = list(sch.timesteps)
        self.sinus = [sinusoidal_embedding_1d(net.freq_dim, t.flatten()) for t in tts]
        self.cu = {}
        for n in (N, 512, 2 * N, 3 * N, 4 * N):
            self.cu[n] = torch.tensor([0, n], dtype=torch.int32, device=dev)
        self.grid = torch.tensor([[FPB, 22, 40]], dtype=torch.long)
        self.graphs = {}
        self.rec = None                  # debug: list collecting every denoising flow_pred (no-graph path)

    # ------------------------------------------------------------------ host-side cache bookkeeping
    def reset(self):
        self.k = 0
        self.ge, self.le, self.ps, self.pl = 0, 0, -1, 0     # global_end, local_end, pinned_start, pinned_len
        self.shot, self.toff = 0, 0.0
        self.cur_prompt = None

    def _eff_sink(self):
        has_pinned = self.ps >= 0 and self.pl > 0
        if has_pinned and self.ps == self.gsink:
            return has_pinned, self.gsink + self.pl
        if has_pinned:
            return has_pinned, self.gsink
        return has_pinned, max(self.gsink, self.sink)

    def _insert(self, cs):
        """One official forward's index update (CausalWanSelfAttention + _apply_cache_updates). Returns (L, roll)."""
        ce = cs + N
        has_pinned, eff = self._eff_sink()
        roll = None
        if ce > self.ge and N + self.le > self.CS:
            ev = N + self.le - self.CS
            r = self.le - ev - eff
            L = self.le + ce - self.ge - ev
            roll = (eff, ev, r)
            if has_pinned and self.ps >= eff:
                self.ps -= ev
        else:
            L = self.le + ce - self.ge
        self.ge, self.le = ce, L
        return L, roll

    def _roll(self, roll):
        s, ev, r = roll
        if r <= 0:
            return
        for buf in self.K + self.V:     # left shift [s+ev, s+ev+r) -> [s, s+r), in non-overlapping pieces
            o = 0
            while o < r:
                n = min(ev, r - o)
                buf[:, s + o:s + o + n].copy_(buf[:, s + ev + o:s + ev + o + n])
                o += n

    # ------------------------------------------------------------------ static inputs
    def _rope_tables(self, start_frame, temporal_offset):
        net = self.net
        c = self.D // 2
        fr = net.freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
        f, h, w = FPB, 22, 40
        toff = select_temporal_offset_for_sample(temporal_offset, 0, f, start_frame=start_frame)
        tf = compute_temporal_freqs(fr[0], f, start_frame, net.t_scale, self.dev, method=net.rope_method,
                                    original_seq_len=net.original_seq_len, temporal_offset=toff)
        fi = torch.cat([tf.view(f, 1, 1, -1).expand(f, h, w, -1),
                        fr[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                        fr[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)], dim=-1).reshape(f * h * w, 1, -1)
        return _split_complex_to_cos_sin(fi)

    def _set_prompt(self, embeds):
        """Cross-attention K/V of this prompt, computed as the official forward does on every call."""
        net = self.net
        context = net.text_embedding(torch.stack([torch.cat([u, u.new_zeros(net.text_len - u.size(0), u.size(1))])
                                                  for u in embeds]))
        for l, blk in enumerate(self.blocks):
            ca = blk.cross_attn
            self.ck[l].copy_(ca.norm_k(ca.k(context)).view(1, -1, self.H, self.D))
            self.cv[l].copy_(ca.v(context).view(1, -1, self.H, self.D))

    # ------------------------------------------------------------------ network
    def _fa(self, q, k, v):
        b, lq, lk = q.size(0), q.size(1), k.size(1)
        return flash_attn.flash_attn_varlen_func(
            q=q.flatten(0, 1), k=k.flatten(0, 1), v=v.flatten(0, 1), cu_seqlens_q=self.cu[lq], cu_seqlens_k=self.cu[lk],
            max_seqlen_q=lq, max_seqlen_k=lk, dropout_p=0., softmax_scale=None, causal=False, window_size=(-1, -1),
            deterministic=False).unflatten(0, (b, lq)).type(q.dtype)

    def _rope(self, x):
        return torch.stack([rope_apply_triton(x[0], self.cos, self.sin, seq_len=N)]).type_as(x)

    def _net(self, x, ti, L):
        """One official generator forward on x [1, F, C, H, W]; writes this chunk's K/V into buf[:, L-N:L].
        Returns flow_pred [1, F, C, H, W] (same permuted view the wrapper returns)."""
        net = self.net
        u = x.permute(0, 2, 1, 3, 4)[0]
        xs = net.patch_embedding(u.unsqueeze(0))
        xs = torch.cat([xs.flatten(2).transpose(1, 2)])
        e = net.time_embedding(self.sinus[ti].unflatten(0, (1, FPB)).type_as(xs))
        e0 = net.time_projection(e).unflatten(2, (6, net.dim))
        b, s, n, d = 1, N, self.H, self.D
        for l, blk in enumerate(self.blocks):
            ee = (blk.modulation.unsqueeze(1) + e0).chunk(6, dim=2)
            mx = adaln_modulate_triton(xs, ee[1], ee[0], TOK, eps=blk.norm1.eps, add_one_to_scale=True)
            sa = blk.self_attn
            q = sa.norm_q(sa.q(mx)).view(b, s, n, d)
            k = sa.norm_k(sa.k(mx)).view(b, s, n, d)
            v = sa.v(mx).view(b, s, n, d)
            rq = self._rope(q).type_as(v)
            rk = self._rope(k).type_as(v)
            self.K[l][:, L - N:L].copy_(rk)
            self.V[l][:, L - N:L].copy_(v)
            y = sa.o(self._fa(rq, self.K[l][:, :L], self.V[l][:, :L]).flatten(2))
            xs = xs + (y.unflatten(dim=1, sizes=(FPB, TOK)) * ee[2]).flatten(1, 2)
            ca = blk.cross_attn
            xc = blk.norm3(xs)
            xc = xc.view(1, 1, N, xs.size(2)).reshape(1, N, xs.size(2))
            qc = ca.norm_q(ca.q(xc)).view(1, -1, n, d)
            yc = ca.o(self._fa(qc, self.ck[l], self.cv[l]).flatten(2))
            xs = xs + yc.view(1, 1, N, xs.size(2)).reshape(1, N, xs.size(2))
            fin = adaln_modulate_triton(xs, ee[4], ee[3], TOK, eps=blk.norm2.eps, add_one_to_scale=True)
            y = blk.ffn(fin)
            xs = xs + (y.unflatten(dim=1, sizes=(FPB, TOK)) * ee[5]).flatten(1, 2)
        xs = net.head(xs, e.unsqueeze(2))
        out = torch.stack(net.unpatchify(xs, self.grid))
        return out.permute(0, 2, 1, 3, 4)

    def _denoise(self, L):
        sch = self.sch
        sch.reset()
        lat = self.x_in
        for i in range(self.n_steps):
            flow = self._net(lat, i, L)
            if self.rec is not None:
                self.rec.append(flow.clone())
            lat = sch.step(flow, self.t_steps[i], lat, return_dict=False)[0]
        return lat

    def _chunk(self, L):
        lat = self._denoise(L)
        self.out.copy_(lat)
        self._net(lat, self.n_steps, L)

    # ------------------------------------------------------------------ public
    def capture(self, buckets=(N, 2 * N, 3 * N, 4 * N), warmup=2):
        """One graph per attention bucket L (whole chunk: 4 forwards + 4 UniPC steps + context forward)."""
        self.graphs = graph.capture({L: (lambda L=L: self._chunk(L)) for L in buckets}, warmup=warmup)
        for t in self.K + self.V:       # capture/warm-up scribbled on the buffers; fresh session = zeros
            t.zero_()
        self.reset()

    def _prep(self, noise_chunk, prompt, embeds):
        """Host bookkeeping + static inputs for the next chunk (mirrors the official per-chunk control flow)."""
        k, cs = self.k, self.k * N
        boundary = k > 0 and isinstance(prompt, str) and prompt.startswith(self.prefix)
        scene_cut = boundary and self.multi_shot_sink and self.sink > 0
        if boundary and self.phi != 0.0:
            self.shot += 1
            self.toff = self.shot * self.phi
        if prompt != self.cur_prompt:
            self._set_prompt(embeds)
            self.cur_prompt = prompt
        cos, sin = self._rope_tables(FPB * k, self.toff)
        self.cos.copy_(cos); self.sin.copy_(sin)
        self.x_in.copy_(noise_chunk)
        L, roll = self._insert(cs)
        if roll is not None:
            self._roll(roll)
        self._insert(cs)                 # steps 2..4 (direct insert, same slot)
        if scene_cut and self.recache:   # _zero_kv_data before the context forward
            self.le, self.ge, self.ps, self.pl = self.gsink, cs, -1, 0
            Lc, rc = self._insert(cs)
            assert rc is None
        else:
            Lc = L
        return L, Lc, scene_cut

    def _post(self, scene_cut):
        if scene_cut:                    # _pin_current_chunk
            self.ps, self.pl = self.le - N, min(self.sink, N)
        self.k += 1

    @torch.no_grad()
    def gen_chunk(self, noise_chunk, prompt, embeds, graph=True):
        """Generate the next chunk. noise_chunk [1, 8, 48, 44, 80]; prompt: str (for scene-cut detection);
        embeds: this chunk's prompt_embeds [1, 512, 4096] (the same tensor the official loop uses)."""
        L, Lc, scene_cut = self._prep(noise_chunk, prompt, embeds)
        if graph and Lc == L and L in self.graphs:
            self.graphs[L].replay()
        else:
            lat = self._denoise(L)
            self.out.copy_(lat)
            self._net(lat, self.n_steps, Lc)
        self._post(scene_cut)
        return self.out

    # ------------------------------------------------------------------ session state
    META = ("k", "ge", "le", "ps", "pl", "shot", "toff", "cur_prompt")

    def live_tokens(self):
        """Tokens of the buffer the NEXT chunk can read (the evicted slot is dead)."""
        if N + self.le <= self.CS:
            return self.le
        _, eff = self._eff_sink()
        ev = N + self.le - self.CS
        return self.le - ev  # eff sink + rolled region

    def state(self):
        n = self.le
        return ({m: getattr(self, m) for m in self.META}, [t[:, :n].clone() for t in self.K + self.V])

    def load(self, st):
        meta, bufs = st
        for m, v in meta.items():
            setattr(self, m, v)
        for d, s in zip(self.K + self.V, bufs):
            d[:, :s.shape[1]].copy_(s)
        self.cur_prompt = None           # force cross K/V refresh

    def state_bytes(self, live=True):
        n = self.live_tokens() if live else self.CS
        return sum(n * t.shape[2] * t.shape[3] * t.element_size() for t in self.K + self.V)
