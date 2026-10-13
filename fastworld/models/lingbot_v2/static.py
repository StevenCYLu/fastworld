"""Static-shape, CUDA-graph runtime for LingBot-World-v2-1.3B-causal-fast (exact: same kernels, same values as eager).

Eager KV cache (per layer, [1, 18 frames * 1508 tok, 12, 128]) with a 6-frame sink: chunk c (4 latent frames) writes its
keys at local_end, attends to cache[:local_end]; when full, the non-sink part is shifted left in place (with `.item()`
bookkeeping). Keys are stored already RoPE'd at their absolute frame index; queries use the absolute index of the
current chunk, so the shift moves roped keys without re-roping. Layout and key order are therefore fully determined by
the chunk index, and we reproduce them with static per-layer buffers of the same size:
  plan(c) = (write_off, key_len, post_roll)  -- computed on the host by simulating the eager bookkeeping
  chunk 0..3: key_len 4/8/12/16 frames; chunk 3 ends with a shift [8:16]->[6:14]; chunk >=4 (steady): key_len 18,
  write at [14:18], ends with a shift [10:18]->[6:14].
(The shift that eager does at the start of chunk c+1 we do at the end of chunk c, inside the graph, as non-overlapping
copies; so the session state is only the live prefix, 14 frames in steady state.) One CUDA graph per distinct plan
(5 graphs): [camera features] + 4 x (DiT forward, x0, add_noise) + KV-update forward + shift.

Everything chunk-dependent enters through static buffers filled outside the graph: latent noise slice, image-condition
slice, Plücker slice, RoPE multiplier table, the 3 add_noise noises (drawn with the session generator in eager order).
Constants precomputed once (same ops as eager): time embeddings per step, sigmas, cross-attn K/V of the prompt.
Optional `hoist=True`: the camera branch (patch_embedding_wancamctrl + per-block cam_injector/scale/shift) depends only
on the chunk's camera, not on the timestep, so it is computed once per chunk instead of 5x (bitwise identical).
"""
import torch
import torch.nn.functional as torch_F
from einops import rearrange
import flash_attn

from fastworld.graph import capture as _capture
from fastworld.models.lingbot_v2.engine import LOCAL_ATTN, SINK


def eager_plans(n, C, W=LOCAL_ATTN, S=SINK):
    """Simulate CausalWanSelfAttention's index bookkeeping (in frames). Returns per chunk (pre_roll, write_off, key_len)."""
    local_end = global_end = 0
    out = []
    for c in range(n):
        cur_end = (c + 1) * C
        roll = None
        if W != -1 and cur_end > global_end and C + local_end > W:
            ev = C + local_end - W
            rolled = local_end - ev - S
            roll = (S + ev, S, rolled)            # src, dst, n frames
            local_end = local_end + cur_end - global_end - ev
        else:
            local_end = local_end + cur_end - global_end
        out.append((roll, local_end - C, local_end))
        global_end = cur_end
    return out


class StaticLingbot:
    def __init__(self, pipe, I, hoist=True, max_chunks=4096):
        self.pipe, self.net, self.hoist = pipe, pipe.model, hoist
        net, dev = self.net, pipe.device
        self.dev, bf = dev, torch.bfloat16
        if net.freqs.device != dev:
            net.freqs = net.freqs.to(dev)
        self.C, self.lh, self.lw = I.chunk_size, I.lat_h, I.lat_w
        self.F = I.frame_seqlen
        self.L = self.C * self.F
        self.grid = [self.C, self.lh // 2, self.lw // 2]
        blocks = net.blocks
        self.H, self.D = blocks[0].self_attn.num_heads, blocks[0].self_attn.head_dim
        nframes = LOCAL_ATTN
        # ---- state: per-layer K/V buffers (eager kv_cache layout)
        self.K = [torch.zeros(1, nframes * self.F, self.H, self.D, device=dev, dtype=bf) for _ in blocks]
        self.V = [torch.zeros_like(t) for t in self.K]
        self.live = 0                       # live prefix (frames) after the last chunk's post-roll
        pl = eager_plans(max_chunks + 1, self.C)
        self.plans = [(pl[c][1], pl[c][2], pl[c + 1][0]) for c in range(max_chunks)]
        # ---- static inputs
        self.x_in = torch.zeros(16, self.C, self.lh, self.lw, device=dev, dtype=torch.float32)
        self.y_in = torch.zeros(I.y.shape[0], self.C, self.lh, self.lw, device=dev, dtype=I.y.dtype)
        self.pl_in = torch.zeros(1, I.plucker.shape[1], self.C, self.lh, self.lw, device=dev, dtype=I.plucker.dtype)
        self.fi = torch.zeros(self.L, 1, self.D // 2, device=dev, dtype=torch.complex128)
        self.nsteps = len(I.timesteps)
        self.noise = [torch.zeros_like(self.x_in) for _ in range(self.nsteps - 1)]
        self.out = torch.zeros_like(self.x_in)
        self.cu = {}
        # ---- prompt-dependent constants (cross-attn K/V)
        self.xk = [torch.zeros(1, 512, self.H, self.D, device=dev, dtype=bf) for _ in blocks]
        self.xv = [torch.zeros_like(t) for t in self.xk]
        # ---- hoisted camera features
        if hoist:
            self.cam1 = [torch.zeros(1, self.L, net.dim, device=dev, dtype=bf) for _ in blocks]   # 1 + cam_scale
            self.camb = [torch.zeros_like(t) for t in self.cam1]                                   # cam_shift
        self.graphs, self.prompt = {}, None
        self._consts(I)
        self.set_session(I)

    # ------------------------------------------------------------------ constants
    @torch.no_grad()
    def _consts(self, I):
        """Per-step time embeddings and scheduler coefficients, computed with eager's exact ops."""
        pipe, net, dev = self.pipe, self.net, self.dev
        from wan.modules.model import sinusoidal_embedding_1d
        sch = pipe.scheduler
        ts = list(I.timesteps) + [I.timesteps[-1] * 0.0]
        seq_lens = torch.tensor([self.L], dtype=torch.long)
        self.e, self.e0 = [], []
        with torch.amp.autocast('cuda', dtype=pipe.param_dtype):
            for t in ts:
                t = torch.stack([t]).to(dev)
                if t.dim() == 1:
                    t = t.expand(t.size(0), seq_lens)
                with torch.amp.autocast('cuda', dtype=torch.float32):
                    bt = t.size(0)
                    t = t.flatten()
                    e = net.time_embedding(sinusoidal_embedding_1d(net.freq_dim, t).unflatten(0, (bt, seq_lens)).float())
                    e0 = net.time_projection(e).unflatten(2, (6, net.dim))
                self.e.append(e); self.e0.append(e0)
        # _convert_flow_pred_to_x0: sigma_t = sigmas.double()[argmin |timesteps - t|]
        sig64, tts = sch.sigmas.double().to(dev), sch.timesteps.double().to(dev)
        self.sig_x0 = []
        for t in I.timesteps:
            tid = torch.argmin((tts - t.double().to(dev)).abs())
            self.sig_x0.append(sig64[tid].reshape(-1, 1, 1, 1))
        # add_noise(x0, noise, t_next): sigma = sigmas.float()[index_for_timestep(t_next)]
        sig32, sched_ts = sch.sigmas.to(device=dev, dtype=torch.float32), sch.timesteps.to(dev)
        self.an_alpha, self.an_sigma = [], []
        for i in range(1, len(I.timesteps)):
            idx = sch.index_for_timestep(I.timesteps[i].to(dev), sched_ts)
            s = sig32[[idx]].flatten()
            while len(s.shape) < 4:
                s = s.unsqueeze(-1)
            a, s = sch._sigma_to_alpha_sigma_t(s)
            self.an_alpha.append(a); self.an_sigma.append(s)

    @torch.no_grad()
    def set_prompt(self, context):
        net = self.net
        with torch.amp.autocast('cuda', dtype=self.pipe.param_dtype):
            ctx = net.text_embedding(torch.stack([torch.cat([u, u.new_zeros(net.text_len - u.size(0), u.size(1))])
                                                  for u in context]))
            for bi, blk in enumerate(net.blocks):
                ca = blk.cross_attn
                b, n, d = ctx.size(0), ca.num_heads, ca.head_dim
                self.xk[bi].copy_(ca.norm_k(ca.k(ctx)).view(b, -1, n, d))
                self.xv[bi].copy_(ca.v(ctx).view(b, -1, n, d))

    def set_session(self, I):
        self.I = I
        if self.prompt is not I.prompt:
            self.set_prompt(I.context); self.prompt = I.prompt
        self.g = torch.Generator(device=self.dev)
        self.g.set_state(I.g_state)
        self.reset()

    def reset(self):
        for t in self.K + self.V:
            t.zero_()
        self.live = 0
        self.g.set_state(self.I.g_state)

    # ------------------------------------------------------------------ inputs
    def _prep(self, k, cam=None, noise=True):
        I, C = self.I, self.C
        s = C * k
        self.x_in.copy_(I.noise[:, s:s + C])
        self.y_in.copy_(I.y[:, s:s + C])
        self.pl_in.copy_(I.plucker[:, :, s:s + C] if cam is None else cam)
        c = self.D // 2
        fr = self.net.freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
        f, h, w = self.grid
        self.fi.copy_(torch.cat([fr[0][s:s + f].view(f, 1, 1, -1).expand(f, h, w, -1),
                                 fr[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                                 fr[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)], dim=-1).reshape(f * h * w, 1, -1))
        if noise:
            for nb in self.noise:
                nb.copy_(torch.randn(nb.shape, generator=self.g, device=self.dev, dtype=torch.float32))

    # ------------------------------------------------------------------ network
    def _cu(self, n):
        if n not in self.cu:
            self.cu[n] = torch.tensor([0, n], dtype=torch.int32, device=self.dev)
        return self.cu[n]

    def _fa(self, q, k, v):
        b, lq, lk = q.size(0), q.size(1), k.size(1)
        return flash_attn.flash_attn_varlen_func(
            q=q.flatten(0, 1), k=k.flatten(0, 1), v=v.flatten(0, 1), cu_seqlens_q=self._cu(lq), cu_seqlens_k=self._cu(lk),
            max_seqlen_q=lq, max_seqlen_k=lk, dropout_p=0., softmax_scale=None, causal=False, window_size=(-1, -1),
            deterministic=False).unflatten(0, (b, lq)).type(q.dtype)

    def _rope(self, x):
        s, n = x.size(1), x.size(2)
        xi = torch.view_as_complex(x[0, :s].to(torch.float64).reshape(s, n, -1, 2))
        return torch.view_as_real(xi * self.fi).flatten(2)[None]

    def _cam_emb(self):
        net = self.net
        p = net.patch_size
        c = torch.cat([rearrange(self.pl_in, '1 c (f c1) (h c2) (w c3) -> 1 (f h w) (c c1 c2 c3)',
                                 c1=p[0], c2=p[1], c3=p[2])], dim=1)
        c = net.patch_embedding_wancamctrl(c)
        hs = net.c2ws_hidden_states_layer2(torch_F.silu(net.c2ws_hidden_states_layer1(c)))
        return c + hs

    @staticmethod
    def _cam_block(blk, emb):
        h = blk.cam_injector_layer2(torch_F.silu(blk.cam_injector_layer1(emb)))
        h = h + emb
        return blk.cam_scale_layer(h), blk.cam_shift_layer(h)

    def _hoist_cam(self):
        emb = self._cam_emb()
        for bi, blk in enumerate(self.net.blocks):
            sc, sh = self._cam_block(blk, emb)
            self.cam1[bi].copy_(1.0 + sc); self.camb[bi].copy_(sh)

    def _net(self, x, si, woff, klen, head=True):
        net = self.net
        xs = net.patch_embedding(torch.cat([x, self.y_in], dim=0).unsqueeze(0))
        xs = torch.cat([xs.flatten(2).transpose(1, 2)])
        e, e0 = self.e[si], self.e0[si]
        emb = None if self.hoist else self._cam_emb()
        F_ = self.F
        for bi, blk in enumerate(net.blocks):
            with torch.amp.autocast('cuda', dtype=torch.float32):
                ee = (blk.modulation.unsqueeze(0) + e0).chunk(6, dim=2)
            # ---- self-attention against the static KV buffer
            sa = blk.self_attn
            yin = blk.norm1(xs).float() * (1 + ee[1].squeeze(2)) + ee[0].squeeze(2)
            b, s, n, d = *yin.shape[:2], sa.num_heads, sa.head_dim
            q = sa.norm_q(sa.q(yin)).view(b, s, n, d)
            k = sa.norm_k(sa.k(yin)).view(b, s, n, d)
            v = sa.v(yin).view(b, s, n, d)
            rq = self._rope(q).type_as(v)
            rk = self._rope(k).type_as(v)
            K, V = self.K[bi], self.V[bi]
            K[:, woff * F_:woff * F_ + s] = rk
            V[:, woff * F_:woff * F_ + s] = v
            y = sa.o(self._fa(rq, K[:, :klen * F_], V[:, :klen * F_]).flatten(2))
            with torch.amp.autocast('cuda', dtype=torch.float32):
                xs = xs + y * ee[2].squeeze(2)
            # ---- camera injection
            if self.hoist:
                xs = self.cam1[bi] * xs + self.camb[bi]
            else:
                sc, sh = self._cam_block(blk, emb)
                xs = (1.0 + sc) * xs + sh
            # ---- cross-attention (prompt K/V precomputed) + ffn
            ca = blk.cross_attn
            xq = blk.norm3(xs)
            qc = ca.norm_q(ca.q(xq)).view(xq.size(0), -1, ca.num_heads, ca.head_dim)
            xs = xs + ca.o(self._fa(qc, self.xk[bi], self.xv[bi]).flatten(2))
            y = blk.ffn(blk.norm2(xs).float() * (1 + ee[4].squeeze(2)) + ee[3].squeeze(2))
            with torch.amp.autocast('cuda', dtype=torch.float32):
                xs = xs + y * ee[5].squeeze(2)
        if not head:
            return None
        out = net.head(xs, e)
        c = net.out_dim
        u = out[0][:self.L].view(*self.grid, *net.patch_size, c)
        u = torch.einsum('fhwpqrc->cfphqwr', u)
        u = u.reshape(c, *[i * j for i, j in zip(self.grid, net.patch_size)])
        return u.float()

    def _roll(self, roll):
        src, dst, n = roll
        F_, sh = self.F, src - dst
        for i in range(0, n, sh):
            m = min(sh, n - i)
            for B in (self.K, self.V):
                for t in B:
                    t[:, (dst + i) * F_:(dst + i + m) * F_].copy_(t[:, (src + i) * F_:(src + i + m) * F_])

    def _chunk(self, plan):
        woff, klen, roll = plan
        with torch.amp.autocast('cuda', dtype=self.pipe.param_dtype):
            if self.hoist:
                self._hoist_cam()
            x = self.x_in
            for i in range(self.nsteps):
                flow = self._net(x, i, woff, klen)
                x0 = (x.double() - self.sig_x0[i] * flow.double()).to(torch.float32)
                if i < self.nsteps - 1:
                    x = self.an_alpha[i] * x0 + self.an_sigma[i] * self.noise[i]
            self.out.copy_(x0)
            self._net(x0, self.nsteps, woff, klen, head=False)
            if roll is not None:
                self._roll(roll)

    # ------------------------------------------------------------------ public
    def keys(self):
        return sorted(set(self.plans[:16]), key=lambda p: (p[1], p[0]))

    def capture(self, warmup=1):
        """One CUDA graph per distinct plan (5), shared pool; warm-up + capture run the chunk body, so the session is
        reset afterwards (callers start a session with set_session/reset anyway)."""
        keys = self.keys()
        for n in [self.L, 512] + [p[1] * self.F for p in keys]:
            self._cu(n)          # device cu_seqlens must exist before capture (no allocation inside the graph)
        self.graphs = _capture({key: (lambda key=key: self._chunk(key)) for key in keys}, warmup=warmup)
        self.reset()

    @torch.no_grad()
    def gen_chunk(self, k, graph=True, cam=None):
        self._prep(k, cam=cam)
        plan = self.plans[k]
        if graph:
            self.graphs[plan].replay()
        else:
            self._chunk(plan)
        self.live = plan[1] if plan[2] is None else plan[2][1] + plan[2][2]
        return self.out.clone()

    # session state = live prefix of the KV buffers + generator state
    def state(self):
        n = self.live * self.F
        return dict(kv=[t[:, :n].clone() for t in self.K + self.V], live=self.live, g=self.g.get_state())

    def load(self, st):
        n = st["live"] * self.F
        for d, s in zip(self.K + self.V, st["kv"]):
            d[:, :n].copy_(s, non_blocking=True)
        self.live = st["live"]
        self.g.set_state(st["g"])

    def state_bytes(self, live=None):
        live = self.live if live is None else live
        return sum(t[:, :live * self.F].numel() * t.element_size() for t in self.K + self.V)
