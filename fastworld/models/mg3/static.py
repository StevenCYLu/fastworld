"""Static-shape CUDA-graph runtime for Matrix-Game-3.0 (exact: same modules, same kernels, same values as eager).

MG3 has no KV cache: every iteration re-runs the DiT over [5 retrieved memory latents | 4 past latents | 10 new latents]
(19 latent frames = 16,720 tokens, padded by the official code to max_seq_len = 17,600). The memory count is always 5
(5 FOV queries; the official code never returns fewer), so from iteration 1 on every tensor shape is constant;
iteration 0 (15 latents, no memory) is a second shape. We capture one CUDA graph per shape ('first', 'steady'), each
containing a whole iteration: per-iteration RoPE tables + 3 DiT forwards + 3 UniPC steps + the img_cond re-injection.

What changes vs eager (all value-preserving):
  * flash_attention(): cu_seqlens built on the CPU and copied H2D every call -> cached device tensors.
  * rope_apply_with_indices(): the per-call index tensor (H2D) and table build (4x per layer) -> one table per
    iteration built inside the graph from static device index buffers (same ops -> same complex64 values).
  * sinusoidal_embedding_1d(): CPU arange + H2D per call -> cached device inverse-frequency vector (same ops).
  * ActionModule.get_rotary_pos_embed(): recomputed on every call (~50 launches) -> cached (constant).
  * pipeline timestep `new_full(.., t)` (.item() sync) -> precomputed per-step tensors (same ops).
  * UniPC step(): torch.tensor(list, device=cuda) H2D copies and a 2x2 linalg.solve (host sync) on tiny
    coefficient tensors -> computed once by the identical code and cached; the big-tensor math is unchanged.
Retrieval (select_memory_idx_fov, 5 .item() syncs), plucker construction and the noise draw stay eager host-side
work in Engine.prep(), in the same RNG order.
"""
import contextlib
import torch

from fastworld.graph import capture as graph_capture
from fastworld.models.mg3.engine import Engine  # noqa: F401  (importing the engine puts MG3_DIR on sys.path)
import wan.modules.model as wm
import wan.modules.action_module as wam
from wan.modules.attention import FLASH_ATTN_3_AVAILABLE
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

if FLASH_ATTN_3_AVAILABLE:
    import flash_attn_interface as _fa3
else:
    _fa3 = None
try:
    import flash_attn as _fa2  # only used if FA3 is unavailable
except ModuleNotFoundError:
    _fa2 = None


# ---------------------------------------------------------------------------------------------- scheduler
class CachedUniPC(FlowUniPCMultistepScheduler):
    """FlowUniPC whose tiny coefficient tensors (rks/b/rhos: H2D copies + linalg.solve) are computed once by the
    original code path and then reused, so step() is capturable. Big-tensor ops are textually the official ones."""
    CACHE = {}

    def _small(self, key, fn):
        if key not in CachedUniPC.CACHE:
            CachedUniPC.CACHE[key] = fn()
        return CachedUniPC.CACHE[key]

    def multistep_uni_p_bh_update(self, model_output, *args, sample=None, order=None, **kwargs):
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
        rks.append(1.0)
        hh = -h if self.predict_x0 else h
        h_phi_1 = torch.expm1(hh)
        h_phi_k = h_phi_1 / hh - 1
        factorial_i = 1
        B_h = torch.expm1(hh) if self.config.solver_type == "bh2" else hh
        key = ("p", self.step_index, order, x.dtype)

        def coeffs():
            nonlocal h_phi_k, factorial_i
            rks_t = torch.tensor(rks, device=device)
            R, b = [], []
            for i in range(1, order + 1):
                R.append(torch.pow(rks_t, i - 1))
                b.append(h_phi_k * factorial_i / B_h)
                factorial_i *= i + 1
                h_phi_k = h_phi_k / hh - 1 / factorial_i
            R = torch.stack(R)
            b = torch.tensor(b, device=device)
            if len(D1s) > 0:
                if order == 2:
                    return torch.tensor([0.5], dtype=x.dtype, device=device)
                return torch.linalg.solve(R[:-1, :-1], b[:-1]).to(device).to(x.dtype)
            return None
        rhos_p = self._small(key, coeffs)
        if len(D1s) > 0:
            D1s = torch.stack(D1s, dim=1)
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

    def multistep_uni_c_bh_update(self, this_model_output, *args, last_sample=None, this_sample=None, order=None, **kwargs):
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
        B_h = torch.expm1(hh) if self.config.solver_type == "bh2" else hh
        key = ("c", self.step_index, order, x.dtype)

        def coeffs():
            nonlocal h_phi_k, factorial_i
            rks_t = torch.tensor(rks, device=device)
            R, b = [], []
            for i in range(1, order + 1):
                R.append(torch.pow(rks_t, i - 1))
                b.append(h_phi_k * factorial_i / B_h)
                factorial_i *= i + 1
                h_phi_k = h_phi_k / hh - 1 / factorial_i
            R = torch.stack(R)
            b = torch.tensor(b, device=device)
            if order == 1:
                return torch.tensor([0.5], dtype=x.dtype, device=device)
            return torch.linalg.solve(R, b).to(device).to(x.dtype)
        rhos_c = self._small(key, coeffs)
        if len(D1s) > 0:
            D1s = torch.stack(D1s, dim=1)
        else:
            D1s = None
        assert self.predict_x0
        x_t_ = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0
        if D1s is not None:
            corr_res = torch.einsum("k,bkc...->bc...", rhos_c[:-1], D1s)
        else:
            corr_res = 0
        D1_t = model_t - m0
        x_t = x_t_ - alpha_t * B_h * (corr_res + rhos_c[-1] * D1_t)
        return x_t.to(x.dtype)


# ---------------------------------------------------------------------------------------------- runtime
class StaticMG3:
    def __init__(self, E):
        self.E, self.model, self.dev = E, E.model, E.device
        m = self.model
        if m.freqs.device != self.dev:
            m.freqs = m.freqs.to(self.dev)
        self.nsteps = E.args.num_inference_steps
        self.cu = {}
        self.rot_cache = {}
        self.inv_freq = {}
        self.tab = {}             # f -> complex64 RoPE table of the current iteration (graph-internal tensor)
        self.idx = {}             # mode -> {"pred": LongTensor, "mem": LongTensor}
        self.b = {}               # mode -> static input buffers
        self.out = {}
        self.graphs = {}
        self.cache_temb = True    # cache time_embedding/time_projection outputs per (mode, step) (constants)
        self.temb = {}
        self.cur = None           # (mode, step) of the forward being run

    # ---------------- patched (value-identical) helpers
    def _cu(self, lens):
        key = tuple(lens)
        if key not in self.cu:
            t = torch.tensor(list(lens), dtype=torch.int32)
            self.cu[key] = torch.cat([t.new_zeros([1]), t]).cumsum(0, dtype=torch.int32).to(self.dev)
        return self.cu[key]

    def _flash_attention(self, q, k, v, q_lens=None, k_lens=None, dropout_p=0., softmax_scale=None, q_scale=None,
                         causal=False, window_size=(-1, -1), deterministic=False, dtype=torch.bfloat16, version=None):
        half_dtypes = (torch.float16, torch.bfloat16)
        b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype
        half = lambda x: x if x.dtype in half_dtypes else x.to(dtype)
        assert q_lens is None
        q = half(q.flatten(0, 1))
        qcu = self._cu([lq] * b)
        if k_lens is None:
            k = half(k.flatten(0, 1)); v = half(v.flatten(0, 1))
            kcu = self._cu([lk] * b)
        else:
            k = half(torch.cat([u[:n] for u, n in zip(k, k_lens)]))
            v = half(torch.cat([u[:n] for u, n in zip(v, k_lens)]))
            kcu = self._cu([int(n) for n in k_lens])
        q = q.to(v.dtype); k = k.to(v.dtype)
        if q_scale is not None:
            q = q * q_scale
        if (version is None or version == 3 or version == "3") and FLASH_ATTN_3_AVAILABLE:
            x = _fa3.flash_attn_varlen_func(q=q, k=k, v=v, cu_seqlens_q=qcu, cu_seqlens_k=kcu, seqused_q=None,
                                            seqused_k=None, max_seqlen_q=lq, max_seqlen_k=lk, softmax_scale=softmax_scale,
                                            causal=causal, deterministic=deterministic).unflatten(0, (b, lq))
        else:
            x = _fa2.flash_attn_varlen_func(q=q, k=k, v=v, cu_seqlens_q=qcu, cu_seqlens_k=kcu, max_seqlen_q=lq,
                                            max_seqlen_k=lk, dropout_p=dropout_p, softmax_scale=softmax_scale,
                                            causal=causal, window_size=window_size,
                                            deterministic=deterministic).unflatten(0, (b, lq))
        return x.type(out_dtype)

    def _sinusoidal(self, dim, position):
        assert dim % 2 == 0
        half = dim // 2
        position = position.type(torch.float64)
        key = (dim, position.device)
        if key not in self.inv_freq:
            self.inv_freq[key] = torch.pow(10000, -torch.arange(half).to(position).div(half))
        sinusoid = torch.outer(position, self.inv_freq[key])
        return torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)

    @staticmethod
    def _table(freqs, n, c, f, h, w, t_idx):
        """freqs_i of rope_apply_with_indices (3-D per-head freqs, sigma_theta > 0), already .to(complex64)."""
        if freqs.dim() == 3:
            fr = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=2)
        else:
            fr = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
        if fr[0].dim() == 3:
            t_freqs = fr[0][:, t_idx, :]
            h_freqs = fr[1][:, :h, :]
            w_freqs = fr[2][:, :w, :]
            fi = torch.cat([
                t_freqs.permute(1, 0, 2).view(f, 1, 1, n, -1).expand(f, h, w, n, -1),
                h_freqs.permute(1, 0, 2).view(1, h, 1, n, -1).expand(f, h, w, n, -1),
                w_freqs.permute(1, 0, 2).view(1, 1, w, n, -1).expand(f, h, w, n, -1),
            ], dim=-1).reshape(f * h * w, n, -1)
        else:
            fi = torch.cat([
                fr[0][t_idx].view(f, 1, 1, -1).expand(f, h, w, -1),
                fr[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                fr[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ], dim=-1).reshape(f * h * w, 1, -1)
        return fi.to(torch.complex64)

    def _rope(self, x, grid_sizes, freqs, t_indices=None):
        n = x.size(2)
        output = []
        for i, (f, h, w) in enumerate(grid_sizes.tolist()):
            seq_len = f * h * w
            x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float32).reshape(seq_len, n, -1, 2))
            x_i = torch.view_as_real(x_i * self.tab[f]).flatten(2)
            x_i = torch.cat([x_i, x[i, seq_len:]])
            output.append(x_i)
        return torch.stack(output).float()

    def _rot_embed(self_static):
        orig = wam.ActionModule.get_rotary_pos_embed

        def cached(self, video_length, height, width, head_dim, rope_dim_list=None):
            key = (id(self), video_length, height, width, head_dim, tuple(rope_dim_list) if rope_dim_list else None)
            if key not in self_static.rot_cache:
                self_static.rot_cache[key] = orig(self, video_length, height, width, head_dim, rope_dim_list)
            return self_static.rot_cache[key]
        return cached

    def _temb_wrap(self, mod, name):
        """time_embedding / time_projection depend only on the timestep vector, which for a given (mode, step) is the
        same every iteration (t for the 10+4 frame window, 0 for memory/img_cond/padding): cache the output computed
        by the identical module call the first time, return it afterwards. (17600 x 3072 -> 17600 x 18432 fp32 GEMMs.)"""
        orig = mod.forward

        def fwd(x):
            if self.cur is None or not self.cache_temb:
                return orig(x)
            key = (self.cur, name)
            if key not in self.temb:
                self.temb[key] = orig(x)
            return self.temb[key]
        return fwd

    @contextlib.contextmanager
    def patched(self):
        m = self.model
        saved = (wm.flash_attention, wm.rope_apply_with_indices, wm.sinusoidal_embedding_1d,
                 wam.ActionModule.get_rotary_pos_embed)
        wm.flash_attention = self._flash_attention
        wm.rope_apply_with_indices = self._rope
        wm.sinusoidal_embedding_1d = self._sinusoidal
        wam.ActionModule.get_rotary_pos_embed = self._rot_embed()
        m.time_embedding.forward = self._temb_wrap(m.time_embedding, "te")
        m.time_projection.forward = self._temb_wrap(m.time_projection, "tp")
        try:
            yield
        finally:
            (wm.flash_attention, wm.rope_apply_with_indices, wm.sinusoidal_embedding_1d,
             wam.ActionModule.get_rotary_pos_embed) = saved
            del m.time_embedding.forward, m.time_projection.forward
            self.cur = None

    # ---------------- static inputs
    def _mode(self, it):
        return "first" if it["first"] else "steady"

    def _alloc(self, it):
        mode = self._mode(it)
        if mode in self.b:
            return mode
        c = it["conds"]
        b = {"lat0": torch.empty_like(it["latents"]), "img_cond": torch.empty_like(it["img_cond"]),
             "mouse_cond": torch.empty_like(c["mouse_cond"]), "keyboard_cond": torch.empty_like(c["keyboard_cond"]),
             "plucker_emb": torch.empty_like(c["plucker_emb"]),
             "context": [torch.empty_like(u) for u in c["context"]]}
        if not it["first"]:
            b["x_memory"] = torch.empty_like(c["x_memory"])
            b["timestep_memory"] = c["timestep_memory"].clone()          # constant zeros
            b["keyboard_cond_memory"] = c["keyboard_cond_memory"].clone()  # constant -1
            b["mouse_cond_memory"] = c["mouse_cond_memory"].clone()        # constant 1
        # per-step timestep tensors, built by the eager ops (values depend only on t and shapes)
        sch = FlowUniPCMultistepScheduler()
        sch.set_timesteps(self.nsteps, device=self.dev, shift=self.E.args.sample_shift)
        lat, ic = it["latents"], it["img_cond"]
        ts = []
        for t in sch.timesteps:
            timestep = lat.new_full((lat.shape[2], lat.shape[3] * lat.shape[4] // 4), t)
            timestep[:ic.shape[2]].zero_()
            ts.append(timestep.flatten().unsqueeze(0))
        b["ts"] = ts
        ls, le = c["predict_latent_idx"]
        self.idx[mode] = {"pred": torch.zeros(le - ls, dtype=torch.long, device=self.dev)}
        if not it["first"]:
            self.idx[mode]["mem"] = torch.zeros(len(c["memory_latent_idx"]), dtype=torch.long, device=self.dev)
        self.b[mode] = b
        self.out[mode] = torch.empty_like(it["latents"])
        return mode

    def fill(self, it):
        mode = self._alloc(it)
        b, c = self.b[mode], it["conds"]
        b["lat0"].copy_(it["latents"]); b["img_cond"].copy_(it["img_cond"])
        for key in ("mouse_cond", "keyboard_cond", "plucker_emb"):
            b[key].copy_(c[key])
        for d, s in zip(b["context"], c["context"]):
            d.copy_(s)
        ls, le = c["predict_latent_idx"]
        # index buffers: tiny pinned-free H2D of host ints (the eager code makes the same tensors inside every rope call)
        self.idx[mode]["pred"].copy_(torch.arange(ls, le, dtype=torch.long))
        if not it["first"]:
            b["x_memory"].copy_(c["x_memory"])
            self.idx[mode]["mem"].copy_(torch.tensor(c["memory_latent_idx"], dtype=torch.long))
        return mode

    # ---------------- one whole iteration (captured)
    def _iter(self, mode):
        E, m, b = self.E, self.model, self.b[mode]
        first = mode == "first"
        freqs = m.freqs
        n, c = m.num_heads, m.dim // m.num_heads // 2
        hh, ww = E.lat_h // 2, E.lat_w // 2
        npred = self.idx[mode]["pred"].numel()
        self.tab = {npred: self._table(freqs, n, c, npred, hh, ww, self.idx[mode]["pred"])}
        if not first:
            nm = self.idx[mode]["mem"].numel()
            self.tab[nm] = self._table(freqs, n, c, nm, hh, ww, self.idx[mode]["mem"])
        sch = CachedUniPC()
        # timesteps stay on the CPU (no H2D inside the capture); with begin_index=0 step() only stores them
        # (eager's index_for_timestep() finds index 0 for the first, unique timestep -> same step indices)
        sch.set_timesteps(self.nsteps, device=None, shift=E.args.sample_shift)
        sch.set_begin_index(0)
        conds = dict(mouse_cond=b["mouse_cond"], keyboard_cond=b["keyboard_cond"], context=b["context"],
                     plucker_emb=b["plucker_emb"], x_memory=b.get("x_memory"), timestep_memory=b.get("timestep_memory"),
                     keyboard_cond_memory=b.get("keyboard_cond_memory"), mouse_cond_memory=b.get("mouse_cond_memory"),
                     memory_latent_idx=None if first else list(range(self.idx[mode]["mem"].numel())),
                     predict_latent_idx=None, fa_version=E.p.fa_version)
        latents, img_cond = b["lat0"], b["img_cond"]
        for i in range(self.nsteps):
            t = sch.timesteps[i]
            self.cur = (mode, i) if mode == "steady" else None   # iteration 0 runs once: not worth 4.5 GB of cache
            noise_pred = m(x=latents, t=b["ts"][i], seq_len=E.max_seq_len, **conds)
            self.cur = None
            latents = sch.step(noise_pred, t, latents, return_dict=False)[0]
            latents = torch.cat([img_cond, latents[:, :, img_cond.shape[2]:]], dim=2)
        self.out[mode].copy_(latents)

    def _iter_patched(self, mode):
        with self.patched():
            self._iter(mode)

    def capture(self, its, warmup=2):
        """its: prepared iterations (Engine.prep) of each mode to capture, e.g. [it0, it1] for 'first' and 'steady'.
        _iter reads/writes only the static buffers (no session state), so no save/restore is needed around capture."""
        modes = [self.fill(it) for it in its]
        self.graphs = graph_capture({m: (lambda m=m: self._iter_patched(m)) for m in modes}, warmup=warmup)

    @torch.no_grad()
    def denoise(self, it, graph=True):
        mode = self.fill(it)
        if graph:
            self.graphs[mode].replay()
        else:
            with self.patched():
                self._iter(mode)
        return self.out[mode]
