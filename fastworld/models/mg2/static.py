"""Static-shape, CUDA-graph runtime for Matrix-Game-2 (exact: same kernels, same values as the eager pipeline).

Eager MG2 is launch-bound at batch 1 (~3.6k launches and ~1.4k host syncs per forward, GPU ~60% busy). Every attention
in MG2 (self-attn, mouse, keyboard) sees exactly one chunk of history plus the current chunk, so from chunk 1 on all
shapes are constant. This runtime keeps per-layer history buffers (the previous chunk's context-forward K/V) instead
of the rolling caches, passes everything chunk-dependent (RoPE tables, noise, conditions) through static input buffers,
and captures a whole chunk (3 denoising forwards + scheduler + context forward) as one CUDA graph. Two graphs:
'first' (chunk 0, no history) and 'steady' (chunk >= 1).

State of a session = the history buffers (~0.65 GB, vs ~1.3 GB for the eager rolling caches) + the chunk index.
"""
import torch
from einops import rearrange
from fastworld import graph
from fastworld.models.mg2.model import CAM, FPB, KEYS, L, TOK, path_seed

TH, TW = 22, 40


def _imports():
    from wan.modules.model import sinusoidal_embedding_1d
    from wan.modules.posemb_layers import rotate_half
    import flash_attn
    return sinusoidal_embedding_1d, rotate_half, flash_attn


class StaticMG:
    def __init__(self, M):
        self.M, self.dev = M, M.dev
        net = self.net = M.p.generator.model
        self.sin1d, self.rot_half, self.fa = _imports()
        dev, bf = self.dev, torch.bfloat16
        if net.freqs.device != dev:
            net.freqs = net.freqs.to(dev)
        sch = M.p.generator.scheduler
        sch.sigmas, sch.timesteps = sch.sigmas.to(dev), sch.timesteps.to(dev)
        if M.p.scheduler is not sch:
            M.p.scheduler.sigmas, M.p.scheduler.timesteps = M.p.scheduler.sigmas.to(dev), M.p.scheduler.timesteps.to(dev)
        self.steps = [int(t) for t in M.steps]
        self.tts = [torch.full((1, FPB), t, device=dev, dtype=torch.int64) for t in self.steps + [0]]
        self.nxt = [M.steps[i + 1] * torch.ones(FPB, device=dev, dtype=torch.long) for i in range(len(self.steps) - 1)]   # float (warped) steps, as eager
        self.sin = [self.sin1d(net.freq_dim, tt.flatten()) for tt in self.tts]   # constant per step; arange+H2D is not capturable
        blocks = net.blocks
        sa0 = blocks[0].self_attn
        self.H, self.D = sa0.num_heads, sa0.head_dim
        # ---- history (= state)
        self.hk = [torch.zeros(1, L, self.H, self.D, device=dev, dtype=bf) for _ in blocks]
        self.hv = [torch.zeros_like(t) for t in self.hk]
        self.am = [b.action_model for b in blocks]
        a0 = next(a for a in self.am if a is not None)
        self.mk = {i: torch.zeros(TOK, FPB, a0.heads_num, 64, device=dev, dtype=bf) for i, a in enumerate(self.am) if a is not None}
        self.mv = {i: torch.zeros_like(t) for i, t in self.mk.items()}
        self.kk = {i: torch.zeros(1, FPB, a0.heads_num, 64, device=dev, dtype=bf) for i, a in enumerate(self.am) if a is not None}
        self.kv = {i: torch.zeros_like(t) for i, t in self.kk.items()}
        # ---- static inputs
        self.r, self.ws = a0.vae_time_compression_ratio, a0.windows_size
        self.pad_t = self.r * self.ws
        self.x_in = torch.zeros(1, 16, FPB, 44, 80, device=dev, dtype=bf)
        self.noise = [torch.zeros(FPB, 16, 44, 80, device=dev, dtype=bf) for _ in range(len(self.steps) - 1)]
        img0 = next(iter(M.images.values())) if M.images else None
        self.cc = torch.zeros(1, 20, FPB, 44, 80, device=dev, dtype=bf)
        self.vis = torch.zeros(1, 257, 1280, device=dev, dtype=bf) if img0 is None else torch.zeros_like(img0["visual_context"])
        self.gm = torch.zeros(1, FPB, self.pad_t, 2, device=dev, dtype=bf)        # grouped mouse condition
        self.kc = torch.zeros(1, FPB * self.r + self.pad_t - 3, 4, device=dev, dtype=bf)   # keyboard window (pre-embed)
        c = self.D // 2
        self.fi = torch.zeros(L, 1, c, device=dev, dtype=torch.complex128)        # main RoPE multipliers of this chunk
        self.acos = torch.zeros(1, FPB, 1, a0.freqs_cos.shape[-1], device=dev, dtype=a0.freqs_cos.dtype)
        self.asin = torch.zeros_like(self.acos)
        self.cos_full = a0.freqs_cos.to(dev); self.sin_full = a0.freqs_sin.to(dev)
        self.cu = {n: torch.tensor([0, n], dtype=torch.int32, device=dev) for n in (L, 2 * L, 257)}
        self.out = torch.zeros(1, 16, FPB, 44, 80, device=dev, dtype=bf)
        self.lat_in = torch.zeros(1, 16, FPB, 44, 80, device=dev, dtype=bf)
        self.graphs, self.cur_img = {}, None

    # ------------------------------------------------------------------ inputs
    def _prep(self, path, k, noise=True):
        """Fill the static input buffers for chunk k of `path` (same values the eager pipeline computes)."""
        M, dev = self.M, self.dev
        img_name = path[0][0]
        img = M.images[img_name]
        if self.cur_img != img_name:
            self.vis.copy_(img["visual_context"]); self.cur_img = img_name
        s = FPB * k
        self.cc.copy_(img["cond_concat"][:, :, s:s + FPB])
        # actions: only the 21-pixel-frame window the action module reads
        n_px = 1 + 4 * (FPB * (k + 1) - 1)
        lo = max(0, n_px - (self.kc.shape[1]))
        kb = torch.zeros(1, n_px - lo, 4, dtype=torch.bfloat16)
        ms = torch.zeros(1, n_px - lo, 2, dtype=torch.bfloat16)
        for j in range(k + 1):
            a, b = (0 if j == 0 else 1 + 4 * (FPB * j - 1)), 1 + 4 * (FPB * (j + 1) - 1)
            a, b = max(a, lo) - lo, b - lo
            if b > 0:
                kb[:, max(a, 0):b] = torch.tensor(KEYS[path[j][1][0]], dtype=torch.bfloat16)
                ms[:, max(a, 0):b] = torch.tensor(CAM[path[j][1][1]], dtype=torch.bfloat16)
        first_ms = torch.tensor(CAM[path[0][1][1]], dtype=torch.bfloat16).view(1, 1, 2)
        first_kb = torch.tensor(KEYS[path[0][1][0]], dtype=torch.bfloat16).view(1, 1, 4)
        r, pad_t, nf = self.r, self.pad_t, FPB
        N_feats = FPB * (k + 1)
        # mouse: cat([pad(first frame), mouse])[start:]  (start in padded coordinates)
        start = r * (N_feats - nf - self.ws) + pad_t
        full_len = n_px + pad_t
        win = []
        for p in range(start, full_len):
            win.append(first_ms[:, 0] if p < pad_t else ms[:, p - pad_t - lo])
        mc = torch.stack(win, 1)
        gm = torch.stack([mc[:, r * (i - self.ws) + pad_t:i * r + pad_t] for i in range(nf)], dim=1)
        win = []
        for p in range(start, full_len):
            win.append(first_kb[:, 0] if p < pad_t else kb[:, p - pad_t - lo])
        kc = torch.stack(win, 1)
        self.gm.copy_(gm, non_blocking=True); self.kc.copy_(kc, non_blocking=True)
        # RoPE tables
        fr = self.net.freqs.split([self.D // 2 - 2 * (self.D // 2 // 3), self.D // 2 // 3, self.D // 2 // 3], dim=1)
        f, h, w = FPB, TH, TW
        self.fi.copy_(torch.cat([fr[0][s:s + f].view(f, 1, 1, -1).expand(f, h, w, -1),
                                 fr[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                                 fr[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)], dim=-1).reshape(f * h * w, 1, -1))
        shape = [1, self.cos_full.shape[0], 1, self.cos_full.shape[1]]
        self.acos.copy_(self.cos_full.view(*shape)[:, s:s + FPB]); self.asin.copy_(self.sin_full.view(*shape)[:, s:s + FPB])
        if not noise:
            return
        # noise, drawn in the eager order
        g = torch.Generator(device=dev).manual_seed(path_seed(path[:k + 1]))
        self.x_in.copy_(torch.randn((1, 16, FPB, 44, 80), generator=g, device=dev, dtype=torch.bfloat16))
        for nb in self.noise:
            nb.copy_(torch.randn(nb.shape, generator=g, device=dev, dtype=torch.bfloat16))

    # ------------------------------------------------------------------ network
    def _rope(self, x):
        n = x.size(2)
        xi = torch.view_as_complex(x[0].to(torch.float64).reshape(x.shape[1], n, -1, 2))
        return torch.view_as_real(xi * self.fi).flatten(2)[None]

    def _arot(self, x):
        return (x.float() * self.acos + self.rot_half(x.float()) * self.asin).type_as(x)

    def _fattn(self, q, k, v):
        b, lq, lk = q.size(0), q.size(1), k.size(1)
        return self.fa.flash_attn_varlen_func(q=q.flatten(0, 1), k=k.flatten(0, 1), v=v.flatten(0, 1),
                                              cu_seqlens_q=self.cu[lq], cu_seqlens_k=self.cu[lk], max_seqlen_q=lq, max_seqlen_k=lk,
                                              dropout_p=0., softmax_scale=None, causal=False, window_size=(-1, -1),
                                              deterministic=False).unflatten(0, (b, lq)).type(q.dtype)

    def _action(self, bi, am, x, first, write):
        B, S, nf = 1, TOK, FPB
        hidden_states = rearrange(x, "B (T S) C -> (B S) T C", T=nf, S=S)
        C = self.gm.shape[-1]
        gm = self.gm.unsqueeze(-1).expand(B, nf, self.pad_t, C, S).permute(0, 4, 1, 2, 3).reshape(B * S, nf, self.pad_t * C)
        gm = am.mouse_mlp(torch.cat([hidden_states, gm], dim=-1))
        q, k, v = rearrange(am.t_qkv(gm), "B L (K H D) -> K B L H D", K=3, H=am.heads_num)
        q = am.img_attn_q_norm(q).to(v); k = am.img_attn_k_norm(k).to(v)
        q, k = self._arot(q), self._arot(k)
        if first:
            Km, Vm = k, v
        else:
            Km, Vm = torch.cat([self.mk[bi], k], 1), torch.cat([self.mv[bi], v], 1)
        attn = self.fa.flash_attn_func(q, Km, Vm)
        if write:
            self.mk[bi].copy_(k); self.mv[bi].copy_(v)
        attn = rearrange(attn, '(b S) T h d -> b (T S) (h d)', b=B)
        hidden_states = rearrange(hidden_states, "(B S) T C -> B (T S) C", B=B)
        hidden_states = hidden_states + am.proj_mouse(attn)
        r, pad_t = self.r, self.pad_t
        kc = am.keyboard_embed(self.kc)
        gk = torch.stack([kc[:, r * (i - self.ws) + pad_t:i * r + pad_t] for i in range(nf)], dim=1)
        gk = gk.reshape(gk.shape[0], gk.shape[1], -1)
        mq = am.mouse_attn_q(hidden_states); kv = am.keyboard_attn_kv(gk)
        Bq, Lq, HD = mq.shape; D = HD // am.heads_num
        q = mq.view(Bq, Lq, am.heads_num, D)
        k, v = kv.view(Bq, kv.shape[1], 2, am.heads_num, D).permute(2, 0, 1, 3, 4)
        q = am.key_attn_q_norm(q).to(v); k = am.key_attn_k_norm(k).to(v)
        T_ = Lq // S
        q = q.view(Bq, T_, S, am.heads_num, D).transpose(1, 2).reshape(Bq * S, T_, am.heads_num, D)
        q, k = self._arot(q), self._arot(k)
        if first:
            Kk, Vk = k[:1].repeat(S, 1, 1, 1), v[:1].repeat(S, 1, 1, 1)
        else:
            Kk = torch.cat([self.kk[bi], k[:1]], 1).repeat(S, 1, 1, 1)
            Vk = torch.cat([self.kv[bi], v[:1]], 1).repeat(S, 1, 1, 1)
        if write:
            self.kk[bi].copy_(k[:1]); self.kv[bi].copy_(v[:1])
        attn = self.fa.flash_attn_func(q, Kk, Vk)
        attn = rearrange(attn, '(B S) T H D -> B (T S) (H D)', S=S)
        return hidden_states + am.proj_keyboard(attn)

    def _cross(self, ca, x, context):
        b, n, d = x.size(0), ca.num_heads, ca.head_dim
        q = ca.norm_q(ca.q(x)).view(b, -1, n, d)
        k = ca.norm_k(ca.k(context)).view(b, -1, n, d)
        v = ca.v(context).view(b, -1, n, d)
        return ca.o(self._fattn(q, k, v).flatten(2))

    def _net(self, x, ti, first, write):
        net, M = self.net, self.M
        tt = self.tts[ti]
        xin = torch.cat([x, self.cc], 1)
        xs = net.patch_embedding(xin).flatten(2).transpose(1, 2)
        e = net.time_embedding(self.sin[ti].type_as(xs))
        e0 = net.time_projection(e).unflatten(1, (6, net.dim)).unflatten(dim=0, sizes=tt.shape)
        context = net.img_emb(self.vis)
        nf, fs = FPB, TOK
        for bi, blk in enumerate(net.blocks):
            ee = (blk.modulation.unsqueeze(1) + e0).chunk(6, dim=2)
            sa = blk.self_attn
            y_in = (blk.norm1(xs).unflatten(dim=1, sizes=(nf, fs)) * (1 + ee[1]) + ee[0]).flatten(1, 2)
            b, s, n, d = *y_in.shape[:2], sa.num_heads, sa.head_dim
            q = sa.norm_q(sa.q(y_in)).view(b, s, n, d)
            k = sa.norm_k(sa.k(y_in)).view(b, s, n, d)
            v = sa.v(y_in).view(b, s, n, d)
            rq = self._rope(q).type_as(v)
            rk = self._rope(k).type_as(v)
            if first:
                y = self._fattn(rq, rk, v)
            else:
                y = self._fattn(rq, torch.cat([self.hk[bi], rk], 1), torch.cat([self.hv[bi], v], 1))
            if write:
                self.hk[bi].copy_(rk); self.hv[bi].copy_(v)
            y = sa.o(y.flatten(2))
            xs = xs + (y.unflatten(dim=1, sizes=(nf, fs)) * ee[2]).flatten(1, 2)
            xs = xs + self._cross(blk.cross_attn, blk.norm3(xs.to(context.dtype)), context)
            if blk.action_model is not None:
                xs = self._action(bi, blk.action_model, xs.to(context.dtype), first, write)
            y = blk.ffn((blk.norm2(xs).unflatten(dim=1, sizes=(nf, fs)) * (1 + ee[4]) + ee[3]).flatten(1, 2))
            xs = xs + (y.unflatten(dim=1, sizes=(nf, fs)) * ee[5]).flatten(1, 2)
        out = net.head(xs, e.unflatten(dim=0, sizes=tt.shape).unsqueeze(2))
        flow = net.unpatchify(out, torch.tensor([FPB, TH, TW], dtype=torch.long))
        x0 = M.p.generator._convert_flow_pred_to_x0(flow_pred=rearrange(flow, 'b c f h w -> (b f) c h w'),
                                                     xt=rearrange(x, 'b c f h w -> (b f) c h w'), timestep=tt.flatten(0, 1))
        return rearrange(x0, '(b f) c h w -> b c f h w', b=1)

    def _chunk(self, first):
        x = self.x_in
        n = len(self.steps)
        for i in range(n):
            den = self._net(x, i, first, False)
            if i < n - 1:
                d = den.transpose(1, 2).flatten(0, 1)
                x = self.M.p.scheduler.add_noise(d, self.noise[i], self.nxt[i])
                x = x.unflatten(0, (1, FPB)).transpose(1, 2)
        self._net(den, n, first, True)
        self.out.copy_(den)

    def _prefill(self, first):
        self._net(self.lat_in, len(self.steps), first, True)

    # ------------------------------------------------------------------ public
    def capture(self, warmup=2):
        """Record the four graphs: (first chunk | steady chunk) x (generate | prefill). ~2 s on an H100."""
        fns = {(True, "gen"): lambda: self._chunk(True), (False, "gen"): lambda: self._chunk(False),
               (True, "pre"): lambda: self._prefill(True), (False, "pre"): lambda: self._prefill(False)}
        self.graphs = graph.capture(fns, warmup=warmup, save=self.state, load=self.load)

    @torch.no_grad()
    def gen_chunk(self, path, k, graph=True):
        self._prep(path, k)
        if graph:
            self.graphs[(k == 0, "gen")].replay()
        else:
            self._chunk(k == 0)
        return self.out.clone()

    @torch.no_grad()
    def prefill(self, lat, path, k, graph=True):
        """Advance the state by chunk k whose latent is already known (one context forward)."""
        self._prep(path, k, noise=False)
        self.lat_in.copy_(lat)
        if graph:
            self.graphs[(k == 0, "pre")].replay()
        else:
            self._prefill(k == 0)

    def state(self):
        return [t.clone() for t in self._state_tensors()]

    def load(self, st):
        for d, s in zip(self._state_tensors(), st):
            d.copy_(s, non_blocking=True)

    def _state_tensors(self):
        return self.hk + self.hv + list(self.mk.values()) + list(self.mv.values()) + list(self.kk.values()) + list(self.kv.values())

    def state_bytes(self):
        return graph.state_bytes(self._state_tensors())
