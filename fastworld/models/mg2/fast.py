"""Approximate (not bitwise) speed-ups of the static MG2 runtime, for devices where the exact runtime is slow (Jetson Thor).

FastMG subclasses StaticMG and keeps its buffers, state and graph capture; each trick can be turned on separately:
  rope32     : RoPE in fp32 real arithmetic instead of the official fp64 complex multiply (fp64 is slow and the
               complex128 temporaries are memory-bound on LPDDR devices).
  small_attn : the action module's attentions (query length 3, key length <= 6, batch 880) as fp32 einsums instead of
               flash-attn, whose 128-row tiles are ~97 % padding at these shapes; the keyboard keys/values are shared
               by all 880 positions, so they are broadcast instead of repeated.
  fp8        : the big DiT linears (self-attn q/k/v/o, FFN, cross-attn q/o) as FP8 e4m3 GEMMs with per-output-channel
               weight scales and per-token dynamic activation scales (torch._scaled_mm, row-wise). Swaps the modules of
               the shared model in place, so build the exact StaticMG runtimes first.
  compile    : torch.compile (Inductor, no cudagraphs) of the whole network forward, to fuse the elementwise glue
               (modulation, norms, residual adds, casts). The per-step tensors are arguments, so only the
               (first, write) variants are compiled. The CUDA graph is captured around the compiled code as before.
None of these is exact; scripts/mg2/thor_fast.py measures the drift against the exact runtime.
"""
import torch
from einops import rearrange
from fastworld.models.mg2.model import FPB, TOK
from fastworld.models.mg2.static import TH, TW, StaticMG


class Fp8Linear(torch.nn.Module):
    F8, MAX = torch.float8_e4m3fn, 448.0

    def __init__(self, lin):
        super().__init__()
        w = lin.weight.detach()
        sw = (w.abs().amax(1, keepdim=True).float() / self.MAX).clamp(min=1e-12)
        self.register_buffer("w8", (w.float() / sw).to(self.F8))
        self.register_buffer("sw", sw.t().contiguous())
        self.register_buffer("bias", None if lin.bias is None else lin.bias.detach().clone())
        self.out_features = lin.out_features

    def forward(self, x):
        x2 = x.reshape(-1, x.shape[-1])
        sa = (x2.abs().amax(1, keepdim=True).float() / self.MAX).clamp(min=1e-12)
        y = torch._scaled_mm((x2.float() / sa).to(self.F8), self.w8.t(), scale_a=sa, scale_b=self.sw, out_dtype=x.dtype)
        if self.bias is not None:
            y = y + self.bias
        return y.view(*x.shape[:-1], self.out_features)


def to_fp8(net):
    """Swap the large linears of every DiT block for Fp8Linear (in place)."""
    for blk in net.blocks:
        sa, ca = blk.self_attn, blk.cross_attn
        for mod, names in ((sa, "qkvo"), (ca, "qo")):
            for n in names:
                if not isinstance(getattr(mod, n), Fp8Linear):
                    setattr(mod, n, Fp8Linear(getattr(mod, n)))
        for i in (0, 2):
            if not isinstance(blk.ffn[i], Fp8Linear):
                blk.ffn[i] = Fp8Linear(blk.ffn[i])


class FastMG(StaticMG):
    def __init__(self, M, rope32=True, small_attn=True, compile=False, fp8=False):
        super().__init__(M)
        if fp8:
            to_fp8(self.net)
        self.rope32, self.small_attn = rope32, small_attn
        c = self.D // 2
        self.fcos = torch.zeros(self.fi.shape[0], 1, c, device=self.dev, dtype=torch.float32)
        self.fsin = torch.zeros_like(self.fcos)
        self._netc = self._net_body
        if compile:
            torch._dynamo.config.cache_size_limit = 64
            self._netc = torch.compile(self._net_body, dynamic=False)

    def _prep(self, path, k, noise=True):
        super()._prep(path, k, noise)
        self.fcos.copy_(self.fi.real); self.fsin.copy_(self.fi.imag)

    def _rope(self, x):
        if not self.rope32:
            return super()._rope(x)
        xr = x.float().unflatten(-1, (-1, 2))
        a, b = xr[..., 0], xr[..., 1]
        return torch.stack([a * self.fcos - b * self.fsin, a * self.fsin + b * self.fcos], -1).flatten(-2)

    @staticmethod
    def _sattn(q, k, v, shared=False):
        """softmax(q k^T / sqrt(d)) v in fp32. q [B,Lq,H,D]; k, v [B,Lk,H,D] or, if shared, [Lk,H,D] for every b."""
        s = torch.einsum("blhd,mhd->bhlm" if shared else "blhd,bmhd->bhlm", q.float(), k.float()) * q.shape[-1] ** -0.5
        return torch.einsum("bhlm,mhd->blhd" if shared else "bhlm,bmhd->blhd", s.softmax(-1), v.float()).to(q.dtype)

    def _action(self, bi, am, x, first, write):
        if not self.small_attn:
            return super()._action(bi, am, x, first, write)
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
        attn = self._sattn(q, Km, Vm)
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
            Kk, Vk = k[0, :1], v[0, :1]
        else:
            Kk, Vk = torch.cat([self.kk[bi], k[:1]], 1)[0], torch.cat([self.kv[bi], v[:1]], 1)[0]
        if write:
            self.kk[bi].copy_(k[:1]); self.kv[bi].copy_(v[:1])
        attn = self._sattn(q, Kk, Vk, shared=True)
        attn = rearrange(attn, '(B S) T H D -> B (T S) (H D)', S=S)
        return hidden_states + am.proj_keyboard(attn)

    def _net(self, x, ti, first, write):
        return self._netc(x, self.tts[ti], self.sin[ti], first, write)

    def _net_body(self, x, tt, sin, first, write):
        net, M = self.net, self.M
        xin = torch.cat([x, self.cc], 1)
        xs = net.patch_embedding(xin).flatten(2).transpose(1, 2)
        e = net.time_embedding(sin.type_as(xs))
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
