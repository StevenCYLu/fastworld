"""Roofline analysis of Matrix-Game-2 on the current GPU: where each kernel class of the DiT and the VAE decoder sits
against the device's compute and bandwidth roofs, and how far the exact and fast runtimes are from the bound.

Three phases, each writes into results/mg2/thor/roofline.json (re-runnable, phases are independent):
  census  : FLOPs and minimum DRAM bytes per operation class for one steady DiT forward and one VAE decoder step,
            from an ATen-level dispatch census of the exact runtime (every op's input + output bytes; views cost 0).
            Attention FLOPs are derived from the flash-attn call shapes. Does not depend on timing: run it any time.
  roofs   : measured ceilings (BF16 / FP8 GEMM at model shapes and square, DRAM copy/add). Needs an idle GPU.
  attn    : MG2's self-attention shape on every attention backend (flash-attn 2, cuDNN / flash / mem-efficient SDPA).
  time    : measured kernel time per class (torch.profiler on the exact DiT forward, the exact VAE step, the FastMG
            fp8+compile forward and the TensorRT VAE step). Needs an idle GPU.
  plot    : results/mg2/thor/roofline.png (roofline) and roofline_time.png (measured vs bound per class).
  python scripts/mg2/roofline.py census roofs attn time plot
"""
import argparse, collections, json, os, re, time
import torch
from fastworld.paths import ROOT

OUT = f"{ROOT}/results/mg2/thor/roofline.json"
VIEW_OPS = {"view", "t", "unsqueeze", "permute", "unbind", "_unsafe_view", "alias", "slice", "select", "split", "expand",
            "transpose", "squeeze", "as_strided", "detach", "lift_fresh", "_local_scalar_dense", "narrow", "reshape"}
GEMM_OPS = {"addmm", "mm", "bmm", "baddbmm", "_scaled_mm"}
CONV_OPS = {"convolution", "_convolution", "cudnn_convolution"}


def load():
    return json.load(open(OUT)) if os.path.exists(OUT) else {}


def save(r):
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(r, open(OUT, "w"), indent=1)


# ----------------------------------------------------------------------------------------------------------- census
class Census(torch.utils._python_dispatch.TorchDispatchMode):
    """Per-class FLOPs, bytes (sum of distinct input + output tensor bytes), op counts. `tag` overrides the class."""

    def __init__(self):
        super().__init__()
        self.flops, self.bytes, self.count, self.elems = (collections.Counter() for _ in range(4))
        self.tag = None

    def classify(self, name, args):
        base = name.split(".")[-1] if name.startswith("aten.") else name
        if "flash_attn" in name:
            return "attention"
        if base in GEMM_OPS:
            return "gemm"
        if base in CONV_OPS:
            return "conv"
        if base in VIEW_OPS:
            return None
        if self.tag:
            return self.tag
        if base == "_to_copy":
            return "cast"
        if base in ("cat", "stack", "repeat", "clone", "copy_", "contiguous"):
            return "copy"
        if base in ("native_layer_norm", "pow", "mean", "rsqrt", "layer_norm", "rms_norm", "linalg_vector_norm"):
            return "norm"
        return "elementwise"

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        name = str(func.overloadpacket) if hasattr(func, "overloadpacket") else str(func)
        cls = self.classify(name, args)
        if cls is None:
            return out
        seen, b = set(), 0
        for t in torch.utils._pytree.tree_leaves((args, kwargs, out)):
            if torch.is_tensor(t) and t.is_cuda and t.data_ptr() not in seen:
                seen.add(t.data_ptr()); b += t.numel() * t.element_size()
        self.bytes[cls] += b; self.count[cls] += 1
        outs = [t for t in torch.utils._pytree.tree_leaves(out) if torch.is_tensor(t) and t.is_cuda]
        self.elems[cls] += sum(t.numel() for t in outs)
        base = name.split(".")[-1]
        if cls == "gemm":
            a, bm = (args[1], args[2]) if base in ("addmm", "baddbmm") else (args[0], args[1])
            self.flops[cls] += 2 * a.numel() * bm.shape[-1]
        elif cls == "conv":
            w = args[1]; o = outs[0]
            self.flops[cls] += 2 * o.numel() * (w.numel() // w.shape[0])
        elif cls == "attention":
            q, k = args[0], args[1]
            if q.dim() == 3:                       # varlen: [tokens, H, D], one sequence
                lq, lk, h, d = q.shape[0], k.shape[0], q.shape[1], q.shape[2]; bsz = 1
            else:                                  # [B, L, H, D]
                bsz, lq, h, d = q.shape; lk = k.shape[1]
            self.flops[cls] += 4 * bsz * lq * lk * h * d
        else:
            self.flops[cls] += sum(t.numel() for t in outs)      # ~1 flop per output element
        return out

    def table(self):
        return {c: {"flops": self.flops[c], "bytes": self.bytes[c], "ops": self.count[c], "out_elems": self.elems[c]}
                for c in sorted(self.bytes)}


CUR = [None]      # the census currently recording; the tagged methods set its tag


def tagged(obj, meth, tag):
    orig = getattr(obj, meth)

    def f(*a, **k):
        c = CUR[0]
        if c is None:
            return orig(*a, **k)
        prev, c.tag = c.tag, tag
        try:
            return orig(*a, **k)
        finally:
            c.tag = prev
    setattr(obj, meth, f)


def census(r):
    from fastworld.models.mg2 import MG, Model, StaticMG, StaticVAE, test_path
    M = Model(max_chunks=6); M.add_image("0000", f"{MG}/demo_images/universal/0000.png")
    S = StaticMG(M); p = test_path("0000", 4)
    for k in range(2):
        S.gen_chunk(p, k, graph=False)
    S._prep(p, 2)
    tagged(S, "_rope", "rope"); tagged(S, "_arot", "rope")
    c = CUR[0] = Census()
    with c:
        S._net(S.x_in, 1, False, False)            # one steady denoising forward (reads the history, no write)
    r["dit_forward"] = c.table()
    c = CUR[0] = Census()
    with c:
        S._chunk(False)                            # whole steady chunk = 3 denoising forwards + context forward
    r["dit_chunk"] = c.table()
    CUR[0] = None
    V = StaticVAE(); V.reset()
    lats = [S.gen_chunk(p, k, graph=False) for k in range(3)]
    V.decode(lats[0], graph=False); V.decode(lats[1], graph=False)
    c = Census()
    with c:
        V.decode(lats[2], graph=False)
    r["vae_step"] = c.table()
    r["shapes"] = {"tokens_per_chunk": 2640, "context_tokens": 5280, "dim": 1536, "heads": 12, "head_dim": 128,
                   "ffn": 8960, "layers": 30, "action_layers": 15, "forwards_per_chunk": 4}
    for k in ("dit_forward", "dit_chunk", "vae_step"):
        t = r[k]; tf = sum(v["flops"] for v in t.values()); tb = sum(v["bytes"] for v in t.values())
        print(f"{k}: {tf/1e9:.0f} GFLOP, {tb/1e9:.2f} GB min traffic")
        for cls, v in sorted(t.items(), key=lambda x: -x[1]["bytes"]):
            print(f"   {cls:12s} {v['flops']/1e9:8.1f} GF {v['bytes']/1e9:7.3f} GB  I={v['flops']/max(v['bytes'],1):8.1f} F/B  ops={v['ops']}")


# ------------------------------------------------------------------------------------------------------------ roofs
def bench(f, n=10):
    for _ in range(3):
        f()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n):
        f()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / n


def roofs(r):
    o = {}
    bf = torch.bfloat16
    for n in (4096, 8192):
        a, b = torch.randn(n, n, device="cuda", dtype=bf), torch.randn(n, n, device="cuda", dtype=bf)
        o[f"bf16_gemm_sq{n}_tflops"] = 2 * n ** 3 / bench(lambda: a @ b) / 1e12
    M = 2640
    best = 0
    for K, N in ((1536, 1536), (1536, 8960), (8960, 1536)):
        a, w = torch.randn(M, K, device="cuda", dtype=bf), torch.randn(N, K, device="cuda", dtype=bf)
        t = bench(lambda: a @ w.t()); o[f"bf16_gemm_{M}x{K}x{N}_tflops"] = 2 * M * K * N / t / 1e12
        best = max(best, o[f"bf16_gemm_{M}x{K}x{N}_tflops"])
        a8, w8 = a.to(torch.float8_e4m3fn), w.to(torch.float8_e4m3fn); one = torch.ones((), device="cuda")
        t = bench(lambda: torch._scaled_mm(a8, w8.t(), scale_a=one, scale_b=one, out_dtype=bf))
        o[f"fp8_gemm_{M}x{K}x{N}_tflops"] = 2 * M * K * N / t / 1e12
    n = 8192
    a8 = torch.randn(n, n, device="cuda").to(torch.float8_e4m3fn); w8 = torch.randn(n, n, device="cuda").to(torch.float8_e4m3fn)
    one = torch.ones((), device="cuda")
    o["fp8_gemm_sq8192_tflops"] = 2 * n ** 3 / bench(lambda: torch._scaled_mm(a8, w8.t(), scale_a=one, scale_b=one, out_dtype=bf)) / 1e12
    x = torch.empty(2 ** 29, device="cuda", dtype=bf); y = torch.empty_like(x)
    o["dram_copy_gbs"] = 4 * x.numel() / bench(lambda: y.copy_(x)) / 1e9
    o["dram_add_gbs"] = 4 * x.numel() / bench(lambda: torch.add(x, x, out=y)) / 1e9
    o["bf16_peak_tflops"] = max(v for k, v in o.items() if k.startswith("bf16"))
    o["fp8_peak_tflops"] = max(v for k, v in o.items() if k.startswith("fp8"))
    o["dram_gbs"] = max(o["dram_copy_gbs"], o["dram_add_gbs"])
    o["ridge_flop_per_byte"] = o["bf16_peak_tflops"] * 1e12 / (o["dram_gbs"] * 1e9)
    r["roofs"] = o
    print(json.dumps(o, indent=1))


# ------------------------------------------------------------------------------------------------------------- time
KCLASS = [("attention", r"flash_fwd|fmha|attention"), ("conv", r"implicit_gemm|conv|nchwToNhwc|nhwcToNchw|cudnn"),
          ("gemm", r"nvjet|gemm|cutlass|Gemm|sgemm|xmma|matmul|scaled_mm"),
          ("fused", r"^triton_"), ("norm", r"layer_norm|rms_norm|ReduceOp|reduce_kernel|NormTwo"),
          ("cast", r"copy_kernel|_to_copy|direct_copy"), ("copy", r"CatArray|cat|repeat|clone|Fill"), ("elementwise", r".")]


def kernel_times(fn, iters=3):
    from torch.profiler import profile, ProfilerActivity
    fn(); torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as pr:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    out = collections.Counter(); names = collections.defaultdict(collections.Counter)
    for e in pr.key_averages():
        if e.device_time_total == 0:
            continue
        cls = next(c for c, pat in KCLASS if re.search(pat, e.key))
        out[cls] += e.device_time_total / 1e3 / iters
        names[cls][e.key[:60]] += e.device_time_total / 1e3 / iters
    return dict(out), {c: dict(n.most_common(4)) for c, n in names.items()}


def event_timed(obj, meth, store):
    """Wrap a method so each call records a CUDA event pair; store[0] accumulates the elapsed ms after a sync."""
    orig = getattr(obj, meth)

    def f(*a, **k):
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
        e0.record(); out = orig(*a, **k); e1.record(); store.append((e0, e1))
        return out
    setattr(obj, meth, f)


def timing(r):
    import gc
    from fastworld.models.mg2 import MG, Model, StaticMG, StaticVAE, test_path
    from fastworld.models.mg2.fast import FastMG
    from fastworld.models.mg2.trt_vae import TRTVAE
    M = Model(max_chunks=6); M.add_image("0000", f"{MG}/demo_images/universal/0000.png")
    p = test_path("0000", 4)
    t = r.setdefault("time_ms", {}); top = r.setdefault("time_top_kernels", {})
    S = StaticMG(M)
    for k in range(2):
        S.gen_chunk(p, k, graph=False)
    S._prep(p, 2)
    t["dit_exact_forward"], top["dit_exact_forward"] = kernel_times(lambda: S._net(S.x_in, 1, False, False))
    ev = []; event_timed(S, "_rope", ev); event_timed(S, "_arot", ev)
    S._net(S.x_in, 1, False, False); ev.clear()
    S._net(S.x_in, 1, False, False); torch.cuda.synchronize()
    rope = sum(a.elapsed_time(b) for a, b in ev)       # fp64 main RoPE + fp32 action RoPE, both generic elementwise kernels
    t["dit_exact_forward"]["rope"] = rope; t["dit_exact_forward"]["elementwise"] -= rope
    V = StaticVAE(); V.reset(); lats = [S.gen_chunk(p, k, graph=False) for k in range(3)]
    V.decode(lats[0], graph=False); V.decode(lats[1], graph=False); st = V.state()
    def vae_step():
        V.load(st); V.decode(lats[2], graph=False)
    t["vae_exact_step"], top["vae_exact_step"] = kernel_times(vae_step)
    del S; gc.collect(); torch.cuda.empty_cache()
    T = TRTVAE(); T.reset(); T.decode(lats[0]); T.decode(lats[1]); st = T.state()
    def trt_step():
        T.load(st); T.decode(lats[2])
    t["vae_trt_step"], top["vae_trt_step"] = kernel_times(trt_step)
    del T, V; gc.collect(); torch.cuda.empty_cache()
    F = FastMG(M, rope32=True, small_attn=True, compile=True, fp8=True)
    for k in range(2):
        F.gen_chunk(p, k, graph=False)
    F._prep(p, 2)
    t["dit_fast_forward"], top["dit_fast_forward"] = kernel_times(lambda: F._net(F.x_in, 1, False, False))
    for k, v in t.items():
        print(f"{k:18s} total {sum(v.values()):7.1f} ms  " + "  ".join(f"{c} {ms:.0f}" for c, ms in sorted(v.items(), key=lambda x: -x[1])))


# ------------------------------------------------------------------------------------------------------------- plot
def attn(r):
    """MG2's main self-attention shape (2640 queries x 5280 keys, 12 heads, d=128) on every available backend."""
    import torch.nn.functional as F
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from flash_attn import flash_attn_func
    bf = torch.bfloat16
    Lq, Lk, H, D = 2640, 5280, 12, 128
    q = torch.randn(1, Lq, H, D, device="cuda", dtype=bf); k = torch.randn(1, Lk, H, D, device="cuda", dtype=bf)
    v = torch.randn_like(k)
    fl = 4 * Lq * Lk * H * D
    ref = flash_attn_func(q, k, v)
    o = {"flash_attn_2": fl / bench(lambda: flash_attn_func(q, k, v), 20) / 1e12}
    qt, kt, vt = (x.transpose(1, 2) for x in (q, k, v))
    for name, be in (("cudnn_sdpa", SDPBackend.CUDNN_ATTENTION), ("torch_flash_sdpa", SDPBackend.FLASH_ATTENTION),
                     ("mem_efficient_sdpa", SDPBackend.EFFICIENT_ATTENTION)):
        try:
            with sdpa_kernel(be):
                out = F.scaled_dot_product_attention(qt, kt, vt)
                o[name] = fl / bench(lambda: F.scaled_dot_product_attention(qt, kt, vt), 20) / 1e12
            o[name + "_maxdiff_vs_fa2"] = (out.transpose(1, 2) - ref).abs().max().item()
        except Exception as e:
            o[name] = None; o[name + "_error"] = str(e).splitlines()[0][:200]
    r["attention_tflops"] = o
    print(json.dumps(o, indent=1))


def _merge(d):
    """Census and profiler split copies differently (the profiler sees every direct_copy as one kernel): merge them."""
    out = {k: v for k, v in d.items() if k not in ("cast", "copy")}
    parts = [d[k] for k in ("cast", "copy") if k in d]
    if parts:
        out["copy & cast"] = ({kk: sum(p[kk] for p in parts) for kk in parts[0]} if isinstance(parts[0], dict) else sum(parts))
    return out


def plot(r):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rf = r["roofs"]; peak, fp8, bw = rf["bf16_peak_tflops"] * 1e12, rf["fp8_peak_tflops"] * 1e12, rf["dram_gbs"] * 1e9
    ink, muted, grid = "#1f2328", "#59636e", "#d0d7de"
    # categorical slots in fixed order; gemm and conv share slot 1 (a panel never shows both)
    cat = {"gemm": "#2a78d6", "conv": "#2a78d6", "attention": "#eb6834", "rope": "#1baf7a", "elementwise": "#eda100",
           "norm": "#e87ba4", "copy & cast": "#6250d6"}
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": grid, "axes.labelcolor": ink, "xtick.color": muted,
                         "ytick.color": muted})
    gpu = r.get("gpu", "GPU")

    # ---------------------------------------------------------------- roofline
    fig, ax = plt.subplots(figsize=(9.5, 6.2), dpi=150)
    xs = [2 ** i for i in range(-5, 15)]
    for name, cap, ls in (("BF16 GEMM", peak, "-"), ("FP8 GEMM", fp8, "-")):
        ax.plot(xs, [min(cap, bw * x) for x in xs], color=ink if name.startswith("BF16") else muted, lw=1.2, ls=ls)
        ax.text(xs[-1], cap * (1.08 if name.startswith("FP8") else 0.92), f"{name} {cap/1e12:.0f} TFLOP/s", ha="right",
                va="bottom" if name.startswith("FP8") else "top", color=ink if name.startswith("BF16") else muted, fontsize=9)
    ax.text(3, bw * 3 * 1.45, f"DRAM {bw/1e9:.0f} GB/s", rotation=36, color=ink, fontsize=9, rotation_mode="anchor")
    ridge = peak / bw
    ax.axvline(ridge, color=grid, lw=0.8); ax.text(ridge * 0.9, 7e9, f"ridge\n{ridge:.0f} FLOP/B", color=muted, fontsize=8, ha="right")
    # label offsets (points) for the crowded ones
    off = {"DiT rope": (-8, -3), "DiT elementwise": (-8, -2), "DiT norm": (6, 6), "DiT copy & cast": (-8, -10),
           "VAE elementwise": (8, 2), "VAE copy & cast": (8, -8), "DiT gemm": (-8, -12), "DiT gemm (fast)": (8, -3),
           "DiT attention (fast)": (-10, 2), "DiT attention": (8, -3), "VAE conv": (8, -10), "self-attn, cuDNN SDPA": (-12, 10)}
    pts = []
    for key, tkey, label, mk in (("dit_forward", "dit_exact_forward", "DiT", "o"), ("vae_step", "vae_exact_step", "VAE", "s")):
        tab, tm = _merge(r[key]), _merge(r["time_ms"][tkey])
        for cls, v in tab.items():
            ms = tm.get(cls, 0)
            if ms <= 0.5 or v["bytes"] == 0:
                continue
            I = v["flops"] / v["bytes"]; perf = v["flops"] / (ms / 1e3)
            ax.scatter(I, perf, s=60, color=cat.get(cls, muted), marker=mk, edgecolor="white", lw=1.5, zorder=3)
            pts.append((I, perf, f"{label} {cls}"))
    tf = r["time_ms"].get("dit_fast_forward", {})
    for cls in ("gemm", "attention"):
        v, ms = r["dit_forward"][cls], tf.get(cls, 0)
        if ms > 0:
            I = v["flops"] / (v["bytes"] * (0.6 if cls == "gemm" else 1))      # FP8 roughly halves the weight bytes
            ax.scatter(I, v["flops"] / (ms / 1e3), s=60, facecolor="white", edgecolor=cat[cls], lw=1.8, marker="o", zorder=3)
            pts.append((I, v["flops"] / (ms / 1e3), f"DiT {cls} (fast)"))
    at = r.get("attention_tflops", {})
    if at.get("cudnn_sdpa"):
        v = r["dit_forward"]["attention"]; I = v["flops"] / v["bytes"]
        ax.scatter(I, at["cudnn_sdpa"] * 1e12, s=60, facecolor="white", edgecolor=cat["attention"], lw=1.8, marker="D", zorder=3)
        pts.append((I, at["cudnn_sdpa"] * 1e12, "self-attn, cuDNN SDPA"))
    for I, perf, lab in pts:
        ax.annotate(lab, (I, perf), xytext=off.get(lab, (6, 4)), textcoords="offset points", fontsize=8, color=ink,
                    ha="right" if off.get(lab, (6,))[0] < 0 else "left")
    ax.set(xscale="log", yscale="log", xlim=(0.04, 2 ** 14), ylim=(5e9, 5e14),
           xlabel="arithmetic intensity (FLOP per byte, unfused op-by-op traffic)", ylabel="attained throughput (FLOP/s)")
    ax.set_title(f"Matrix-Game-2 roofline on {gpu}\n", loc="left", fontsize=11, color=ink)
    ax.text(0, 1.015, "filled = exact runtime, hollow = fast runtime, diamond = cuDNN attention microbenchmark;  "
            "circles = DiT forward, squares = VAE decoder step", transform=ax.transAxes, fontsize=8, color=muted, va="bottom")
    ax.grid(True, which="major", color=grid, lw=0.5); ax.grid(False, which="minor")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{ROOT}/results/mg2/thor/roofline.png"); plt.close(fig)

    # ---------------------------------------------------------------- time vs estimate
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), dpi=150, gridspec_kw={"width_ratios": [1, 1]})
    panels = (("dit_forward", "dit_exact_forward", "dit_fast_forward", "DiT, one steady forward", "FP8 + compile"),
              ("vae_step", "vae_exact_step", "vae_trt_step", "VAE decoder, one chunk", "TensorRT FP16"))
    for ax, (key, tex, tfast, title, fastname) in zip(axes, panels):
        tab = _merge(r[key]); tm, tf = _merge(r["time_ms"][tex]), r["time_ms"][tfast]
        est = {c: max(tab[c]["flops"] / peak, tab[c]["bytes"] / bw) * 1e3 for c in tab}
        classes = [c for c in ("gemm", "conv", "attention", "rope", "elementwise", "norm", "copy & cast") if c in tab and est[c] > 1]
        meas = [tm.get(c, 0) for c in classes]; e = [est[c] for c in classes]; n = len(classes)
        ax.barh([i + 0.2 for i in range(n)], meas, 0.38, color=[cat[c] for c in classes], label="measured (exact runtime)")
        ax.barh([i - 0.2 for i in range(n)], e, 0.38, color="white", edgecolor=[cat[c] for c in classes], lw=1.2,
                label="roofline estimate, unfused")
        ax.barh(n + 0.2, sum(tm.values()), 0.38, color=ink); ax.barh(n - 0.2, sum(e), 0.38, color="white", edgecolor=ink, lw=1.2)
        ax.barh(n + 1, sum(tf.values()), 0.38, color=muted)
        # fully fused floor: only the math-heavy classes remain, at peak
        floor = sum(tab[c]["flops"] for c in ("gemm", "conv", "attention") if c in tab) / peak * 1e3
        ax.axvline(floor, color=ink, lw=1)
        ax.text(floor, -0.75, f" fully-fused floor {floor:.0f} ms", fontsize=8, color=ink, va="center")
        ax.set_yticks(list(range(n + 2))); ax.set_yticklabels(classes + ["total, exact", f"total, fast"], fontsize=9)
        for i, m in enumerate(meas):
            ax.text(m + 3, i + 0.2, f"{m:.0f} ms", va="center", fontsize=8, color=ink)
        ax.text(sum(tm.values()) + 3, n + 0.2, f"{sum(tm.values()):.0f} ms", va="center", fontsize=8, color=ink)
        ax.text(sum(e) + 3, n - 0.2, f"estimate {sum(e):.0f} ms", va="center", fontsize=8, color=muted)
        ax.text(sum(tf.values()) + 3, n + 1, f"{sum(tf.values()):.0f} ms ({fastname})", va="center", fontsize=8, color=ink)
        ax.set_xlim(0, max(sum(tm.values()), sum(e)) * 1.35); ax.set_ylim(n + 1.6, -1.2)
        ax.set_title(title, fontsize=10, color=ink, loc="left"); ax.set_xlabel("GPU kernel time (ms)")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(True, axis="x", color=grid, lw=0.5)
    axes[0].legend(loc="upper right", fontsize=8, frameon=False)
    fig.suptitle(f"{gpu}: measured kernel time vs roofline estimate (max of FLOPs / BF16 peak, unfused bytes / DRAM bandwidth)",
                 fontsize=10, color=ink, x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(f"{ROOT}/results/mg2/thor/roofline_time.png"); plt.close(fig)
    print("wrote results/mg2/thor/roofline.png, roofline_time.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("phases", nargs="+", choices=["census", "roofs", "attn", "time", "plot"])
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    r = load(); r["gpu"] = torch.cuda.get_device_name()
    for ph in args.phases:
        {"census": census, "roofs": roofs, "attn": attn, "time": timing, "plot": plot}[ph](r)
        save(r)
