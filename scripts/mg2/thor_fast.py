"""Speed vs drift of the approximate MG2 fast paths (fastworld.models.mg2.fast) against the exact graph runtime.
Every variant generates the same path; its latents are decoded with the exact eager Wan VAE and compared with the exact
runtime's frames (PSNR per chunk, so drift compounding through the history is visible).
  python scripts/mg2/thor_fast.py [--chunks 12] [--variants rope32,small,both,compile,fp8]   (fp8 swaps the model's linears in place: keep it last)
"""
import argparse, gc, json, os, time
import torch
from fastworld.models.mg2 import MG, Model, StaticMG, StaticVAE, test_path
from fastworld.models.mg2.fast import FastMG
from fastworld.paths import ROOT

VARIANTS = {"rope32": dict(rope32=True, small_attn=False), "small": dict(rope32=False, small_attn=True),
            "both": dict(rope32=True, small_attn=True), "compile": dict(rope32=True, small_attn=True, compile=True),
            "fp8": dict(rope32=True, small_attn=True, compile=True, fp8=True)}


def psnr(a, b):
    mse = ((a - b) ** 2).mean().item()
    return 99.0 if mse == 0 else 10 * torch.log10(torch.tensor(4.0 / mse)).item()


def run(S, p, cap=True):
    t0 = time.time()
    if cap:
        S.capture()
    tcap = time.time() - t0
    lats, ts = [], []
    for k in range(len(p)):
        torch.cuda.synchronize(); t = time.perf_counter()
        lats.append(S.gen_chunk(p, k)); torch.cuda.synchronize(); ts.append(time.perf_counter() - t)
    return lats, ts, tcap


def decode(V, lats):
    V.reset()
    return [V.decode(l, graph=False) for l in lats]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=int, default=12)
    ap.add_argument("--variants", default="rope32,small,both,compile")
    ap.add_argument("--out", default=f"{ROOT}/results/mg2/thor/fast.json")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.set_grad_enabled(False)
    M = Model(max_chunks=args.chunks + 2)
    M.add_image("0000", f"{MG}/demo_images/universal/0000.png")
    p = test_path("0000", args.chunks)
    V = StaticVAE()
    res = json.load(open(args.out)) if os.path.exists(args.out) else {}
    res["gpu"] = torch.cuda.get_device_name()

    S = StaticMG(M)
    ref, ts, tcap = run(S, p)
    run(S, p, cap=False)                                   # second pass for warm timings
    _, ts, _ = run(S, p, cap=False)
    del S; gc.collect(); torch.cuda.empty_cache()
    fref = decode(V, ref)
    med = lambda x: sorted(x[1:])[len(x[1:]) // 2]
    res["exact"] = {"chunk_ms": med(ts) * 1e3}
    print(f"exact      chunk {med(ts)*1e3:7.1f} ms", flush=True)

    for name in args.variants.split(","):
        S = FastMG(M, **VARIANTS[name])
        lats, _, tcap = run(S, p)
        _, ts, _ = run(S, p, cap=False)
        fr = decode(V, lats)
        rel = [((a.float() - b.float()).norm() / b.float().norm()).item() for a, b in zip(lats, ref)]
        ps = [psnr(a, b) for a, b in zip(fr, fref)]
        res[name] = {"chunk_ms": med(ts) * 1e3, "capture_s": tcap, "latent_rel_err": rel, "psnr_db": ps}
        print(f"{name:10s} chunk {med(ts)*1e3:7.1f} ms ({res['exact']['chunk_ms']/med(ts)/1e3:.2f}x)  capture {tcap:5.1f}s  "
              f"PSNR first/mid/last {ps[0]:.1f}/{ps[len(ps)//2]:.1f}/{ps[-1]:.1f} dB  latent rel err last {rel[-1]:.3g}", flush=True)
        json.dump(res, open(args.out, "w"), indent=1)
        del S; gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
