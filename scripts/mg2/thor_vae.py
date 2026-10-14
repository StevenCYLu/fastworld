"""Decoder speed vs fidelity on real MG2 latents: exact Wan VAE (eager / CUDA graph), TensorRT FP16 engine, TAEHV.
Latents come from the exact graph runtime; every decoder's frames are compared with the exact eager decoder's.
  python scripts/mg2/thor_vae.py [--chunks 12]       (build the engine first: python -m fastworld.models.mg2.trt_vae)
"""
import argparse, json, os, time
import torch
from fastworld.models.mg2 import MG, Model, StaticMG, StaticVAE, TinyVAE, test_path
from fastworld.models.mg2.trt_vae import TRTVAE
from fastworld.paths import ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=int, default=12)
    ap.add_argument("--out", default=f"{ROOT}/results/mg2/thor/vae.json")
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    M = Model(max_chunks=args.chunks + 2)
    M.add_image("0000", f"{MG}/demo_images/universal/0000.png")
    S = StaticMG(M); S.capture()
    p = test_path("0000", args.chunks)
    lats = [S.gen_chunk(p, k) for k in range(len(p))]
    del S

    def run(dec, graph=None):
        dec.reset(); fr, ts = [], []
        for l in lats:
            torch.cuda.synchronize(); t = time.perf_counter()
            fr.append(dec.decode(l) if graph is None else dec.decode(l, graph=graph))
            torch.cuda.synchronize(); ts.append(time.perf_counter() - t)
        return fr, sorted(ts[2:])[len(ts[2:]) // 2]

    V = StaticVAE()
    ref, t_eager = run(V, graph=False)
    V.reset(); V.decode(lats[0], graph=False); V.decode(lats[1], graph=False); V.capture()
    fr_g, t_graph = run(V)
    res = {"gpu": torch.cuda.get_device_name(), "wan_eager": {"ms": t_eager * 1e3}}
    for name, dec in (("wan_graph", None), ("trt_fp16", TRTVAE()), ("taehv", TinyVAE())):
        fr, t = (fr_g, t_graph) if dec is None else run(dec)
        ps, mm = [], []
        for a, b in zip(fr, ref):
            mse = ((a.float() - b.float()) ** 2).mean().item()
            ps.append(99.0 if mse == 0 else 10 * torch.log10(torch.tensor(4.0 / mse)).item())
            q = lambda x: ((x.float() + 1) * 127.5).round().clamp(0, 255)
            mm.append((q(a) != q(b)).float().mean().item())
        res[name] = {"ms": t * 1e3, "psnr_db": ps, "u8_mismatch": mm}
        print(f"{name:9s} {t*1e3:7.1f} ms/chunk ({t_eager/t:.2f}x vs eager {t_eager*1e3:.0f} ms)  PSNR min {min(ps):.1f} "
              f"median {sorted(ps)[len(ps)//2]:.1f} dB  uint8 mismatch max {max(mm)*100:.2f}%", flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
