"""End-to-end MG2 FPS on-device (Jetson Thor): exact runtime vs the approximate fast paths.
Configs: exact = StaticMG graphs + exact Wan VAE graph; fast_trt = FastMG (fp8 + compile) + TensorRT VAE;
fast_tae = FastMG + TAEHV. Workloads: one 16-chunk playthrough, and the rewind workload (one playthrough + 6 rewinds)
served through PrefixCache (checkpoint every 2 chunks). Also writes a side-by-side mp4 (exact | fast_trt) of the
playthrough.
  python scripts/mg2/thor_e2e.py [--depth 16] [--configs exact,fast_trt,fast_tae]
"""
import argparse, gc, json, os
import torch
from fastworld import PrefixCache, write_mp4
from fastworld.models.mg2 import MG, Model, StaticMG, StaticVAE, TinyVAE, rewind_paths
from fastworld.models.mg2.fast import FastMG
from fastworld.models.mg2.trt_vae import TRTVAE
from fastworld.paths import ROOT


def make(name, M, img):
    if name == "exact":
        S = StaticMG(M); S.capture()
        V = StaticVAE()
        p = [(img, ("w", "u"))] * 2
        V.reset(); [V.decode(S.gen_chunk(p, k), graph=False) for k in range(2)]
        V.capture()
        return S, V
    S = FastMG(M, rope32=True, small_attn=True, compile=True, fp8=True); S.capture()
    return S, (TRTVAE() if name == "fast_trt" else TinyVAE())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=16)
    ap.add_argument("--configs", default="exact,fast_trt,fast_tae")
    ap.add_argument("--out", default=f"{ROOT}/results/mg2/thor/e2e.json")
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    M = Model(max_chunks=args.depth + 2)
    img = "0000"
    M.add_image(img, f"{MG}/demo_images/universal/{img}.png")
    wl = rewind_paths(img, depth=args.depth)
    res = json.load(open(args.out)) if os.path.exists(args.out) else {}
    res["gpu"] = torch.cuda.get_device_name()
    videos = {}
    for name in args.configs.split(","):
        S, V = make(name, M, img)
        PrefixCache(S, V, every=None).serve(wl[0][:3])                   # warm-up
        # straight playthrough, no sharing
        frames, st = PrefixCache(S, V, every=None).serve(wl[0])
        nfr = sum(f.shape[0] for f in frames)
        videos[name] = torch.cat(frames)
        r = {"play_fps": nfr / st["time"], "play_s": st["time"], "chunk_ms": 1e3 * st["time"] / len(wl[0])}
        # rewind workload with exact prefix sharing
        cache = PrefixCache(S, V, every=2)
        tot_fr, tot_t, ttff = 0, 0.0, []
        for i, p in enumerate(wl):
            fr, st = cache.serve(p)
            tot_fr += sum(f.shape[0] for f in fr); tot_t += st["time"]
            if i:
                ttff.append(st["done"][st["hit"]])
        r.update(rewind_fps=tot_fr / tot_t, rewind_ttff_s=sum(ttff) / len(ttff))
        res[name] = r
        print(f"{name:9s} playthrough {r['play_fps']:5.1f} FPS ({r['chunk_ms']:.0f} ms/chunk)   rewind workload "
              f"{r['rewind_fps']:5.1f} FPS   first new frame after rewind {r['rewind_ttff_s']:.2f} s", flush=True)
        json.dump(res, open(args.out, "w"), indent=1)
        del S, V, cache; gc.collect(); torch.cuda.empty_cache()
    if "exact" in videos and "fast_trt" in videos:
        n = min(len(videos["exact"]), len(videos["fast_trt"]))
        sbs = torch.cat([videos["exact"][:n], videos["fast_trt"][:n]], dim=3)          # [T,3,H,2W]
        path = f"{os.path.dirname(args.out)}/side_by_side_exact_vs_fast.mp4"
        write_mp4(path, sbs, fps=16)
        print("wrote", path)


if __name__ == "__main__":
    main()
