"""Regenerate the frames of one benchmark workload (graph runtime, exact Wan VAE, no sharing), verify each request's
frame hash against results/bench.json (so they are the exact frames every exact mode delivered), save downscaled uint8.
  python scripts/mg2/make_frames.py [--workload rewind_0000] [--scale 0.5]
"""
import argparse, json
import numpy as np, torch
from fastworld import PrefixCache, frames_hash as h
from fastworld.models.mg2 import MG, Model, StaticMG, StaticVAE, explore_paths, rewind_paths
from fastworld.paths import ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", default="rewind_0000")
    ap.add_argument("--scale", type=float, default=0.5)
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    kind, img = args.workload.split("_")
    paths = (rewind_paths if kind == "rewind" else explore_paths)(img, depth=16)
    ref = json.load(open(f"{ROOT}/results/mg2/bench.json"))["runs"][f"{args.workload}|eager_none|wan"]
    M = Model(max_chunks=20); M.add_image(img, f"{MG}/demo_images/universal/{img}.png")
    S = StaticMG(M); S.capture()
    V = StaticVAE()
    V.reset(); lats = [S.gen_chunk(paths[0], k) for k in range(2)]
    for l in lats:
        V.decode(l, graph=False)
    V.capture()
    eng = PrefixCache(S, V, every=None)
    out = {}
    for j, p in enumerate(paths):
        frames, st = eng.serve(p)
        ok = h(frames) == ref[j]["hash"]
        print(f"request {j}: {len(p)} chunks, hash match {ok}", flush=True)
        assert ok
        chunks = []
        for f in frames:                       # [T,3,352,640] uint8
            x = torch.nn.functional.interpolate(f.float(), scale_factor=args.scale, mode="area")
            chunks.append(x.round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).numpy())
        out[f"r{j}_n"] = np.array([c.shape[0] for c in chunks])
        out[f"r{j}"] = np.concatenate(chunks)
    np.savez_compressed(f"{ROOT}/results/mg2/frames_{args.workload}.npz", **out)
    print("DONE")


if __name__ == "__main__":
    main()
