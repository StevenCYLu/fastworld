"""End-to-end serving benchmark: MG2 as shipped vs graph runtime vs graph + exact sharing, with exact (Wan VAE) or tiny
(TAEHV) decoding. Every request's uint8 frames are hashed and checked against the reference of its decoder family
(exact family: eager MG2 + eager Wan VAE; tiny family: graph + TAEHV without sharing).
Resumable: finished (workload, mode) entries in the JSON are skipped.
  python scripts/mg2/run_bench.py [--roots 2] [--smoke]
"""
import argparse, json, os
import numpy as np, torch
from fastworld import PrefixCache, frames_hash as h, timed
from fastworld.models.mg2 import MG, Model, StaticMG, StaticVAE, TinyVAE, explore_paths, rewind_paths
from fastworld.paths import ROOT

MODES = ["eager_none|wan", "graph_none|wan", "graph_ckpt4|wan", "graph_ckpt2|wan",
         "graph_none|tae", "graph_ckpt2|tae", "graph_ckpt1|tae"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", type=int, default=2)
    ap.add_argument("--out", default=f"{ROOT}/results/mg2/bench.json")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.out = args.out.replace(".json", "_smoke.json")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.set_grad_enabled(False)
    M = Model(max_chunks=20)
    imgs = [f"{i:04d}" for i in range(args.roots)]
    for n in imgs:
        M.add_image(n, f"{MG}/demo_images/universal/{n}.png")
    S = StaticMG(M); S.capture()
    V = StaticVAE(); T = TinyVAE()
    wl = {}
    for n in imgs:
        wl[f"explore_{n}"] = explore_paths(n, depth=8 if args.smoke else 16)
        wl[f"rewind_{n}"] = rewind_paths(n, depth=8 if args.smoke else 16)
    if args.smoke:
        wl = {k: v[:3] for k, v in list(wl.items())[:2]}
    res = json.load(open(args.out)) if os.path.exists(args.out) else {}
    res.setdefault("runs", {})
    res["gpu"] = torch.cuda.get_device_name()

    # warm-up + decoder graph capture + unit costs
    p = wl[next(iter(wl))][0]
    V.reset(); lats = [S.gen_chunk(p, k) for k in range(4)]
    for l in lats[:2]:
        V.decode(l, graph=False)
    V.capture()
    if "unit" not in res:
        u = {}
        M.reset(); [M.gen_chunk(p, k) for k in range(2)]
        u["gen_eager"] = timed(lambda: M.gen_chunk(p, 2))[1]
        S.gen_chunk(p, 0); S.gen_chunk(p, 1)
        u["gen_graph"] = timed(lambda: S.gen_chunk(p, 2))[1]
        u["prefill_graph"] = timed(lambda: S.prefill(lats[2], p, 2))[1]
        V.reset(); [V.decode(l) for l in lats[:2]]
        u["dec_wan_eager"] = timed(lambda: V.decode(lats[2], graph=False))[1]
        u["dec_wan_graph"] = timed(lambda: V.decode(lats[3]))[1]
        T.reset(); [T.decode(l) for l in lats[:3]]
        u["dec_tae"] = timed(lambda: T.decode(lats[3]))[1]
        st_d, u["save_dit"] = timed(S.state); u["load_dit"] = timed(lambda: S.load(st_d))[1]
        st_v, u["save_wan"] = timed(V.state); u["load_wan"] = timed(lambda: V.load(st_v))[1]
        st_t, u["save_tae"] = timed(T.state); u["load_tae"] = timed(lambda: T.load(st_t))[1]
        u["bytes_dit"], u["bytes_wan"], u["bytes_tae"] = S.state_bytes(), V.state_bytes(), T.state_bytes()
        res["unit"] = u
        del st_d, st_v, st_t
        print("unit (ms):", {k: (round(v * 1e3, 1) if k[:5] != "bytes" else v) for k, v in u.items()}, flush=True)

    for name, paths in wl.items():
        for mode in MODES:
            key = f"{name}|{mode}"
            if key in res["runs"]:
                continue
            run, decname = mode.split("|")
            eager = run.startswith("eager")
            every = int(run.split("ckpt")[1]) if "ckpt" in run else None
            V.use_graph = not eager
            eng = PrefixCache(M if eager else S, V if decname == "wan" else T, every)
            recs = []
            try:
                for j, pth in enumerate(paths):
                    frames, st = eng.serve(pth)
                    st["hash"] = h(frames); st["n_frames"] = sum(f.shape[0] for f in frames)
                    recs.append(st)
            except torch.OutOfMemoryError:
                print(f"{key}: OOM after {len(recs)} requests", flush=True)
                del eng; torch.cuda.empty_cache(); continue
            n_ck = eng.n_ckpt
            del eng; torch.cuda.empty_cache()
            ref = res["runs"].get(f"{name}|{'eager_none|wan' if decname == 'wan' else 'graph_none|tae'}")
            for j, st in enumerate(recs):
                st["bitwise"] = (ref is None and mode in ("eager_none|wan", "graph_none|tae")) or (ref is not None and ref[j]["hash"] == st["hash"])
                base = res["runs"].get(f"{name}|eager_none|wan")
                k = min(st["hit"], len(paths[j]) - 1)
                st["t_branch"] = st["done"][k]
                st["t_branch_base"] = base[j]["done"][k] if base else None
            res["runs"][key] = recs
            res.setdefault("n_ckpt", {})[key] = n_ck
            tot = sum(r["time"] for r in recs); nf = sum(r["n_frames"] for r in recs)
            line = (f"{name:13s} {mode:16s} n={len(recs):2d} total {tot:7.2f}s  {nf / tot:6.1f} FPS  gen {sum(r['generated'] for r in recs):3d}/"
                    f"{sum(len(q) for q in paths)} replayed {sum(r['replayed'] for r in recs):3d} ckpts {n_ck:3d} "
                    f"bitwise {sum(bool(r['bitwise']) for r in recs)}/{len(recs)}")
            br = [r for r in recs if r["hit"] > 0]
            if br and br[0]["t_branch_base"] is not None:
                line += f"  branch {np.mean([r['t_branch'] for r in br]):.2f}s vs {np.mean([r['t_branch_base'] for r in br]):.2f}s"
            print(line, flush=True)
            json.dump(res, open(args.out, "w"), indent=1)
    print("DONE")


if __name__ == "__main__":
    main()
