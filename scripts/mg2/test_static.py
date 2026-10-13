"""Exactness + speed of the static/CUDA-graph MG2 runtime against the eager pipeline.
  python scripts/mg2/test_static.py [--chunks 10] [--out results/mg2/static.json]
"""
import argparse, json, os
import torch
from fastworld import median as med, timed
from fastworld.models.mg2 import FPB, MG, Model, StaticMG, test_path
from fastworld.paths import ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=int, default=10)
    ap.add_argument("--images", type=int, default=2)
    ap.add_argument("--out", default=f"{ROOT}/results/mg2/static.json")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.set_grad_enabled(False)
    M = Model(max_chunks=args.chunks + 2)
    imgs = [f"{i:04d}" for i in range(args.images)]
    for n in imgs:
        M.add_image(n, f"{MG}/demo_images/universal/{n}.png")
    S = StaticMG(M)
    res = {"gpu": torch.cuda.get_device_name(), "state_bytes_static": S.state_bytes(), "state_bytes_eager": M.state_bytes()}
    paths = {img: test_path(img, args.chunks, offset=3 * i) for i, img in enumerate(imgs)}

    # eager reference (+ warm-up)
    M.reset(); [M.gen_chunk(paths[imgs[0]], k) for k in range(3)]
    ref, t_eager = {}, {}
    for img, p in paths.items():
        M.reset(); ref[img], t_eager[img] = [], []
        for k in range(len(p)):
            lat, dt = timed(lambda: M.gen_chunk(p, k)); ref[img].append(lat.clone()); t_eager[img].append(dt)

    # action-window prep vs the eager action module's own slicing (from M.cond)
    S_r, S_pad, S_ws = S.r, S.pad_t, S.ws
    ok_prep = []
    for img, p in paths.items():
        for k in range(len(p)):
            S._prep(p, k)
            c = M.cond(p, k)
            N_feats = int((c["keyboard_cond"].shape[1] - 1) / S_r) + 1
            st = S_r * (N_feats - FPB - S_ws) + S_pad
            mc = torch.cat([c["mouse_cond"][:, 0:1].expand(-1, S_pad, -1), c["mouse_cond"]], 1)[:, st:]
            gm = torch.stack([mc[:, S_r * (i - S_ws) + S_pad:i * S_r + S_pad] for i in range(FPB)], dim=1)
            kc = torch.cat([c["keyboard_cond"][:, 0:1].expand(-1, S_pad, -1), c["keyboard_cond"]], 1)[:, st:]
            ok_prep.append(bool(torch.equal(gm, S.gm)) and bool(torch.equal(kc, S.kc)) and bool(torch.equal(c["cond_concat"], S.cc)))
    res["prep_equal"] = all(ok_prep)
    print("prep equal", res["prep_equal"], ok_prep, flush=True)
    # single-forward parity at chunk 0 (no history), both images
    res["fwd0_bitwise"] = {}
    for img, p in paths.items():
        S._prep(p, 0)
        M.reset(); a = M._fwd(S.x_in.clone(), p, 0, M.steps[0])
        b = S._net(S.x_in, 0, True, False)
        res["fwd0_bitwise"][img] = (bool(torch.equal(a, b)), float((a.float() - b.float()).abs().max()))
    print("fwd0", res["fwd0_bitwise"], flush=True)

    # static, no graph
    res["nograph_bitwise"] = {}
    for img, p in paths.items():
        out = [S.gen_chunk(p, k, graph=False) for k in range(len(p))]
        res["nograph_bitwise"][img] = [bool(torch.equal(a, b)) for a, b in zip(out, ref[img])]
        res.setdefault("nograph_maxdiff", {})[img] = [float((a.float() - b.float()).abs().max()) for a, b in zip(out, ref[img])]
    print("no-graph bitwise", res["nograph_bitwise"], "maxdiff", res["nograph_maxdiff"], flush=True)

    # capture + graph
    (_, t_cap) = timed(S.capture)
    res["capture_s"] = t_cap
    res["graph_bitwise"], t_graph = {}, {}
    for img, p in paths.items():
        out, t_graph[img] = [], []
        for k in range(len(p)):
            lat, dt = timed(lambda: S.gen_chunk(p, k)); out.append(lat); t_graph[img].append(dt)
        res["graph_bitwise"][img] = [bool(torch.equal(a, b)) for a, b in zip(out, ref[img])]
    print("graph bitwise", res["graph_bitwise"], flush=True)

    # branching: save state after chunk b-1, take a different action at b, restore, continue the original path
    img = imgs[0]; p = paths[img]; b = len(p) // 2
    for k in range(b):
        S.gen_chunk(p, k)
    st = S.state()
    alt = p[:b] + [(img, ("s", "k"))]
    S.gen_chunk(alt, b)
    S.load(st)
    rest = [S.gen_chunk(p, k) for k in range(b, len(p))]
    res["branch_restore_bitwise"] = [bool(torch.equal(a, r)) for a, r in zip(rest, ref[img][b:])]
    _, res["state_save_s"] = timed(S.state)
    _, res["state_load_s"] = timed(lambda: S.load(st))
    print("branch", res["branch_restore_bitwise"], flush=True)

    te = med([t for v in t_eager.values() for t in v[1:]]); tg = med([t for v in t_graph.values() for t in v[1:]])
    res.update(t_eager_chunk=te, t_graph_chunk=tg, speedup=te / tg, t_eager=t_eager, t_graph=t_graph)

    # kernel time of one graph chunk (no-overhead floor) vs wall
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for k in range(1, 4):
            S.gen_chunk(p, k)
        torch.cuda.synchronize()
    kev = [e for e in prof.key_averages() if getattr(e, "device_type", None) == torch.autograd.DeviceType.CUDA]
    res["graph_kernel_chunk"] = sum(e.self_device_time_total for e in kev) / 3 / 1e6
    print(f"chunk: eager {te*1e3:.1f} ms  graph {tg*1e3:.1f} ms  ({te/tg:.2f}x)  graph kernel {res['graph_kernel_chunk']*1e3:.1f} ms"
          f"  capture {t_cap:.1f}s  state {res['state_bytes_static']/2**30:.2f} GiB (eager {res['state_bytes_eager']/2**30:.2f})", flush=True)
    json.dump(res, open(args.out, "w"), indent=1)
    print("DONE")


if __name__ == "__main__":
    main()
