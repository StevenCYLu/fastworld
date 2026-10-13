"""Exactness + speed of the static/CUDA-graph MG3 runtime against the official eager loop, plus the eager profile.
  python scripts/mg3/test_static.py --iters 8 --out results/mg3/static_h100.json [--profile 1] [--decode 1]
(run from MG3_DIR, e.g. via scripts/mg3/job.sh; a relative --out is taken relative to the repo root)
"""
import argparse, json, os, time
import torch
from fastworld import median as med, timed as sync_time
from fastworld.models.mg3 import FIRST, Engine, StaticMG3
from fastworld.paths import ROOT


def dump(res, path):
    json.dump(res, open(path, "w"), indent=1)


def kernel_profile(fn, n=1):
    """CUDA kernel time (DeviceType.CUDA events only), launch and sync counts for fn() run n times."""
    from torch.profiler import profile, ProfilerActivity
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
    ka = prof.key_averages()
    kern = sum(e.self_device_time_total for e in ka if getattr(e, "device_type", None) == torch.autograd.DeviceType.CUDA) / 1e6
    names = {e.key: e.count for e in ka}
    launches = sum(c for k, c in names.items() if k in ("cudaLaunchKernel", "cuLaunchKernel", "cudaLaunchKernelExC",
                                                        "cuLaunchKernelEx", "cudaLaunchCooperativeKernel"))
    syncs = sum(c for k, c in names.items() if k in ("cudaStreamSynchronize", "cudaDeviceSynchronize", "cudaEventSynchronize"))
    memcpy = sum(c for k, c in names.items() if k in ("cudaMemcpyAsync", "cudaMemcpy"))
    graph_launch = names.get("cudaGraphLaunch", 0)
    top = sorted([e for e in ka if getattr(e, "device_type", None) == torch.autograd.DeviceType.CUDA],
                 key=lambda e: -e.self_device_time_total)[:40]
    return dict(kernel_s=kern / n, wall_profiled_s=wall / n, launches=launches / n, syncs=syncs / n, memcpy=memcpy / n,
                graph_launches=graph_launch / n,
                top_kernels=[(e.key[:90], e.self_device_time_total / 1e3 / n, e.count / n) for e in top])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--int8", type=int, default=1)
    ap.add_argument("--profile", type=int, default=1)
    ap.add_argument("--decode", type=int, default=1)
    ap.add_argument("--out", default=f"{ROOT}/results/mg3/static.json")
    a = ap.parse_args()
    if not os.path.isabs(a.out):
        a.out = os.path.join(ROOT, a.out)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    torch.set_grad_enabled(False)
    t0 = time.time()
    E = Engine(use_int8=bool(a.int8), num_iterations=a.iters)
    res = {"gpu": torch.cuda.get_device_name(), "iters": a.iters, "int8": a.int8, "load_s": time.time() - t0,
           "torch": torch.__version__}
    E.start(); E.free_t5()
    S = StaticMG3(E)
    N = a.iters

    # ---------------- eager reference x2 (warm-up = run 0), timed per piece
    runs = []
    for r in range(2):
        E.start()
        out, tp, td, tr = [], [], [], []
        for k in range(N):
            it, dp = sync_time(lambda: E.prep(k))
            lat, dd = sync_time(lambda: E.denoise(it))
            E.commit(it, lat)
            out.append(lat.clone()); tp.append(dp); td.append(dd); tr.append(it["t_retrieval"])
            if r == 1:
                res.setdefault("memory_index", []).append(it["selected_index"])
        runs.append(out)
        res[f"eager_run{r}"] = dict(prep=tp, denoise=td, retrieval=tr)
        print(f"eager run {r}: denoise {[round(x, 3) for x in td]} prep {[round(x, 4) for x in tp]}", flush=True)
    ref = runs[1]
    res["eager_vs_eager_bitwise"] = [bool(torch.equal(x, y)) for x, y in zip(runs[0], runs[1])]
    print("eager vs eager", res["eager_vs_eager_bitwise"], "memory idx", res["memory_index"], flush=True)
    del runs
    sb = E.state_bytes()
    res["state_bytes_eager"] = {"latent_history+img_cond": sb[0], "vae_stream_cache": sb[1]}
    dump(res, a.out)

    # ---------------- eager profile of one steady iteration (k = N-1 prepared again on a fresh rollout)
    if a.profile:
        E.start()
        for k in range(N - 1):
            it = E.prep(k); E.commit(it, ref[k])
        itp = E.prep(N - 1)
        p = kernel_profile(lambda: E.denoise(itp))
        wall = med(res["eager_run1"]["denoise"][1:])
        p["wall_unprofiled_s"] = wall
        p["gpu_busy"] = p["kernel_s"] / wall
        res["eager_profile_steady"] = p
        pp = kernel_profile(lambda: E.prep(N - 1))   # host prep (incl. retrieval) of a steady iteration
        res["eager_profile_prep"] = {k: v for k, v in pp.items() if k != "top_kernels"}
        print("eager profile:", {k: v for k, v in p.items() if k != "top_kernels"}, flush=True)
        for row in p["top_kernels"]:
            print("   ", row, flush=True)
        # ablation on the same prepared steady iteration (denoise does not mutate session state)
        abl = {}
        for name, fn in (("eager", lambda: E.denoise(itp)),
                         ("static_nograph_no_temb_cache", lambda: (setattr(S, "cache_temb", False), S.denoise(itp, graph=False))),
                         ("static_nograph", lambda: (setattr(S, "cache_temb", True), S.denoise(itp, graph=False)))):
            fn()
            abl[name] = med([sync_time(fn)[1] for _ in range(3)])
        res["ablation_steady_s"] = abl
        print("ablation", abl, flush=True)
        dump(res, a.out)

    # ---------------- single-forward parity (steady iteration, step 0)
    E.start()
    for k in range(N - 1):
        it = E.prep(k); E.commit(it, ref[k])
    it = E.prep(N - 1)
    S.fill(it)
    mode = "steady"
    from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
    sch = FlowUniPCMultistepScheduler(); sch.set_timesteps(3, device=E.device, shift=E.args.sample_shift)
    lat = it["latents"]
    ts0 = lat.new_full((lat.shape[2], lat.shape[3] * lat.shape[4] // 4), sch.timesteps[0])
    ts0[:it["img_cond"].shape[2]].zero_(); ts0 = ts0.flatten().unsqueeze(0)
    fa = E.model(x=it["latents"], t=ts0, seq_len=E.max_seq_len, **it["conds"])
    with S.patched():
        m = E.model
        n, c = m.num_heads, m.dim // m.num_heads // 2
        S.tab = {14: S._table(m.freqs, n, c, 14, E.lat_h // 2, E.lat_w // 2, S.idx[mode]["pred"]),
                 5: S._table(m.freqs, n, c, 5, E.lat_h // 2, E.lat_w // 2, S.idx[mode]["mem"])}
        b = S.b[mode]
        fb = m(x=b["lat0"], t=b["ts"][0], seq_len=E.max_seq_len, mouse_cond=b["mouse_cond"], keyboard_cond=b["keyboard_cond"],
               context=b["context"], plucker_emb=b["plucker_emb"], x_memory=b["x_memory"], timestep_memory=b["timestep_memory"],
               keyboard_cond_memory=b["keyboard_cond_memory"], mouse_cond_memory=b["mouse_cond_memory"],
               memory_latent_idx=[0] * 5, predict_latent_idx=None, fa_version=E.p.fa_version)
    res["timestep_equal"] = bool(torch.equal(ts0, b["ts"][0]))
    res["fwd_steady_bitwise"] = bool(torch.equal(fa, fb))
    res["fwd_steady_maxdiff"] = float((fa.float() - fb.float()).abs().max())
    print("fwd parity", res["timestep_equal"], res["fwd_steady_bitwise"], res["fwd_steady_maxdiff"], flush=True)
    del fa, fb

    # ---------------- static, no graph
    E.start()
    out, tng = [], []
    for k in range(N):
        it = E.prep(k)
        lat, dt = sync_time(lambda: S.denoise(it, graph=False).clone())
        E.commit(it, lat); out.append(lat); tng.append(dt)
    res["nograph_denoise"] = tng
    res["nograph_bitwise"] = [bool(torch.equal(x, y)) for x, y in zip(out, ref)]
    res["nograph_maxdiff"] = [float((x.float() - y.float()).abs().max()) for x, y in zip(out, ref)]
    print("no-graph bitwise", res["nograph_bitwise"], res["nograph_maxdiff"], flush=True)
    dump(res, a.out)

    # ---------------- capture (first + steady), from a fresh rollout's prepared iterations
    E.start()
    it0 = E.prep(0); E.commit(it0, ref[0]); it1 = E.prep(1)
    torch.cuda.synchronize(); t0 = time.time()
    S.capture([it0, it1])
    res["capture_s"] = time.time() - t0
    del it0, it1
    print("captured in", res["capture_s"], flush=True)

    # ---------------- graph rollout
    E.start()
    out, tg, tgp = [], [], []
    for k in range(N):
        it, dp = sync_time(lambda: E.prep(k))
        lat, dd = sync_time(lambda: S.denoise(it))
        lat = lat.clone()
        E.commit(it, lat); out.append(lat); tg.append(dd); tgp.append(dp)
    res["graph_bitwise"] = [bool(torch.equal(x, y)) for x, y in zip(out, ref)]
    res["graph_maxdiff"] = [float((x.float() - y.float()).abs().max()) for x, y in zip(out, ref)]
    res["graph_run"] = dict(prep=tgp, denoise=tg)
    print("graph bitwise", res["graph_bitwise"], "denoise", [round(x, 3) for x in tg], flush=True)
    dump(res, a.out)

    # ---------------- graph kernel profile of a steady iteration
    if a.profile:
        E.start()
        for k in range(N - 1):
            it = E.prep(k); E.commit(it, ref[k])
        itp = E.prep(N - 1)
        p = kernel_profile(lambda: S.denoise(itp))
        p["wall_unprofiled_s"] = med(tg[1:])
        p["gpu_busy"] = p["kernel_s"] / p["wall_unprofiled_s"]
        res["graph_profile_steady"] = p
        print("graph profile:", {k: v for k, v in p.items() if k != "top_kernels"}, flush=True)

    # ---------------- branch: save state at b, take a different action at b, restore, continue the original path
    def save():
        return dict(lat=[t.clone() for t in E.all_latents], img=E.img_cond.clone(), gen=E.generator.get_state(), k=E.k,
                    vae=[t.clone() if torch.is_tensor(t) else t for t in E.vae_cache])
    def load(st):
        E.all_latents = [t.clone() for t in st["lat"]]; E.img_cond = st["img"].clone()
        E.generator.set_state(st["gen"]); E.k = st["k"]; E.vae_cache = list(st["vae"])
    E.start()
    bpt = N // 2
    for k in range(bpt):
        it = E.prep(k); E.commit(it, S.denoise(it).clone())
    st = save()
    s0, s1 = FIRST + bpt * 40 - 56, FIRST + bpt * 40
    kb_save, ms_save = E.kb_all[:, s0:s1].clone(), E.ms_all[:, s0:s1].clone()
    E.kb_all[:, s0:s1] = torch.flip(kb_save, dims=[1]); E.ms_all[:, s0:s1] = -ms_save
    it = E.prep(bpt); alt = S.denoise(it).clone(); E.commit(it, alt)
    E.kb_all[:, s0:s1] = kb_save; E.ms_all[:, s0:s1] = ms_save
    load(st)
    rest = []
    for k in range(bpt, N):
        it = E.prep(k); lat = S.denoise(it).clone(); E.commit(it, lat); rest.append(lat)
    res["state_bytes_static"] = E.state_bytes()[0]   # after N iterations, same point as the eager measurement
    res["branch_differs"] = not bool(torch.equal(alt, ref[bpt]))
    res["branch_restore_bitwise"] = [bool(torch.equal(x, y)) for x, y in zip(rest, ref[bpt:])]
    _, res["state_save_s"] = sync_time(save)
    _, res["state_load_s"] = sync_time(lambda: load(st))
    print("branch", res["branch_differs"], res["branch_restore_bitwise"], flush=True)
    res["temb_cache_bytes"] = sum(t.numel() * t.element_size() for t in S.temb.values())
    res["max_mem_allocated_GB"] = torch.cuda.max_memory_allocated() / 2**30
    res["mem_reserved_GB"] = torch.cuda.memory_reserved() / 2**30
    res["static_buffer_bytes"] = sum(t.numel() * t.element_size() for bb in S.b.values() for v in bb.values()
                                     for t in (v if isinstance(v, list) else [v]))

    # ---------------- end-to-end incl. VAE decode (graph vs eager), steady iterations
    if a.decode:
        for name, fn in (("eager", lambda it: E.denoise(it)), ("graph", lambda it: S.denoise(it).clone())):
            E.start(); ts, tdec = [], []
            for k in range(N):
                torch.cuda.synchronize(); t0 = time.perf_counter()
                it = E.prep(k); lat = fn(it); den = E.commit(it, lat)
                torch.cuda.synchronize(); t1 = time.perf_counter()
                E.decode(den, it["first"])
                torch.cuda.synchronize(); t2 = time.perf_counter()
                ts.append(t2 - t0); tdec.append(t2 - t1)
            res[f"e2e_{name}"] = dict(total=ts, decode=tdec)
            print(f"e2e {name}", [round(x, 3) for x in ts], "decode", [round(x, 3) for x in tdec], flush=True)
        res["state_bytes_eager"]["vae_stream_cache"] = E.state_bytes()[1]

    te, tg_ = med(res["eager_run1"]["denoise"][1:]), med(tg[1:])
    res.update(t_eager_iter=te, t_graph_iter=tg_, speedup=te / tg_, t_nograph_iter=med(tng[2:]),
               t_eager_first=res["eager_run1"]["denoise"][0], t_graph_first=tg[0],
               t_retrieval_med=med(res["eager_run1"]["retrieval"][1:]), t_prep_med=med(res["eager_run1"]["prep"][1:]))
    print(f"steady iter DiT: eager {te*1e3:.1f} ms  graph {tg_*1e3:.1f} ms  ({te/tg_:.3f}x)  capture {res['capture_s']:.1f}s",
          flush=True)
    dump(res, a.out)
    print("DONE")


if __name__ == "__main__":
    main()
