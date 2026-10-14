"""Exactness + speed of the static/CUDA-graph LingBot-v2 runtime against the official eager loop.
  python scripts/lingbot_v2/test_static.py [--chunks 12] [--out results/lingbot_v2/static_<gpu>.json] [--skip_official] [--decode]
Layers: (0) engine == official pipe.generate, eager == eager; (1) prepared inputs (RoPE table) == eager;
(2) single forward bitwise; (3) multi-chunk rollout without graph; (4) with graph; (5) state save/load + branch.
"""
import argparse, json, os, time
import torch
from fastworld import median as med, timed
from fastworld.models.lingbot_v2 import (EagerSession, StaticLingbot, eager_plans, ex_paths, ex_prompt, load_pipe,
                                         official_latents, prepare_inputs)
from fastworld.paths import ROOT

EXS = ["00", "02"]


def eq(a, b):
    return bool(torch.equal(a, b))


def mdiff(a, b):
    return float((a.float() - b.float()).abs().max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=int, default=12)
    ap.add_argument("--out", default=None)
    ap.add_argument("--skip_official", action="store_true")
    ap.add_argument("--decode", action="store_true")
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    gpu = torch.cuda.get_device_name()
    out = args.out or f"{ROOT}/results/lingbot_v2/static_{gpu.split()[-1] if 'H100' not in gpu else 'H100'}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    res = {"gpu": gpu, "torch": torch.__version__, "chunks": args.chunks}
    save = lambda: json.dump(res, open(out, "w"), indent=1, default=str)
    t0 = time.time()
    pipe = load_pipe()
    res["load_s"] = time.time() - t0
    n = args.chunks
    I = {ex: prepare_inputs(pipe, ex_prompt(ex), *ex_paths(ex), n) for ex in EXS}
    i0 = I[EXS[0]]
    res["lat_shape"] = [16, i0.lat_f, i0.lat_h, i0.lat_w]
    res["frame_seqlen"] = i0.frame_seqlen
    res["plans"] = [str(p) for p in eager_plans(n, i0.chunk_size)]
    print("lat", res["lat_shape"], "plans", res["plans"], flush=True)

    # ---- eager reference (twice for ex 0: warm-up + eager-vs-eager determinism)
    ref, t_eager = {}, {}
    for rep in range(2):
        for ex in (EXS if rep == 1 else EXS[:1]):
            E = EagerSession(pipe, I[ex])
            lats, ts = [], []
            for k in range(n):
                lat, dt = timed(lambda: E.gen_chunk(k)); lats.append(lat.clone()); ts.append(dt)
            if rep == 0:
                first = lats
            else:
                ref[ex], t_eager[ex] = lats, ts
            if rep == 1 and ex == EXS[0]:
                res["state_bytes_eager"] = E.state_bytes()
            del E
    res["eager_vs_eager_bitwise"] = [eq(a, b) for a, b in zip(first, ref[EXS[0]])]
    res["t_eager"] = t_eager
    print("eager vs eager", res["eager_vs_eager_bitwise"], flush=True)
    print("eager chunk times", {k: [round(x * 1e3, 1) for x in v] for k, v in t_eager.items()}, flush=True)
    save()

    # ---- (0) engine vs unmodified official generate()
    if not args.skip_official:
        ex = EXS[0]
        lat = official_latents(pipe, ex_prompt(ex), *ex_paths(ex), n)
        mine = torch.cat(ref[ex], dim=1)
        res["engine_vs_official"] = (list(lat.shape), eq(lat, mine), mdiff(lat, mine))
        print("engine vs official generate()", res["engine_vs_official"], flush=True)
        save()

    for hoist in (False, True):
        tag = "hoist" if hoist else "nohoist"
        S = StaticLingbot(pipe, i0, hoist=hoist)
        res["state_bytes_static_steady"] = S.state_bytes(live=S.plans[8][2][1] + S.plans[8][2][2])
        # ---- (1) RoPE table vs eager causal_rope_apply
        if not hoist:
            from wan.modules.model_fast import causal_rope_apply
            q = torch.randn(1, S.L, S.H, S.D, device=S.dev, dtype=torch.bfloat16)
            gs = torch.tensor([S.grid], dtype=torch.long)
            ok = []
            for k in sorted({0, min(3, n - 1), min(7, n - 1), n - 1}):
                S._prep(k, noise=False)
                ok.append(eq(causal_rope_apply(q, gs, S.net.freqs, start_frame=k * S.C), S._rope(q).type_as(q)))
            res["rope_equal"] = ok
            # ---- (2) single forward at chunk 0, step 0 (official forward, fresh caches) vs static
            E = EagerSession(pipe, i0)
            with torch.amp.autocast('cuda', dtype=pipe.param_dtype):
                a = pipe.model(x=[E.lat[0]], t=torch.stack([i0.timesteps[0]]).to(pipe.device), context=[i0.context[0]],
                               seq_len=i0.max_seq_len, y=[E.cond[0]], dit_cond_dict={"c2ws_plucker_emb": E.cam[0].chunk(1, dim=0)},
                               kv_cache=E.kv, crossattn_cache=E.xkv, current_start=0, max_attention_size=i0.kv_size,
                               frame_seqlen=i0.frame_seqlen, cross_attn_first_call=True)[0]
                S.reset(); S._prep(0)
                b = S._net(S.x_in, 0, *S.plans[0][:2])
            res["fwd0_bitwise"] = (eq(a, b), mdiff(a, b))
            print("rope", res["rope_equal"], "fwd0", res["fwd0_bitwise"], flush=True)
            del E
        # ---- (3) no graph
        res[f"nograph_bitwise_{tag}"], res[f"nograph_maxdiff_{tag}"] = {}, {}
        for ex in EXS:
            S.set_session(I[ex])
            o = [S.gen_chunk(k, graph=False) for k in range(n)]
            res[f"nograph_bitwise_{tag}"][ex] = [eq(x, r) for x, r in zip(o, ref[ex])]
            res[f"nograph_maxdiff_{tag}"][ex] = [mdiff(x, r) for x, r in zip(o, ref[ex])]
        print(tag, "no-graph bitwise", res[f"nograph_bitwise_{tag}"], "maxdiff", res[f"nograph_maxdiff_{tag}"], flush=True)
        save()
        # ---- (4) graph
        S.set_session(i0)
        _, res[f"capture_s_{tag}"] = timed(S.capture)
        res[f"n_graphs_{tag}"] = len(S.graphs)
        res[f"graph_bitwise_{tag}"], t_graph = {}, {}
        for rep in range(2):
            for ex in EXS:
                S.set_session(I[ex])
                o, ts = [], []
                for k in range(n):
                    lat, dt = timed(lambda: S.gen_chunk(k)); o.append(lat); ts.append(dt)
                res[f"graph_bitwise_{tag}"][ex] = [eq(x, r) for x, r in zip(o, ref[ex])]
                t_graph[ex] = ts
        res[f"t_graph_{tag}"] = t_graph
        print(tag, "graph bitwise", res[f"graph_bitwise_{tag}"], flush=True)
        print(tag, "graph chunk times", {k: [round(x * 1e3, 1) for x in v] for k, v in t_graph.items()}, flush=True)
        # ---- (5) branch: save after chunk b-1, run chunk b with another camera, restore, continue the original path
        ex = EXS[0]; S.set_session(I[ex]); b = n // 2
        for k in range(b):
            S.gen_chunk(k)
        st = S.state()
        alt = I[EXS[1]].plucker[:, :, b * S.C:(b + 1) * S.C]
        alt_lat = S.gen_chunk(b, cam=alt)
        S.gen_chunk(b + 1)
        S.load(st)
        rest = [S.gen_chunk(k) for k in range(b, n)]
        res[f"branch_restore_bitwise_{tag}"] = [eq(x, r) for x, r in zip(rest, ref[ex][b:])]
        res[f"branch_alt_differs_{tag}"] = not eq(alt_lat, ref[ex][b])
        _, res[f"state_save_s_{tag}"] = timed(S.state)
        _, res[f"state_load_s_{tag}"] = timed(lambda: S.load(st))
        res[f"state_bytes_live_{tag}"] = sum(t.numel() * t.element_size() for t in st["kv"])
        print(tag, "branch", res[f"branch_restore_bitwise_{tag}"], "alt differs", res[f"branch_alt_differs_{tag}"], flush=True)
        # ---- kernel time of steady graph chunks
        from torch.profiler import profile, ProfilerActivity
        S.set_session(i0)
        kp = list(range(6, 9)) if n >= 9 else list(range(max(4, n - 3), n))   # steady chunks (window full)
        for k in range(kp[0]):
            S.gen_chunk(k)
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for k in kp:
                S.gen_chunk(k)
            torch.cuda.synchronize()
        kev = [e for e in prof.key_averages() if getattr(e, "device_type", None) == torch.autograd.DeviceType.CUDA]
        res[f"graph_kernel_chunk_{tag}"] = sum(e.self_device_time_total for e in kev) / len(kp) / 1e6
        te = med([t for v in t_eager.values() for t in v[4:]])     # steady chunks (window full)
        tg = med([t for v in t_graph.values() for t in v[4:]])
        res[f"t_eager_steady"], res[f"t_graph_steady_{tag}"], res[f"speedup_{tag}"] = te, tg, te / tg
        print(f"{tag}: steady chunk eager {te*1e3:.1f} ms  graph {tg*1e3:.1f} ms ({te/tg:.3f}x)  "
              f"graph kernel {res[f'graph_kernel_chunk_{tag}']*1e3:.1f} ms  capture {res[f'capture_s_{tag}']:.1f}s", flush=True)
        save()
        del S
        torch.cuda.empty_cache()

    if args.decode:
        lat = torch.cat(ref[EXS[0]], dim=1)
        with torch.amp.autocast('cuda', dtype=pipe.param_dtype):
            pipe.vae.decode([lat[:, :4]])
            v, res["vae_decode_all_s"] = timed(lambda: pipe.vae.decode([lat]))
        res["vae_decode_per_chunk_s"] = res["vae_decode_all_s"] / n
        res["video_shape"] = list(v[0].shape)
        print("vae decode", res["vae_decode_all_s"], "s for", res["video_shape"], flush=True)
    res["state_bytes_eager_gib"] = res["state_bytes_eager"] / 2**30
    res["state_bytes_static_steady_gib"] = res["state_bytes_static_steady"] / 2**30
    save()
    print("DONE")


if __name__ == "__main__":
    main()
