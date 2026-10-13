"""Exactness + speed of the static/CUDA-graph LongLive-2.0 runtime against the official eager pipeline (same GPU, same
process).  python scripts/longlive2/test_static.py --chunks 12 [--recache 0|1] [--branch 1] --out results/longlive2/static_h100.json
"""
import argparse, json, os
import torch
from fastworld import gpu_profile, median as med, timed
import fastworld.models.longlive2 as C
from fastworld.paths import ROOT

N = C.N


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=int, default=12)
    ap.add_argument("--recache", type=int, default=0)
    ap.add_argument("--branch", type=int, default=1)
    ap.add_argument("--out", default=f"{ROOT}/results/longlive2/static.json")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    res = {"gpu": torch.cuda.get_device_name(), "chunks": args.chunks, "shot_clean_recache": bool(args.recache)}
    dump = lambda: json.dump(res, open(args.out, "w"), indent=1)
    pipe, cfg = C.build_pipeline(shot_clean_recache=bool(args.recache))
    prompts = C.prompt_schedule(args.chunks)
    noise = C.make_noise(args.chunks)
    pipe.text_encoder = C.CachedTextEncoder(pipe.text_encoder, prompts)
    torch.cuda.empty_cache()
    C.init_caches(pipe)
    res["state_bytes_eager_allocated"] = C.eager_state_bytes(pipe, True)
    res["state_bytes_eager_pos_only"] = C.eager_state_bytes(pipe, False)
    C.shrink_neg_cache(pipe)
    embeds = [d["prompt_embeds"] for d in C.encode_prompts(pipe, prompts)]
    nch = args.chunks
    sl = lambda t, k: t[:, k * C.FPB:(k + 1) * C.FPB]

    # ---------------- eager reference: warm-up run, recorded run, determinism run
    timer = C.ChunkTimer(pipe)
    idx_log = []
    orig_hook = pipe._initialize_sample_scheduler

    def idx_hook(noise_):
        c = pipe.kv_cache_pos[0]
        idx_log.append(tuple(int(c[n].item()) for n in ("global_end_index", "local_end_index", "pinned_start", "pinned_len")))
        return orig_hook(noise_)
    pipe._initialize_sample_scheduler = idx_hook
    rec = []
    gfwd = pipe.generator.forward

    def rec_fwd(*a, **kw):
        out = gfwd(*a, **kw)
        if rec is not None:
            rec.append(out[0].clone())
        return out
    C.run_official(pipe, noise, prompts, timer)                      # warm-up (triton autotune, cudnn, ...)
    pipe.generator.forward = rec_fwd
    rec.clear(); idx_log.clear()
    ref, t_eager1 = C.run_official(pipe, noise, prompts, timer)
    ref = ref.clone()
    pipe.generator.forward = gfwd
    flows_ref = list(rec); rec = None
    idx_ref = list(idx_log)
    ref2, t_eager2 = C.run_official(pipe, noise, prompts, timer)
    res["eager_vs_eager_bitwise"] = [bool(torch.equal(sl(ref, k), sl(ref2, k))) for k in range(nch)]
    del ref2
    res["t_eager"] = [t_eager1, t_eager2]
    print("eager per-chunk s", [round(x, 3) for x in t_eager2], "eager==eager", res["eager_vs_eager_bitwise"], flush=True)
    dump()

    S = C.StaticLL(pipe)
    res["state_bytes_static_resident"] = S.state_bytes(live=False)

    # ---------------- (1) prepared inputs == eager's: RoPE (vs causal_rope_apply), host index plan (vs eager indices)
    from wan_5b.modules.causal_model import causal_rope_apply
    g = torch.Generator(device="cuda").manual_seed(1)
    qr = torch.randn(1, N, S.H, S.D, device="cuda", dtype=torch.bfloat16, generator=g)
    ok_rope = []
    for k, off in ((0, 0.0), (5, 8), (9, 16)):
        cos, sin = S._rope_tables(C.FPB * k, off)
        S.cos.copy_(cos); S.sin.copy_(sin)
        a = causal_rope_apply(qr, [(C.FPB, 22, 40)], S.net.freqs, start_frame=C.FPB * k, t_scale=S.net.t_scale,
                              method=S.net.rope_method, original_seq_len=S.net.original_seq_len, temporal_offset=off)
        ok_rope.append(bool(torch.equal(a, S._rope(qr))))
    res["prep_rope_equal"] = ok_rope
    S.reset()
    plan = []
    for k in range(nch):
        plan.append((S.ge, S.le, S.ps, S.pl))
        S._insert(k * N)        # dry run of the index bookkeeping
        S._insert(k * N)
        boundary = k > 0 and prompts[k].startswith(S.prefix)
        if boundary and S.recache:
            S.le, S.ge, S.ps, S.pl = S.gsink, k * N, -1, 0
            S._insert(k * N)
        if boundary:
            S.ps, S.pl = S.le - N, min(S.sink, N)
    res["prep_index_plan_equal"] = plan == idx_ref
    res["index_plan"] = plan
    print("prep: rope", ok_rope, "index plan == eager", res["prep_index_plan_equal"], plan, flush=True)
    S.reset()

    # ---------------- (2) single forward (chunk 0, step 0)
    with torch.no_grad():
        L, Lc, cut = S._prep(sl(noise, 0), prompts[0], embeds[0])
        f0 = S._net(S.x_in, 0, L)
    res["fwd0_bitwise"] = (bool(torch.equal(f0, flows_ref[0])), float((f0.float() - flows_ref[0].float()).abs().max()))
    print("fwd0", res["fwd0_bitwise"], flush=True)
    for t in S.K + S.V:
        t.zero_()
    S.reset()

    # ---------------- (3) static, no graph (records every denoising flow_pred)
    S.rec = []
    out_ng, t_ng = [], []
    for k in range(nch):
        o, dt = timed(lambda: S.gen_chunk(sl(noise, k), prompts[k], embeds[k], graph=False).clone())
        out_ng.append(o); t_ng.append(dt)
    flows_ng = S.rec; S.rec = None
    res["nograph_bitwise"] = [bool(torch.equal(o, sl(ref, k))) for k, o in enumerate(out_ng)]
    res["nograph_maxdiff"] = [float((o.float() - sl(ref, k).float()).abs().max()) for k, o in enumerate(out_ng)]
    fl_ok = []
    for k in range(nch):
        for i in range(S.n_steps):
            fl_ok.append(bool(torch.equal(flows_ng[k * S.n_steps + i], flows_ref[k * (S.n_steps + 1) + i])))
    res["nograph_flow_bitwise_first_fail"] = fl_ok.index(False) if False in fl_ok else None
    res["t_static_nograph"] = t_ng
    print("no-graph bitwise", res["nograph_bitwise"], "maxdiff", res["nograph_maxdiff"],
          "first flow mismatch", res["nograph_flow_bitwise_first_fail"], flush=True)
    del flows_ng, out_ng
    dump()

    # ---------------- (4) capture + graph rollout
    S.reset()
    _, res["capture_s"] = timed(S.capture)
    res["graph_mem_gib"] = torch.cuda.max_memory_allocated() / 2 ** 30
    print(f"captured {list(S.graphs)} in {res['capture_s']:.1f}s", flush=True)
    t_graph, gb = [], []
    for rep in range(2):
        S.reset()
        tg, outs = [], []
        for k in range(nch):
            o, dt = timed(lambda: S.gen_chunk(sl(noise, k), prompts[k], embeds[k]).clone())
            outs.append(o); tg.append(dt)
        gb.append([bool(torch.equal(o, sl(ref, k))) for k, o in enumerate(outs)])
        t_graph.append(tg)
    res["graph_bitwise"] = gb
    res["t_static_graph"] = t_graph
    res["state_bytes_static_live"] = S.state_bytes(live=True)
    print("graph bitwise", gb, "per-chunk s", [round(x, 3) for x in t_graph[-1]], flush=True)
    dump()

    # ---------------- (5) branch: save after chunk b-1, take a different prompt at b, restore, continue original
    if args.branch:
        b = nch // 2 + 1
        S.reset()
        for k in range(b):
            S.gen_chunk(sl(noise, k), prompts[k], embeds[k])
        st, res["state_save_s"] = timed(S.state)
        alt_prompt = C.CUT + C.PROMPTS[(b // 4 + 1) % len(C.PROMPTS)]
        alt_emb = embeds[(b + 4) % nch]          # some other prompt embedding
        S.gen_chunk(torch.randn_like(sl(noise, b)), alt_prompt, alt_emb)
        S.gen_chunk(sl(noise, b + 1), prompts[b + 1], embeds[b + 1])
        _, res["state_load_s"] = timed(lambda: S.load(st))
        rest = [S.gen_chunk(sl(noise, k), prompts[k], embeds[k]).clone() for k in range(b, nch)]
        res["branch_restore_bitwise"] = [bool(torch.equal(o, sl(ref, b + i))) for i, o in enumerate(rest)]
        print("branch", res["branch_restore_bitwise"], flush=True)
        del st

    # ---------------- (6) speed summary + kernel time of one steady graph chunk
    te = med([t for tt in res["t_eager"] for t in tt[4:]])
    tg = med([t for tt in t_graph for t in tt[4:]])
    tn = med(t_ng[4:])
    res.update(t_eager_chunk=te, t_graph_chunk=tg, t_nograph_chunk=tn, speedup_graph=te / tg, speedup_nograph=te / tn)
    pc = min(6, nch - 1)                      # a steady (full-window) chunk
    S.reset()
    for k in range(pc - 1):
        S.gen_chunk(sl(noise, k), prompts[k], embeds[k])
    kk = [pc - 1]                             # gpu_profile runs fn once to warm, then `iters` times profiled

    def step():
        S.gen_chunk(sl(noise, kk[0]), prompts[kk[0]], embeds[kk[0]]); kk[0] += 1
    p = gpu_profile(step, iters=1)
    res["graph_chunk_kernel_s"] = p["kernel"]
    res["graph_chunk_launches"], res["graph_chunk_syncs"] = p["launches"], p["syncs"]
    res["graph_gpu_busy_pct"] = 100 * res["graph_chunk_kernel_s"] / tg
    print(f"chunk: eager {te*1e3:.1f} ms  static-nograph {tn*1e3:.1f} ms ({te/tn:.3f}x)  graph {tg*1e3:.1f} ms ({te/tg:.3f}x)"
          f"  graph kernel {res['graph_chunk_kernel_s']*1e3:.1f} ms  launches {p['launches']:.0f} syncs {p['syncs']:.0f}"
          f"  capture {res['capture_s']:.1f}s"
          f"  state static live {res['state_bytes_static_live']/2**30:.2f} GiB resident {res['state_bytes_static_resident']/2**30:.2f}"
          f"  eager {res['state_bytes_eager_allocated']/2**30:.2f} (pos only {res['state_bytes_eager_pos_only']/2**30:.2f})", flush=True)
    dump()

    # (7) VAE decode cost is measured separately by vae_time.py (official batch decode; identical for both paths)
    print("DONE")


if __name__ == "__main__":
    main()
