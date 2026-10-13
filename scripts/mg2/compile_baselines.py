"""Can built-in PyTorch tools get the static runtime's speedup on the UNMODIFIED Matrix-Game-2 forward?
One mode per process (compiled state must not leak between modes):
  eager        : MG2 as shipped
  naive_graph  : torch.cuda.graph() around the eager chunk (gen_chunk), after warm-up
  compile      : torch.compile(model)                                  (Inductor, no CUDA graphs)
  compile_mat  : torch.compile(model, mode="max-autotune-no-cudagraphs")  (what the Self-Forcing README recommends)
  compile_ro   : torch.compile(model, mode="reduce-overhead")          (Inductor + CUDA graph trees)
  static       : our static runtime (static_mg.StaticMG), for reference
Each run: eager reference for 12 chunks first (same process), then the mode on the same path twice (pass 1 includes
compilation, pass 2 is warm). Reports per-chunk time, bitwise equality, dynamo graph breaks / recompiles / cudagraph skips.
  python scripts/mg2/compile_baselines.py --mode compile_ro [--chunks 12]
"""
import argparse, json, os, traceback
import torch
from fastworld import median as med, timed
from fastworld.models.mg2 import MG, Model, test_path
from fastworld.paths import ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True)
    ap.add_argument("--chunks", type=int, default=12)
    args = ap.parse_args()
    os.makedirs(f"{ROOT}/results/mg2/compile", exist_ok=True)
    out_path = f"{ROOT}/results/mg2/compile/{args.mode}.json"
    torch.set_grad_enabled(False)
    M = Model(max_chunks=args.chunks + 2)
    M.add_image("0000", f"{MG}/demo_images/universal/0000.png")
    path = test_path("0000", args.chunks)
    res = dict(mode=args.mode, gpu=torch.cuda.get_device_name(), torch=torch.__version__)

    def run(gen):
        lats, ts = [], []
        for k in range(len(path)):
            lat, dt = timed(lambda: gen(k)); lats.append(lat.clone()); ts.append(dt)
        return lats, ts

    def eager_gen(k):
        if k == 0:
            M.reset()
        return M.gen_chunk(path, k)

    run(eager_gen)                                   # warm-up
    ref, t_ref = run(eager_gen)
    res["t_eager"] = t_ref
    gen = eager_gen
    try:
        if args.mode == "eager":
            pass
        elif args.mode == "naive_graph":
            M.reset(); [M.gen_chunk(path, k) for k in range(2)]
            g = torch.cuda.CUDAGraph()
            s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                M.gen_chunk(path, 2)
            torch.cuda.current_stream().wait_stream(s)
            with torch.cuda.graph(g):
                M.gen_chunk(path, 3)
            res["note"] = "capture succeeded"
            # replay cannot change chunk index / RNG / python-side cache bookkeeping; record what it produces
            g.replay()
        elif args.mode.startswith("compile"):
            mode = {"compile": None, "compile_mat": "max-autotune-no-cudagraphs", "compile_ro": "reduce-overhead"}[args.mode]
            M.p.generator.model = torch.compile(M.p.generator.model, mode=mode)
        elif args.mode == "static":
            from fastworld.models.mg2 import StaticMG
            S = StaticMG(M); S.capture()
            gen = lambda k: S.gen_chunk(path, k)
        else:
            raise ValueError(args.mode)
        if args.mode != "naive_graph":
            l1, t1 = run(gen)                        # pass 1: includes compilation
            l2, t2 = run(gen)                        # pass 2: warm
            res.update(t_pass1=t1, t_pass2=t2,
                       bitwise_pass1=[bool(torch.equal(a, b)) for a, b in zip(l1, ref)],
                       bitwise_pass2=[bool(torch.equal(a, b)) for a, b in zip(l2, ref)],
                       maxdiff_pass2=[float((a.float() - b.float()).abs().max()) for a, b in zip(l2, ref)])
            res["chunk_eager"], res["chunk_mode"] = med(t_ref[2:]), med(t2[2:])
            res["speedup"] = res["chunk_eager"] / res["chunk_mode"]
            res["compile_overhead_s"] = sum(t1) - sum(t2)
            # GPU kernel time of warm chunks (no-overhead floor) -> how busy the GPU is
            from torch.profiler import profile, ProfilerActivity
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                for k in range(3):
                    gen(k)
                torch.cuda.synchronize()
            kev = [e for e in prof.key_averages() if getattr(e, "device_type", None) == torch.autograd.DeviceType.CUDA]
            res["kernel_3chunks_s"] = sum(e.self_device_time_total for e in kev) / 1e6
    except Exception as e:
        res["error"] = f"{type(e).__name__}: {str(e)[:600]}"
        res["traceback"] = traceback.format_exc()[-2000:]
    try:
        from torch._dynamo.utils import counters
        res["dynamo"] = {k: dict(v) for k, v in counters.items() if v}
        res["graph_breaks_total"] = sum(counters["graph_break"].values())
        res["graph_break_reasons"] = sorted(((str(k)[:160], v) for k, v in counters["graph_break"].items()), key=lambda x: -x[1])[:12]
    except Exception as e:
        res["dynamo_err"] = str(e)
    json.dump(res, open(out_path, "w"), indent=1, default=str)
    summ = {k: res.get(k) for k in ("chunk_eager", "chunk_mode", "speedup", "compile_overhead_s", "graph_breaks_total", "error", "note")}
    print(json.dumps(summ, default=str), flush=True)
    if "bitwise_pass2" in res:
        print("bitwise pass2", res["bitwise_pass2"], "maxdiff", [round(x, 3) for x in res["maxdiff_pass2"]], flush=True)
    for r in res.get("graph_break_reasons", []):
        print("  break", r, flush=True)
    print("DONE")


if __name__ == "__main__":
    main()
