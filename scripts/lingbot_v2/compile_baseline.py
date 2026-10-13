"""torch.compile baseline on the UNMODIFIED LingBot-v2 eager loop (the repo ships no compile setting; NVIDIA flashdreams
compiles its own re-implementation with dynamic=True and no CUDA graphs). One mode per process:
  compile     : torch.compile(pipe.model)                       (default Inductor)
  compile_dyn : torch.compile(pipe.model, dynamic=True)         (flashdreams-like)
  compile_ro  : torch.compile(pipe.model, mode="reduce-overhead")
Eager reference for N chunks first (same process), then the compiled model on the same path twice (pass 1 includes
compilation, pass 2 is warm). Reports per-chunk time, bitwise equality, graph breaks.
  python scripts/lingbot_v2/compile_baseline.py --mode compile [--chunks 12] [--out results/lingbot_v2/compile/<mode>_<gpu>.json]
"""
import argparse, json, os, traceback
import torch
from fastworld import median as med, timed
from fastworld.models.lingbot_v2 import EagerSession, ex_paths, ex_prompt, load_pipe, prepare_inputs
from fastworld.paths import ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="compile")
    ap.add_argument("--chunks", type=int, default=12)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    gpu = torch.cuda.get_device_name()
    out = args.out or f"{ROOT}/results/lingbot_v2/compile/{args.mode}_{'H100' if 'H100' in gpu else gpu.split()[-1]}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pipe = load_pipe()
    ex = "00"
    I = prepare_inputs(pipe, ex_prompt(ex), *ex_paths(ex), args.chunks)
    res = dict(mode=args.mode, gpu=gpu, torch=torch.__version__)
    n = args.chunks

    def run():
        E = EagerSession(pipe, I)
        lats, ts = [], []
        for k in range(n):
            lat, dt = timed(lambda: E.gen_chunk(k)); lats.append(lat.clone()); ts.append(dt)
        return lats, ts

    run()
    ref, t_ref = run()
    res["t_eager"] = t_ref
    try:
        kw = {"compile": {}, "compile_dyn": {"dynamic": True}, "compile_ro": {"mode": "reduce-overhead"}}[args.mode]
        pipe.model = torch.compile(pipe.model, **kw)
        l1, t1 = run()
        l2, t2 = run()
        res.update(t_pass1=t1, t_pass2=t2,
                   bitwise_pass1=[bool(torch.equal(a, b)) for a, b in zip(l1, ref)],
                   bitwise_pass2=[bool(torch.equal(a, b)) for a, b in zip(l2, ref)],
                   maxdiff_pass2=[float((a.float() - b.float()).abs().max()) for a, b in zip(l2, ref)])
        res["chunk_eager_steady"], res["chunk_mode_steady"] = med(t_ref[4:]), med(t2[4:])
        res["speedup"] = res["chunk_eager_steady"] / res["chunk_mode_steady"]
        res["compile_overhead_s"] = sum(t1) - sum(t2)
    except Exception as e:
        res["error"] = f"{type(e).__name__}: {str(e)[:600]}"
        res["traceback"] = traceback.format_exc()[-2000:]
    try:
        from torch._dynamo.utils import counters
        res["dynamo"] = {k: dict(v) for k, v in counters.items() if v}
        res["graph_breaks_total"] = sum(counters["graph_break"].values())
        res["graph_break_reasons"] = sorted(((str(k)[:160], v) for k, v in counters["graph_break"].items()),
                                            key=lambda x: -x[1])[:12]
    except Exception as e:
        res["dynamo_err"] = str(e)
    json.dump(res, open(out, "w"), indent=1, default=str)
    summ = {k: res.get(k) for k in ("chunk_eager_steady", "chunk_mode_steady", "speedup", "compile_overhead_s",
                                    "graph_breaks_total", "error")}
    print(json.dumps(summ, default=str), flush=True)
    if "bitwise_pass2" in res:
        print("bitwise pass2", res["bitwise_pass2"], "maxdiff", [round(x, 4) for x in res["maxdiff_pass2"]], flush=True)
    for r in res.get("graph_break_reasons", []):
        print("  break", r, flush=True)
    print("DONE")


if __name__ == "__main__":
    main()
