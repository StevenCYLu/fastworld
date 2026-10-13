"""Baseline: the official torch.compile setting (configs/fp8/inference_fp8.yaml: inductor, max-autotune-no-cudagraphs,
fullgraph=False, dynamic=False, suppress_errors=False) applied to the BF16 generator through the official
`WanDiffusionWrapper.configure_torch_compile` hook. Eager reference first (same process), then 3 compiled runs.
  python scripts/longlive2/compile_baseline.py --chunks 12 --out results/longlive2/compile_h100.json
"""
import argparse, json, os, time
import torch
import torch._dynamo
from fastworld import median as med
import fastworld.models.longlive2 as C
from fastworld.paths import ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=int, default=12)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--out", default=f"{ROOT}/results/longlive2/compile.json")
    ap.add_argument("--recompile_limit", type=int, default=0, help="0 = torch default (8), as in the official setting")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    res = {"gpu": torch.cuda.get_device_name(), "chunks": args.chunks}
    pipe, cfg = C.build_pipeline()
    prompts = C.prompt_schedule(args.chunks)
    noise = C.make_noise(args.chunks)
    pipe.text_encoder = C.CachedTextEncoder(pipe.text_encoder, prompts)
    torch.cuda.empty_cache()
    C.init_caches(pipe); C.shrink_neg_cache(pipe)
    timer = C.ChunkTimer(pipe)
    C.run_official(pipe, noise, prompts, timer)
    ref, te = C.run_official(pipe, noise, prompts, timer)
    ref = ref.clone()
    res["t_eager"] = te
    print("eager per-chunk", [round(x, 3) for x in te], flush=True)
    comp = dict(backend="inductor", mode="max-autotune-no-cudagraphs", fullgraph=False, dynamic=False, options=None,
                suppress_errors=False)
    res["compile_cfg"] = comp
    if args.recompile_limit:
        torch._dynamo.config.recompile_limit = args.recompile_limit
        res["recompile_limit"] = args.recompile_limit
    pipe.generator.configure_torch_compile(**comp)
    import torch._dynamo.utils as du
    runs = []
    for r in range(args.runs):
        t0 = time.perf_counter()
        lat, tc = C.run_official(pipe, noise, prompts, timer)
        tot = time.perf_counter() - t0
        diff = [float((lat[:, k*C.FPB:(k+1)*C.FPB].float() - ref[:, k*C.FPB:(k+1)*C.FPB].float()).abs().max()) for k in range(args.chunks)]
        bit = [bool(torch.equal(lat[:, k*C.FPB:(k+1)*C.FPB], ref[:, k*C.FPB:(k+1)*C.FPB])) for k in range(args.chunks)]
        cc = getattr(pipe.generator, "_compiled_model_call", None)
        runs.append(dict(total_s=tot, per_chunk=tc, maxdiff=diff, bitwise=bit,
                         fell_back_to_eager=bool(cc is not None and cc.failed), failure=getattr(cc, "failure_reason", None),
                         dynamo_counters={k: dict(v) for k, v in du.counters.items() if k in ("stats", "recompiles", "graph_break", "frames")}))
        print(f"compiled run {r}: total {tot:.1f}s per-chunk", [round(x, 3) for x in tc], "bitwise", bit,
              "maxdiff", [f"{d:.3g}" for d in diff], flush=True)
        res["runs"] = runs
        json.dump(res, open(args.out, "w"), indent=1)
    warm = [t for t in runs[-1]["per_chunk"][4:]]
    res["t_compile_chunk_warm_median"] = med(warm)
    res["t_eager_chunk_median"] = med(te[4:])
    res["compile_time_s_est"] = runs[0]["total_s"] - sum(runs[-1]["per_chunk"])
    print(f"compile warm median {res['t_compile_chunk_warm_median']*1e3:.1f} ms vs eager {res['t_eager_chunk_median']*1e3:.1f} ms;"
          f" first-run overhead ~{res['compile_time_s_est']:.0f}s", flush=True)
    json.dump(res, open(args.out, "w"), indent=1)
    print("DONE")


if __name__ == "__main__":
    main()
