"""torch.compile baseline. The repo only offers --compile_vae (VAE decoder); there is no official DiT compile setting,
so this is torch.compile(model) with defaults.
  python scripts/mg3/compile_baseline.py --iters 8 --out results/mg3/compile_h100.json   (run via scripts/mg3/job.sh)
"""
import argparse, json, os, sys
import torch
from fastworld import median as med, timed as sync_time
from fastworld.models.mg3 import Engine
from fastworld.paths import ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--int8", type=int, default=1)
    ap.add_argument("--mode", default=None)
    ap.add_argument("--out", default=f"{ROOT}/results/mg3/compile.json")
    a = ap.parse_args()
    if not os.path.isabs(a.out):
        a.out = os.path.join(ROOT, a.out)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    torch.set_grad_enabled(False)
    E = Engine(use_int8=bool(a.int8), num_iterations=a.iters)
    E.start(); E.free_t5()
    N = a.iters
    res = {"gpu": torch.cuda.get_device_name(), "iters": N, "mode": a.mode, "torch": torch.__version__}
    ref, te = [], []
    E.start()
    for k in range(N):
        it = E.prep(k)
        lat, dt = sync_time(lambda: E.denoise(it))
        E.commit(it, lat); ref.append(lat.clone()); te.append(dt)
    res["eager_denoise"] = te
    print("eager", [round(x, 3) for x in te], flush=True)
    eager_model = E.model
    # model.py loads wan/triton_kernels.py via spec_from_file_location without registering it in sys.modules, which
    # makes dynamo crash (InternalTorchDynamoError: No module named 'triton_kernels') and fall back to eager for
    # every frame that touches Int8Linear. Registering the already-loaded module is a no-op for eager.
    import wan.modules.model as wm
    sys.modules.setdefault("triton_kernels", wm._get_triton_kernels())
    E.model = torch.compile(eager_model, mode=a.mode) if a.mode else torch.compile(eager_model)
    out = []
    for r in range(2):
        E.start(); tc = []
        for k in range(N):
            it = E.prep(k)
            lat, dt = sync_time(lambda: E.denoise(it))
            E.commit(it, lat); tc.append(dt)
            if r == 1:
                out.append(lat.clone())
            print(f"compiled run {r} iter {k}: {dt:.3f}s", flush=True)
        res[f"compiled_run{r}"] = tc
        json.dump(res, open(a.out, "w"), indent=1)
    from torch._dynamo.utils import counters
    res["dynamo_counters"] = {k: dict(v) for k, v in counters.items() if k in ("stats", "graph_break", "recompiles")}
    res["compiled_bitwise"] = [bool(torch.equal(x, y)) for x, y in zip(out, ref)]
    res["compiled_maxdiff"] = [float((x.float() - y.float()).abs().max()) for x, y in zip(out, ref)]
    res["compiled_meanabsdiff"] = [float((x.float() - y.float()).abs().mean()) for x, y in zip(out, ref)]
    res["ref_meanabs"] = [float(y.float().abs().mean()) for y in ref]
    res["compiled_cos"] = [float(torch.nn.functional.cosine_similarity(x.float().flatten(), y.float().flatten(), dim=0))
                           for x, y in zip(out, ref)]
    res["t_eager_iter"] = med(te[1:])
    res["t_compiled_iter"] = med(res["compiled_run1"][1:])
    res["compile_overhead_s"] = sum(res["compiled_run0"]) - sum(res["compiled_run1"])
    print(json.dumps({k: v for k, v in res.items() if k != "dynamo_counters"}, indent=1), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("DONE")


if __name__ == "__main__":
    main()
