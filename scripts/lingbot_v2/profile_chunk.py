"""Profile one steady-state chunk (window full, rolling): eager vs static (no graph) vs static (graph).
Reports wall time (no profiler, median of 4 chunks), CUDA kernel time (sum of DeviceType.CUDA self time), kernel
launches, memcpys, host syncs, GPU-busy % = kernel / wall, and the top kernels.
  python scripts/lingbot_v2/profile_chunk.py [--out results/lingbot_v2/profile_<gpu>.json]
"""
import argparse, json, os, time
import torch
from torch.profiler import profile, ProfilerActivity
from fastworld import median
from fastworld.models.lingbot_v2 import EagerSession, StaticLingbot, ex_paths, ex_prompt, load_pipe, prepare_inputs
from fastworld.paths import ROOT

LAUNCH = {"cudaLaunchKernel", "cudaLaunchKernelExC", "cuLaunchKernel", "cuLaunchKernelEx"}
SYNC = {"cudaStreamSynchronize", "cudaDeviceSynchronize", "cudaEventSynchronize", "cudaMemcpy", "cudaStreamWaitEvent_sync"}
PROF_K = 7


def analyze(prof):
    names = [e.name for e in prof.events()]
    cnt = lambda s: sum(1 for n in names if n in s)
    kev = [e for e in prof.key_averages() if getattr(e, "device_type", None) == torch.autograd.DeviceType.CUDA]
    ktime = sum(e.self_device_time_total for e in kev) / 1e6
    top = sorted(((e.key[:90], e.self_device_time_total / 1e3, e.count) for e in kev), key=lambda x: -x[1])[:15]
    return dict(kernel_s=ktime, launches=cnt(LAUNCH), memcpy_async=cnt({"cudaMemcpyAsync"}), syncs=cnt(SYNC),
                stream_syncs=cnt({"cudaStreamSynchronize"}), item_calls=cnt({"aten::_local_scalar_dense"}),
                graph_launches=cnt({"cudaGraphLaunch"}), n_kernels=sum(e.count for e in kev), top_kernels_ms=top)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    gpu = torch.cuda.get_device_name()
    out = args.out or f"{ROOT}/results/lingbot_v2/profile_{'H100' if 'H100' in gpu else gpu.split()[-1]}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pipe = load_pipe()
    ex = "00"
    I = prepare_inputs(pipe, ex_prompt(ex), *ex_paths(ex), 12)
    res = {"gpu": gpu, "chunk_profiled": PROF_K}

    def measure(name, reset, gen):
        # wall: chunks 5..10 (steady), no profiler
        reset()
        ts = []
        for k in range(11):
            torch.cuda.synchronize(); t0 = time.time(); gen(k); torch.cuda.synchronize()
            if k >= 5:
                ts.append(time.time() - t0)
        wall = median(ts)
        reset()
        for k in range(PROF_K):
            gen(k)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            gen(PROF_K)
            torch.cuda.synchronize()
        r = analyze(prof)
        r.update(wall_s=wall, wall_all_s=ts, gpu_busy=r["kernel_s"] / wall)
        res[name] = r
        print(name, {k: v for k, v in r.items() if k != "top_kernels_ms"}, flush=True)
        for t in r["top_kernels_ms"][:8]:
            print("   ", t, flush=True)
        json.dump(res, open(out, "w"), indent=1, default=str)

    E = EagerSession(pipe, I)
    measure("eager", E.reset, E.gen_chunk)
    res["eager_state_bytes"] = E.state_bytes()
    del E
    for hoist in (False, True):
        S = StaticLingbot(pipe, I, hoist=hoist)
        tag = "hoist" if hoist else "nohoist"
        measure(f"static_nograph_{tag}", S.reset, lambda k: S.gen_chunk(k, graph=False))
        S.capture()
        measure(f"static_graph_{tag}", S.reset, S.gen_chunk)
        del S
        torch.cuda.empty_cache()
    print("DONE")


if __name__ == "__main__":
    main()
