"""Official eager LongLive-2.0-5B: per-chunk wall times, eager-vs-eager determinism, and a torch.profiler breakdown of
one steady-state chunk (CUDA kernel time vs wall, kernel launches, host syncs). The profile is schedule-stepped at the
chunk boundaries of the official loop (not a plain repeated call), so it keeps its own profiler code.
  python scripts/longlive2/eager_profile.py --chunks 12 --profile_chunk 6 --out results/longlive2/eager_profile_h100.json
"""
import argparse, json, os
import torch
from torch.profiler import profile, ProfilerActivity, schedule
from fastworld import median as med
import fastworld.models.longlive2 as C
from fastworld.paths import ROOT

LAUNCH = ("cudaLaunchKernel", "cuLaunchKernel", "cudaLaunchKernelExC", "cuLaunchKernelEx", "cudaGraphLaunch")
SYNC = ("cudaStreamSynchronize", "cudaDeviceSynchronize", "cudaEventSynchronize")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=int, default=12)
    ap.add_argument("--profile_chunk", type=int, default=6)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--out", default=f"{ROOT}/results/longlive2/eager_profile.json")
    ap.add_argument("--save_ref", default="")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    res = {"gpu": torch.cuda.get_device_name(), "chunks": args.chunks}
    pipe, cfg = C.build_pipeline()
    prompts = C.prompt_schedule(args.chunks)
    res["prompts"] = prompts
    noise = C.make_noise(args.chunks)
    pipe.text_encoder = C.CachedTextEncoder(pipe.text_encoder, prompts)
    torch.cuda.empty_cache()
    C.init_caches(pipe)
    res["eager_state_bytes_allocated"] = C.eager_state_bytes(pipe, True)
    res["eager_state_bytes_pos_only"] = C.eager_state_bytes(pipe, False)
    C.shrink_neg_cache(pipe)
    timer = C.ChunkTimer(pipe)
    lats, times = [], []
    for r in range(args.runs):
        lat, t = C.run_official(pipe, noise, prompts, timer)
        lats.append(lat.clone()); times.append(t)
        print(f"run {r}: per-chunk s", [round(x, 3) for x in t], flush=True)
    res["times"] = times
    res["eager_vs_eager_bitwise"] = [[bool(torch.equal(lats[0][:, k*C.FPB:(k+1)*C.FPB], l[:, k*C.FPB:(k+1)*C.FPB]))
                                      for k in range(args.chunks)] for l in lats[1:]]
    print("eager vs eager bitwise", res["eager_vs_eager_bitwise"], flush=True)
    if args.save_ref:
        torch.save(lats[-1].cpu(), args.save_ref)
    steady = [t for tt in times[1:] for t in tt[4:]]
    res["eager_chunk_median_s"] = med(steady)
    print("eager steady median s/chunk", res["eager_chunk_median_s"], flush=True)

    # profile ONE steady chunk (chunk index profile_chunk); profiler.step() is called at each chunk start
    pc = args.profile_chunk
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 schedule=schedule(wait=pc - 1, warmup=1, active=1, repeat=1)) as prof:
        timer.prof = prof
        _, tprof = C.run_official(pipe, noise, prompts, timer)
        timer.prof = None
    ka = prof.key_averages()
    cuda_ev = [e for e in ka if e.device_type == torch.autograd.DeviceType.CUDA and not e.key.startswith("ProfilerStep")]
    kern = sum(e.self_device_time_total for e in cuda_ev) / 1e6
    launches = {e.key: e.count for e in ka if e.key in LAUNCH}
    syncs = {e.key: e.count for e in ka if e.key in SYNC}
    memcpy = {e.key: e.count for e in ka if "Memcpy" in e.key or "memcpy" in e.key.lower()}
    top = sorted(cuda_ev, key=lambda e: -e.self_device_time_total)[:25]
    wall = times[-1][pc]
    res.update(profile_chunk=pc, profiled_wall_s=tprof[pc], unprofiled_wall_s=wall, cuda_kernel_s=kern,
               gpu_busy_pct=100 * kern / wall, kernel_launches=sum(launches.values()), launches=launches,
               host_syncs=sum(syncs.values()), syncs=syncs, memcpy=memcpy, n_cuda_kernel_events=sum(e.count for e in cuda_ev),
               top_kernels=[(e.key[:120], e.count, e.self_device_time_total / 1e3) for e in top])
    print(f"chunk {pc}: wall {wall*1e3:.1f} ms (profiled {tprof[pc]*1e3:.1f}), CUDA kernel {kern*1e3:.1f} ms, busy {100*kern/wall:.1f}%,"
          f" launches {sum(launches.values())}, syncs {sum(syncs.values())} {syncs}, memcpy {memcpy}", flush=True)
    for k, n, ms in res["top_kernels"]:
        print(f"  {ms:9.1f} ms  x{n:5d}  {k}")
    json.dump(res, open(args.out, "w"), indent=1)
    print("DONE")


if __name__ == "__main__":
    main()
