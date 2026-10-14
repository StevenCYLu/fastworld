"""Small helpers shared by every runtime and script: wall-clock timing, frame hashing, and a GPU-busy profile."""
import hashlib
import time

import torch


def timed(fn):
    """Run fn() between two device syncs; returns (result, seconds)."""
    torch.cuda.synchronize()
    t0 = time.time()
    r = fn()
    torch.cuda.synchronize()
    return r, time.time() - t0


def median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2]


def frames_hash(frames):
    """sha1 over a list of uint8 CPU frame tensors; equal hashes = bitwise identical video."""
    s = hashlib.sha1()
    for f in frames:
        s.update(f.numpy().tobytes())
    return s.hexdigest()


def gpu_profile(fn, iters=3):
    """How busy is the GPU while fn() runs? Returns per-iteration wall / CUDA-kernel seconds, GPU-busy fraction,
    kernel launches and host syncs. Only DeviceType.CUDA events are summed for kernel time (summing every event
    double-counts the CPU-side launch records)."""
    from torch.profiler import profile, ProfilerActivity
    fn()
    torch.cuda.synchronize()
    t0 = time.time()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    wall = (time.time() - t0) / iters
    ka = prof.key_averages()
    kernel = sum(e.self_device_time_total for e in ka if getattr(e, "device_type", None) == torch.autograd.DeviceType.CUDA) / 1e6 / iters
    launches = sum(e.count for e in ka if e.key in ("cudaLaunchKernel", "cudaGraphLaunch", "cuLaunchKernel", "cudaLaunchKernelExC")) / iters
    syncs = sum(e.count for e in ka if "Synchronize" in e.key or e.key == "cudaMemcpy") / iters
    return dict(wall=wall, kernel=kernel, busy=kernel / wall, launches=launches, syncs=syncs)


def write_mp4(path, frames, fps=16):
    """frames: uint8 [T, 3, H, W] (as served by PrefixCache). H.264 via imageio-ffmpeg: recent torchvision (0.29, the
    one that matches the cu130 torch used on Jetson Thor) no longer has torchvision.io.write_video."""
    import imageio
    imageio.mimwrite(path, frames.permute(0, 2, 3, 1).cpu().numpy(), fps=fps, codec="libx264", quality=8)

