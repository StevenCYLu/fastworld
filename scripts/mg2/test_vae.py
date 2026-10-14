"""Decoder side of the frame budget on MG2 latents: eager Wan VAE vs CUDA-graph Wan VAE (exactness, speed, state) vs TAEHV.
  python scripts/mg2/test_vae.py [--chunks 10] [--images 2]
"""
import argparse, json, os
import torch
from fastworld import median as med, timed, write_mp4
from fastworld.models.mg2 import MG, Model, StaticVAE, TinyVAE, test_path
from fastworld.paths import ROOT


def save_mp4(frames, path, fps=16):
    """frames: [T,3,H,W] float in [-1,1]."""
    write_mp4(path, ((frames.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8), fps=fps)


def psnr(a, b):
    mse = ((a.float() - b.float()) / 2).pow(2).flatten(1).mean(1).clamp_min(1e-10)
    return (10 * torch.log10(1 / mse)).mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=int, default=10)
    ap.add_argument("--images", type=int, default=2)
    ap.add_argument("--out", default=f"{ROOT}/results/mg2/vae.json")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.set_grad_enabled(False)
    M = Model(max_chunks=args.chunks + 2)
    imgs = [f"{i:04d}" for i in range(args.images)]
    lats = {}
    for i, img in enumerate(imgs):
        M.add_image(img, f"{MG}/demo_images/universal/{img}.png")
        p = test_path(img, args.chunks, offset=3 * i)
        M.reset(); lats[img] = [M.gen_chunk(p, k).clone() for k in range(len(p))]
    del M.p.kv_cache1
    torch.cuda.empty_cache()
    V = StaticVAE()
    T = TinyVAE()
    import lpips
    lp = lpips.LPIPS(net="alex", verbose=False).cuda().eval()
    res = {"gpu": torch.cuda.get_device_name()}
    # warm-up eager
    V.reset(); [V.decode(l, graph=False) for l in lats[imgs[0]][:3]]
    ref, t_eager = {}, []
    for img in imgs:
        V.reset(); ref[img] = []
        for k, l in enumerate(lats[img]):
            f, dt = timed(lambda: V.decode(l, graph=False)); ref[img].append(f.clone())
            if k: t_eager.append(dt)
    res["vae_state_bytes"] = V.state_bytes()
    # graph
    V.reset(); V.decode(lats[imgs[0]][0], graph=False)
    _, res["capture_s"] = timed(V.capture)
    t_graph, res["graph_bitwise"] = [], {}
    for img in imgs:
        V.reset(); out = []
        for k, l in enumerate(lats[img]):
            f, dt = timed(lambda: V.decode(l)); out.append(f)
            if k: t_graph.append(dt)
        res["graph_bitwise"][img] = [bool(torch.equal(a, b)) for a, b in zip(out, ref[img])]
    print("vae graph bitwise", res["graph_bitwise"], flush=True)
    # fork: save after chunk b-1, decode another session's chunk, restore, continue
    img = imgs[0]; L = lats[img]; b = len(L) // 2
    V.reset(); [V.decode(l) for l in L[:b]]
    st = V.state()
    V.decode(lats[imgs[-1]][b])
    _, res["vae_state_load_s"] = timed(lambda: V.load(st))
    rest = [V.decode(l) for l in L[b:]]
    res["fork_bitwise"] = [bool(torch.equal(a, r)) for a, r in zip(rest, ref[img][b:])]
    print("vae fork", res["fork_bitwise"], flush=True)
    # profile graph chunk
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for l in L[1:4]:
            V.decode(l)
        torch.cuda.synchronize()
    kev = [e for e in prof.key_averages() if getattr(e, "device_type", None) == torch.autograd.DeviceType.CUDA]
    res["graph_kernel_chunk"] = sum(e.self_device_time_total for e in kev) / 3 / 1e6
    res["graph_top"] = [(e.key[:60], round(e.self_device_time_total / 3e3, 2)) for e in sorted(kev, key=lambda e: -e.self_device_time_total)[:8]]
    # TAEHV
    t_tiny, q = [], {"psnr": [], "lpips": []}
    T.reset(); [T.decode(l) for l in lats[imgs[0]][:3]]
    for img in imgs:
        T.reset(); out = []
        for k, l in enumerate(lats[img]):
            f, dt = timed(lambda: T.decode(l)); out.append(f)
            if k: t_tiny.append(dt)
        for a, r in zip(out, ref[img]):
            assert a.shape == r.shape, (a.shape, r.shape)
            q["psnr"].append(psnr(a, r)); q["lpips"].append(float(lp(a, r).mean()))
        if img == imgs[0]:
            save_mp4(torch.cat([torch.cat(ref[img]), torch.cat(out)], -2).cpu(), f"{os.path.dirname(args.out)}/{img}_wanvae_vs_taehv.mp4")
    res["tiny_state_bytes"] = T.state_bytes()
    res.update(t_eager=med(t_eager), t_graph=med(t_graph), t_tiny=med(t_tiny),
               tiny_psnr=sum(q["psnr"]) / len(q["psnr"]), tiny_lpips=sum(q["lpips"]) / len(q["lpips"]))
    print(f"VAE chunk (12 frames 352x640): eager {res['t_eager']*1e3:.1f} ms  graph {res['t_graph']*1e3:.1f} ms "
          f"(kernel {res['graph_kernel_chunk']*1e3:.1f})  TAEHV {res['t_tiny']*1e3:.1f} ms | TAEHV vs Wan VAE PSNR "
          f"{res['tiny_psnr']:.2f} dB LPIPS {res['tiny_lpips']:.3f} | state Wan {res['vae_state_bytes']/2**30:.2f} GiB, "
          f"TAEHV {res['tiny_state_bytes']/2**20:.1f} MiB", flush=True)
    print("top kernels", res["graph_top"])
    json.dump(res, open(args.out, "w"), indent=1)
    print("DONE")


if __name__ == "__main__":
    main()
