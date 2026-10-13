"""Official VAE decode cost (default `streaming_vae: false` path: one `vae.decode_to_pixel(output)` over the whole
latent video at the end). The streaming path (`vae.model.cached_decode`) does not exist in this release of
wan_5b/modules/vae2_2.py, so streaming_vae=true would raise. Reports total and amortized per-chunk time.
  python scripts/longlive2/vae_time.py --chunks 12 --out results/longlive2/vae_h100.json
"""
import argparse, json, os
import torch
from fastworld import timed
import fastworld.models.longlive2 as C
from fastworld.paths import ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=int, default=12)
    ap.add_argument("--out", default=f"{ROOT}/results/longlive2/vae.json")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    torch.set_grad_enabled(False)
    vae = C.build_vae()
    g = torch.Generator(device="cuda").manual_seed(0)
    lat = torch.randn(1, a.chunks * C.FPB, 48, 44, 80, device="cuda", dtype=torch.bfloat16, generator=g) * 0.5
    ts = []
    for r in range(3):
        v, dt = timed(lambda: vae.decode_to_pixel(lat))
        ts.append(dt)
        print("decode", tuple(v.shape), f"{dt:.2f}s", flush=True)
        del v
    res = {"gpu": torch.cuda.get_device_name(), "chunks": a.chunks, "decode_total_s": ts,
           "per_chunk_s": min(ts[1:]) / a.chunks}
    print(res); json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
