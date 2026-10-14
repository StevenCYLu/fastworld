"""Jetson Thor demo GIF from the side-by-side mp4 written by scripts/mg2/thor_e2e.py: exact runtime (left) vs
FastMG (fp8 + compile) + TensorRT VAE (right), same start image and actions, with each side's measured playthrough FPS.
  python demo/build_thor_sbs.py [--src results/mg2/thor/side_by_side_exact_vs_fast.mp4] [--out demo/demo_thor_sbs.gif]
"""
import argparse, json, os, sys
import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(__file__))
from fonts import FONTB  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=f"{ROOT}/results/mg2/thor/side_by_side_exact_vs_fast.mp4")
    ap.add_argument("--e2e", default=f"{ROOT}/results/mg2/thor/e2e.json")
    ap.add_argument("--out", default=f"{ROOT}/demo/demo_thor_sbs.gif")
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--step", type=int, default=2, help="keep every n-th frame")
    args = ap.parse_args()
    r = json.load(open(args.e2e))
    labels = [f"exact runtime  {r['exact']['play_fps']:.1f} FPS",
              f"FP8 + compile + TensorRT VAE  {r['fast_trt']['play_fps']:.1f} FPS"]
    frames = [f for f in imageio.get_reader(args.src)][::args.step]
    h, w = frames[0].shape[:2]
    W, H = int(w * args.scale) // 2 * 2, int(h * args.scale) // 2 * 2
    bar = 22
    font = ImageFont.truetype(FONTB, 13)
    out = []
    for f in frames:
        im = Image.new("RGB", (W, H + bar), (16, 16, 16))
        im.paste(Image.fromarray(f).resize((W, H), Image.BILINEAR), (0, bar))
        d = ImageDraw.Draw(im)
        for i, t in enumerate(labels):
            d.text((i * W // 2 + 8, 4), t, fill=(235, 235, 235), font=font)
        out.append(np.asarray(im))
    fps = 16 / args.step
    imageio.mimwrite(args.out, out, duration=1000 / fps, loop=0)
    print(f"wrote {args.out}: {len(out)} frames, {W}x{H + bar}, {os.path.getsize(args.out) / 2**20:.1f} MiB")


if __name__ == "__main__":
    main()
