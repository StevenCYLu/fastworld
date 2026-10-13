"""Race GIF: eager MG2 vs the CUDA-graph runtime generating the SAME 16-chunk clip from a standing start, each panel
showing frames as they are produced (measured per-chunk completion times from results/bench.json; frames of a chunk
spread evenly over that chunk's interval). No prefix sharing on either side.
  python demo/build_race.py [--seconds 10] [--fps 12]
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from fonts import FONT, FONTB, MONO, MONOB
from fastworld.paths import ROOT
BG, FG, DIM, CELL = (18, 20, 24), (236, 238, 242), (150, 156, 168), (44, 48, 56)
ACC = {0: (214, 128, 92), 1: (70, 200, 160)}


def events(rec):
    ev, done = [], rec["done"]
    for k in range(len(done)):
        n = 9 if k == 0 else 12
        a, b = (done[k - 1] if k else 0.0), done[k]
        ev += [(a + (i + 1) / n * (b - a), k, i) for i in range(n)]
    return ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=10)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--out", default=f"{ROOT}/demo/demo_race.gif")
    args = ap.parse_args()
    runs = json.load(open(f"{ROOT}/results/mg2/bench.json"))["runs"]
    recs = [runs["rewind_0000|eager_none|wan"][0], runs["rewind_0000|graph_none|wan"][0]]
    z = np.load(f"{ROOT}/results/mg2/frames_rewind_0000.npz")
    fr, off = z["r0"], np.concatenate([[0], np.cumsum(z["r0_n"])])
    H, W = fr.shape[1:3]
    nchunk, nfr = len(recs[0]["done"]), int(z["r0_n"].sum())
    names = [("Matrix-Game-2 as shipped", "eager PyTorch"), ("+ CUDA graphs", "static-shape runtime, one graph per chunk")]
    evs = [events(r) for r in recs]
    fin = [r["done"][-1] for r in recs]
    f_t, f_s, f_m, f_mb, f_big, f_clock = (ImageFont.truetype(FONTB, 15), ImageFont.truetype(FONT, 11), ImageFont.truetype(MONO, 12),
                                            ImageFont.truetype(MONOB, 13), ImageFont.truetype(FONTB, 22), ImageFont.truetype(MONOB, 16))
    pad, top, head, gap = 10, 30, 40, 12
    strip, foot = 18, 40
    CW = pad * 2 + W * 2 + gap
    CH = top + head + H + 8 + strip + foot
    out = []
    for ti in range(int(args.seconds * args.fps) + 1):
        tau = ti / args.fps
        img = Image.new("RGB", (CW, CH), BG)
        d = ImageDraw.Draw(img)
        clock = f"t = {tau:.1f} s"
        d.text((CW // 2 - d.textlength(clock, font=f_clock) // 2, 7), clock, font=f_clock, fill=FG)
        if tau >= max(fin):
            d.text((pad, 8), f"{fin[0] / fin[1]:.2f}x faster · identical frames", font=f_mb, fill=ACC[1])
        else:
            d.text((pad, 9), "same 16-chunk clip, same frames, one H100", font=f_s, fill=DIM)
        for p in range(2):
            x0, y0 = pad + p * (W + gap), top
            d.text((x0, y0), names[p][0], font=f_t, fill=FG)
            d.text((x0, y0 + 19), names[p][1], font=f_s, fill=DIM)
            yv = y0 + head
            shown = [e for e in evs[p] if e[0] <= tau]
            if shown:
                _, k, i = shown[-1]
                img.paste(Image.fromarray(fr[off[k] + i]), (x0, yv))
            else:
                d.rectangle([x0, yv, x0 + W - 1, yv + H - 1], fill=(30, 33, 40))
                d.text((x0 + W // 2 - 34, yv + H // 2 - 7), "generating…", font=f_s, fill=DIM)
            done_chunks = sum(1 for t in recs[p]["done"] if t <= tau)
            if tau >= fin[p]:
                d.rectangle([x0, yv + H - 40, x0 + W - 1, yv + H - 1], fill=(0, 0, 0))
                msg = f"finished in {fin[p]:.1f} s"
                d.text((x0 + W // 2 - d.textlength(msg, font=f_big) // 2, yv + H - 35), msg, font=f_big, fill=ACC[p])
            # chunk progress strip
            ys = yv + H + 8
            cw = (W - (nchunk - 1) * 2) / nchunk
            for c in range(nchunk):
                cx = x0 + c * (cw + 2)
                d.rectangle([cx, ys, cx + cw - 1, ys + strip - 1], fill=ACC[p] if c < done_chunks else CELL)
            yf = ys + strip + 8
            fps = len(shown) / min(tau, fin[p]) if tau > 0.3 else 0.0
            d.text((x0, yf), f"chunk {done_chunks:2d}/{nchunk}   frames {len(shown):3d}/{nfr}", font=f_m, fill=FG)
            per = fin[p] / nchunk * 1e3
            d.text((x0, yf + 17), f"{per:.0f} ms per chunk", font=f_s, fill=DIM)
            s = f"{fps:4.1f} FPS"
            d.text((x0 + W - d.textlength(s, font=f_mb), yf), s, font=f_mb, fill=ACC[p])
        out.append(img.quantize(colors=144, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE))
    out[0].save(args.out, save_all=True, append_images=out[1:], duration=int(1000 / args.fps), loop=0, optimize=True)
    print(args.out, len(out), "frames", f"{os.path.getsize(args.out) / 1e6:.1f} MB", f"{CW}x{CH}", "finish", fin)


if __name__ == "__main__":
    main()
