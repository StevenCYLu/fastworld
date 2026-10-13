"""Race GIF, both sides on the CUDA-graph runtime: no sharing vs exact prefix sharing (checkpoint every 2 chunks).
The clock starts after the shared 16-chunk playthrough; the player then rewinds 6 times (to chunks 4, 6, ..., 14) and
plays 4 new chunks each time. Frames shown on each side's measured per-chunk timeline (results/bench.json); chunks
served from the prefix cache are not replayed. Same frames on both sides.
  python demo/build_rewind_race.py [--seconds 10] [--fps 12]
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from fonts import FONT, FONTB, MONO, MONOB
from fastworld.paths import ROOT
BG, FG, DIM, CELL = (18, 20, 24), (236, 238, 242), (150, 156, 168), (44, 48, 56)
REGEN, NEW, RESTORE = (240, 110, 90), (70, 200, 160), (250, 200, 80)
SIDE = {0: (120, 150, 210), 1: NEW}


def events(recs, rewind_t):
    ev, rest, T = [], [], 0.0
    for j, r in enumerate(recs):
        if j == 0:
            continue
        done, hit = r["done"], r["hit"]
        if hit:
            rest.append((T + done[hit - 1], j))
        for k in range(hit, len(done)):
            n = 9 if k == 0 else 12
            a, b = T + (done[k - 1] if k else 0.0), T + done[k]
            st = "regen" if k < rewind_t[j] else "new"
            ev += [(a + (i + 1) / n * (b - a), j, k, i, st) for i in range(n)]
        T += r["time"]
    return ev, rest, T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=10)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--out", default=f"{ROOT}/demo/demo_rewind_race.gif")
    args = ap.parse_args()
    runs = json.load(open(f"{ROOT}/results/mg2/bench.json"))["runs"]
    A, B = runs["rewind_0000|graph_none|wan"], runs["rewind_0000|graph_ckpt2|wan"]
    rewind_t = [0] + [r["hit"] for r in B[1:]]
    nrw = len(B) - 1
    z = np.load(f"{ROOT}/results/mg2/frames_rewind_0000.npz")
    fr = {j: z[f"r{j}"] for j in range(len(B))}
    off = {j: np.concatenate([[0], np.cumsum(z[f"r{j}_n"])]) for j in range(len(B))}
    H, W = fr[0].shape[1:3]
    sides = [("CUDA graphs", "no sharing: each rewind replays from chunk 0", *events(A, rewind_t)),
             ("CUDA graphs + prefix sharing", "ours: restore checkpoint, generate only new chunks", *events(B, rewind_t))]
    f_t, f_s, f_m, f_mb, f_big, f_clock, f_bd = (ImageFont.truetype(FONTB, 15), ImageFont.truetype(FONT, 11), ImageFont.truetype(MONO, 12),
                                                 ImageFont.truetype(MONOB, 13), ImageFont.truetype(FONTB, 20), ImageFont.truetype(MONOB, 16),
                                                 ImageFont.truetype(FONTB, 12))
    pad, top, head, gap, strip, foot = 10, 30, 40, 12, 18, 40
    CW, CH = pad * 2 + W * 2 + gap, top + head + H + 8 + strip + foot
    out = []
    last_frame = [fr[0][-1], fr[0][-1]]                      # both sides start on the end of the playthrough
    for ti in range(int(args.seconds * args.fps) + 1):
        tau = ti / args.fps
        img = Image.new("RGB", (CW, CH), BG)
        d = ImageDraw.Draw(img)
        clock = f"t = {tau:.1f} s"
        d.text((CW - pad - d.textlength(clock, font=f_clock), 7), clock, font=f_clock, fill=FG)
        if tau >= sides[1][4]:
            d.text((pad, 8), f"same 6 rewinds without sharing: {sides[0][4]:.1f} s", font=f_mb, fill=NEW)
        else:
            d.text((pad, 9), "after a 16-chunk playthrough: 6 rewinds, 4 new chunks each", font=f_s, fill=DIM)
        for p, (name, sub, ev, rest, total) in enumerate(sides):
            x0, y0 = pad + p * (W + gap), top
            d.text((x0, y0), name, font=f_t, fill=FG)
            d.text((x0, y0 + 19), sub, font=f_s, fill=DIM)
            yv = y0 + head
            shown = [e for e in ev if e[0] <= tau]
            if shown:
                _, j, k, i, st = shown[-1]
                frame = fr[j][off[j][k] + i]
            else:
                j, k, st, frame = 1, 0, "regen", last_frame[p]
            img.paste(Image.fromarray(frame), (x0, yv))
            fin = tau >= total
            cur = min(j + (1 if shown and shown[-1][0] >= max(e[0] for e in ev if e[1] == j) and not fin else 0), nrw)
            recent = [r for r in rest if r[0] <= tau and tau - r[0] < 0.45]
            if fin:
                label, c = "all 6 rewinds done", NEW
            elif recent:
                label, c = f"rewind {recent[-1][1]}: checkpoint restored (<1 ms)", RESTORE
            elif st == "regen":
                label, c = f"rewind {j}: re-generating chunks 0–{rewind_t[j] - 1} again", REGEN
            else:
                label, c = f"rewind {j}: new chunks from {rewind_t[j]}", NEW
            tw = d.textlength(label, font=f_bd)
            d.rectangle([x0 + 6, yv + 6, x0 + 16 + tw, yv + 26], fill=c)
            d.text((x0 + 11, yv + 9), label, font=f_bd, fill=(12, 14, 18))
            if fin:
                d.rectangle([x0, yv + H - 38, x0 + W - 1, yv + H - 1], fill=(0, 0, 0))
                msg = f"6 rewinds in {total:.1f} s"
                d.text((x0 + W // 2 - d.textlength(msg, font=f_big) // 2, yv + H - 33), msg, font=f_big, fill=SIDE[p])
            # one cell per rewind, filled when that rewind's new chunks are done
            done_rw = sum(1 for jj in range(1, nrw + 1) if any(e[1] == jj for e in ev) and max(e[0] for e in ev if e[1] == jj) <= tau)
            ys = yv + H + 8
            cw = (W - (nrw - 1) * 4) / nrw
            for c_ in range(nrw):
                cx = x0 + c_ * (cw + 4)
                d.rectangle([cx, ys, cx + cw - 1, ys + strip - 1], fill=SIDE[p] if c_ < done_rw else CELL)
                d.text((cx + cw / 2 - 3, ys + 2), str(c_ + 1), font=f_s, fill=(12, 14, 18) if c_ < done_rw else DIM)
            yf = ys + strip + 8
            nnew = sum(1 for e in shown if e[4] == "new")
            nreg = sum(1 for e in shown if e[4] == "regen")
            d.text((x0, yf), f"rewinds {done_rw}/{nrw}   new frames {nnew:3d}", font=f_m, fill=FG)
            d.text((x0, yf + 17), f"frames re-generated for nothing: {nreg}", font=f_s, fill=REGEN if nreg else DIM)
        out.append(img.quantize(colors=144, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE))
    out[0].save(args.out, save_all=True, append_images=out[1:], duration=int(1000 / args.fps), loop=0, optimize=True)
    print(args.out, len(out), "frames", f"{os.path.getsize(args.out) / 1e6:.1f} MB", f"{CW}x{CH}",
          "totals", [round(s[4], 2) for s in sides])


if __name__ == "__main__":
    main()
