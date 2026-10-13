"""Minimal end-to-end use of fastworld on Matrix-Game-2: generate an 8-chunk clip, then rewind to chunk 5 and play a
different action, with exact prefix sharing. Writes two mp4s and prints the timings.
  python scripts/mg2/example.py [--image 0000] [--decoder wan|tae]
"""
import argparse, os
import torch
from torchvision.io import write_video
from fastworld import PrefixCache
from fastworld.models.mg2 import MG, Model, StaticMG, StaticVAE, TinyVAE
from fastworld.paths import ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="0000")
    ap.add_argument("--decoder", default="wan", choices=["wan", "tae"])
    ap.add_argument("--out", default=f"{ROOT}/results/mg2")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_grad_enabled(False)

    M = Model(max_chunks=12)                                    # official pipeline, one chunk per call
    M.add_image(args.image, f"{MG}/demo_images/universal/{args.image}.png")
    S = StaticMG(M)
    S.capture()                                                 # 4 CUDA graphs (first/steady x generate/prefill)
    if args.decoder == "wan":
        V = StaticVAE()                                         # exact Wan VAE; its graph needs one eager chunk first
        p = [(args.image, ("w", "u"))] * 2
        V.reset()
        for k in range(2):
            V.decode(S.gen_chunk(p, k), graph=False)
        V.capture()
    else:
        V = TinyVAE()
    cache = PrefixCache(S, V, every=2)

    path = [(args.image, ("w", "u"))] * 8                       # walk forward for 8 chunks (96 frames, ~6 s)
    frames, st = cache.serve(path)
    print(f"playthrough: {st['generated']} chunks generated in {st['time']:.2f} s "
          f"({sum(f.shape[0] for f in frames) / st['time']:.1f} FPS)")
    write_video(f"{args.out}/example_play.mp4", torch.cat(frames).permute(0, 2, 3, 1), fps=16)

    rewind = path[:5] + [(args.image, ("a", "l"))] * 3          # rewind to chunk 5, strafe left + look left
    frames2, st2 = cache.serve(rewind)
    print(f"rewind: {st2['hit']} chunks from cache (checkpoint restored: {bool(st2['restored'])}, replayed {st2['replayed']}), "
          f"{st2['generated']} new chunks; first new frame after {st2['done'][st2['hit']]:.2f} s, total {st2['time']:.2f} s")
    assert all(torch.equal(a, b) for a, b in zip(frames2[:5], frames[:5]))
    write_video(f"{args.out}/example_rewind.mp4", torch.cat(frames2).permute(0, 2, 3, 1), fps=16)
    print("wrote", f"{args.out}/example_play.mp4", f"{args.out}/example_rewind.mp4")


if __name__ == "__main__":
    main()
