"""Check that the Engine's transcribed loop == the official MatrixGame3Pipeline.generate() (bitwise latents), and run
the engine twice to check eager-vs-eager determinism.
  python scripts/mg3/check_official.py --iters 4 --out results/mg3/check_official.json   (run via scripts/mg3/job.sh)
"""
import argparse, json, os, time
import torch
from fastworld import timed as sync_time
from fastworld.models.mg3 import Engine
from fastworld.paths import ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--int8", type=int, default=1)
    ap.add_argument("--out", default=f"{ROOT}/results/mg3/check_official.json")
    a = ap.parse_args()
    if not os.path.isabs(a.out):
        a.out = os.path.join(ROOT, a.out)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    torch.set_grad_enabled(False)
    E = Engine(use_int8=bool(a.int8), num_iterations=a.iters)
    res = {"gpu": torch.cuda.get_device_name(), "iters": a.iters, "int8": a.int8}

    # ---- official generate(), with the VAE decode stubbed out (we compare latents) and no mp4 writing
    import pipeline.inference_pipeline as ip
    from PIL import Image
    from utils.misc import set_seed
    rec = []
    def fake_decode(z, cache, first_chunk=False, **kw):
        rec.append(z.clone()); return torch.zeros(1, 3, 1, 8, 8), cache
    real_decode = E.vae.stream_decode
    E.vae.stream_decode = fake_decode
    ip.process_video = lambda *x, **k: None
    set_seed(E.args.seed)
    t0 = time.time()
    try:
        E.p.generate(E.args.prompt, Image.open(E.args.image).convert("RGB"), max_area=704 * 1280, shift=E.args.sample_shift,
                     num_inference_steps=3, guide_scale=5.0, seed=E.args.seed, use_base_model=False, args=E.args)
    except SystemExit:
        pass
    res["official_s"] = time.time() - t0
    E.vae.stream_decode = real_decode
    print("official done", len(rec), res["official_s"], flush=True)

    # ---- engine, twice
    runs = []
    for r in range(2):
        E.start()
        out, ts = [], []
        for k in range(a.iters):
            den, dt = sync_time(lambda: E.step())
            out.append(den.clone()); ts.append(dt)
        runs.append(out)
        res[f"engine_run{r}_s"] = ts
        print("engine run", r, ts, flush=True)
    res["engine_vs_official_bitwise"] = [bool(torch.equal(x, y)) for x, y in zip(runs[0], rec)]
    res["engine_vs_official_maxdiff"] = [float((x.float() - y.float()).abs().max()) for x, y in zip(runs[0], rec)]
    res["eager_vs_eager_bitwise"] = [bool(torch.equal(x, y)) for x, y in zip(runs[0], runs[1])]
    res["actions_frames"] = int(E.kb_all.shape[1])
    res["keyboard_unique_rows"] = int(torch.unique(E.kb_all[0].float(), dim=0).shape[0])
    res["mouse_unique_rows"] = int(torch.unique(E.ms_all[0].float(), dim=0).shape[0])
    print(json.dumps({k: v for k, v in res.items()}, indent=1), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("DONE")


if __name__ == "__main__":
    main()
