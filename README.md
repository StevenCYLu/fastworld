# fastworld

**Faster inference for autoregressive video world models, without changing a single output pixel.**

fastworld is a drop-in runtime for released interactive world models (Matrix-Game-2, Matrix-Game-3, LingBot-World-v2,
LongLive-2.0). It re-runs the official model as static-shape CUDA graphs with no host syncs, exposes the session state
so it can be saved, restored and branched in under a millisecond, and adds an exact prefix cache so that a player who
rewinds, or many players who share a start, never pay for the same chunk twice. Everything it produces is **bitwise
identical** to the official code: every chunk, every frame, every rewind.

<p align="center">
  <img src="demo/demo_race.gif" width="672" alt="Matrix-Game-2 as shipped vs the CUDA-graph runtime generating the same clip">
</p>

*Same 16-chunk clip, same frames, one H100. Matrix-Game-2 as shipped (eager PyTorch) finishes in 8.5 s at 22 FPS;
the CUDA-graph runtime finishes the identical clip in 6.0 s at 31.6 FPS.*

<p align="center">
  <img src="demo/demo_rewind_race.gif" width="672" alt="Six rewinds with and without exact prefix sharing">
</p>

*Both sides use CUDA graphs. After a 16-chunk playthrough the player rewinds six times and plays four new chunks each
time. Without sharing, each rewind replays the whole prefix (207 frames re-generated for nothing in 10 s); with the
prefix cache a rewind is a checkpoint restore and only the new chunks are generated.*

## What you get

| Matrix-Game-2, one H100, 352x640, 12 frames per chunk | explore (8 players, shared start) | rewind (6 rewinds) | time to first new frame after a rewind | frames identical |
|---|---|---|---|---|
| Matrix-Game-2 as shipped (eager DiT + Wan VAE) | 22.4 FPS | 22.2 FPS | 5.3 s | reference |
| + CUDA-graph DiT (`StaticMG`) | 31.3 FPS (1.40x) | 31.5 FPS (1.42x) | 5.3 s | yes |
| + exact prefix cache (`PrefixCache`, checkpoint every 2 chunks) | **66.3 FPS (2.96x)** | **73.0 FPS (3.29x)** | **0.38 s** | **yes** |
| same, with the lossy TAEHV decoder instead of the Wan VAE | 105 FPS (4.7x) | 117 FPS (5.3x) | 0.24 s | no (PSNR 24.7 dB vs Wan VAE) |

Per chunk the DiT goes from 369 ms to 229 ms (1.61x; the GPU is then 99 % busy), the session state shrinks from 1.51 GiB
to 0.60 GiB, state save/restore takes 0.8 ms, and all 60 benchmark requests in every exact mode hash to the same frames
as the official pipeline. Numbers: `results/mg2/`; how they were measured: `scripts/mg2/run_bench.py`.

## Why not just `torch.compile`?

We tried the built-in options on the unmodified Matrix-Game-2 forward (`scripts/mg2/compile_baselines.py`, H100):

| | warm chunk | speedup | bitwise | notes |
|---|---|---|---|---|
| eager (as shipped) | 375 ms | 1.00x | reference | ~3,600 kernel launches and ~1,400 host syncs per forward, GPU 62 % busy |
| `torch.cuda.graph()` around a chunk | fails | | | the official loop creates RNG generators and does `.item()` during capture |
| `torch.compile(model)` | 335 ms | 1.12x | no (max diff 4.3) | 65 graph breaks, 77 s compile |
| `mode="max-autotune-no-cudagraphs"` (the Self-Forcing recommendation) | 397 ms | 0.96x | no | 18 min compile |
| `mode="reduce-overhead"` (Inductor + CUDA-graph trees) | 349 ms | 1.06x | no | ~40 cudagraph skips (mutated KV-cache inputs, CPU-side grid sizes), 12 min compile |
| **fastworld `StaticMG`** | **229 ms** | **1.60x** | **yes** | 2 s capture, 0 graph breaks |

The compiled variants drift because Inductor fuses and reorders the floating-point work, and the drift compounds
through the KV cache. fastworld changes no kernels: it only removes the host-side work between them, so the graphs
replay exactly what eager computed.

## How it works

1. **Static shapes.** These models attend over a bounded window (Matrix-Game-2: exactly one chunk of history), so
   from the second chunk on every tensor has a constant shape. The rolling KV caches, whose indices live on the device
   and are read back with `.item()`, are replaced by fixed per-layer history buffers written in place by the
   context forward. Session state = those buffers.
2. **Static inputs.** Everything that changes per chunk (RoPE tables, noise in the official RNG order, action
   windows, image conditioning) is written into static buffers before replay. Per-step constants (timestep
   embeddings, scheduler sigmas) are computed once with the official ops.
3. **No hidden host traffic.** The official forward builds `cu_seqlens` on the CPU for every flash-attention call,
   runs `torch.arange` on the CPU for the timestep embedding and keeps some RoPE tables on the host. These are
   replaced by device constants and direct `flash_attn_varlen_func` calls.
4. **One graph per chunk.** All denoising steps, the scheduler and the context forward are captured as one
   `torch.cuda.CUDAGraph` (two graphs for Matrix-Game-2: first chunk and steady state). `fastworld.graph.capture`
   handles warm-up, pool sharing and restoring the state that capture mutates.
5. **Exact prefix cache.** A radix tree over chunk keys `(image, action)` keeps each chunk's latent and decoded
   frames, plus a full checkpoint (DiT history + decoder state, on GPU) every *N* chunks. Serving a request = return
   the cached frames of the matched prefix, restore the deepest checkpoint, replay at most *N-1* chunks from their
   stored latents (one context forward + one decoder step each), then generate the rest.
6. **Decoder with forkable state.** The Wan2.1 causal VAE decoder runs as a CUDA graph with its 32-tensor feature
   cache in static buffers (exact; it is compute-bound so this is for the state handling, not speed). The TAEHV tiny
   decoder is an optional lossy alternative (10 ms instead of 153 ms per chunk, 36 MiB of state instead of 1 GiB).

Details and gotchas per model are in `docs/models/`; the recipe for a new model is in [`docs/porting.md`](docs/porting.md).

## Does it help other models?

The same runtime was built for three more released world models, with the same exactness checks (official loop
vs ours, with and without graphs, plus save/branch/restore, all bitwise on H100). The gain depends on how
launch-bound the official code is:

| model | setting | official loop, GPU busy | fastworld (bitwise) | `torch.compile` (not bitwise) | doc |
|---|---|---|---|---|---|
| Matrix-Game-2 1.3B | 352x640, 3 steps | 62 % | **1.61x** DiT, 2.96-3.29x served with sharing | 1.12x | [docs/models/mg2.md](docs/models/mg2.md) |
| LingBot-World-v2 1.3B causal-fast | 480x832, 4 steps | 94 % | 1.16x (975 -> 843 ms/chunk) | 1.20x, drift up to 3.7 | [docs/models/lingbot_v2.md](docs/models/lingbot_v2.md) |
| Matrix-Game-3.0 5B distilled | 704x1280, 3 steps, INT8 + FA3 | 98 % | 1.12x (2800 -> 2509 ms/iter) | 1.34x, cos 0.78 by iter 7 | [docs/models/mg3.md](docs/models/mg3.md) |
| LongLive-2.0 5B | 704x1280, 4 UniPC steps | 97.5 % | 1.04x (1758 -> 1688 ms/chunk) | recompiles every chunk, no faster | [docs/models/longlive2.md](docs/models/longlive2.md) |

CUDA graphs pay off when the GPU is idle between launches (Matrix-Game-2). On the compute-bound models the gain comes
from the redundant work the static rewrite exposes: whole-cache clones per forward, cross-attention and camera
features recomputed every denoising step, fp64 RoPE tables rebuilt per call. `torch.compile` is faster there because
it fuses the remaining fp32 elementwise work, at the price of exactness; nothing stops you from compiling the static
forward if you can accept that. What every port gives regardless of speed is an exact, sync-free session engine with
state you can save, restore and branch.

## Quick start (Matrix-Game-2)

```sh
git clone https://github.com/StevenCYLu/fastworld && cd fastworld
pip install -e .
# official code + weights (unmodified), see docs/setup.md
git clone https://github.com/SkyworkAI/Matrix-Game third_party/Matrix-Game-src && ln -s Matrix-Game-src/Matrix-Game-2 third_party/Matrix-Game-2
pip install -r third_party/Matrix-Game-2/requirements.txt && pip install flash-attn --no-build-isolation
huggingface-cli download Skywork/Matrix-Game-2.0 --local-dir third_party/Matrix-Game-2/ckpt
```

```python
import torch
from fastworld import PrefixCache
from fastworld.models.mg2 import MG, Model, StaticMG, StaticVAE

M = Model()                                   # the official pipeline, driven one chunk at a time
M.add_image("beach", f"{MG}/demo_images/universal/0000.png")
S = StaticMG(M); S.capture()                  # static runtime, 4 CUDA graphs, ~2 s
V = StaticVAE()                               # exact Wan VAE with forkable state
cache = PrefixCache(S, V, every=2)            # exact prefix sharing, checkpoint every 2 chunks

path = [("beach", ("w", "u"))] * 8            # 8 chunks: walk forward, mouse still
frames, stats = cache.serve(path)             # list of [12, 3, 352, 640] uint8 tensors
rewind = path[:5] + [("beach", ("a", "l"))] * 3   # rewind to chunk 5, strafe left + look left
frames2, stats2 = cache.serve(rewind)         # 0.4 s to the first new frame, no re-generation of chunks 0-4
```

`scripts/mg2/example.py` is this snippet with video output; `scripts/mg2/test_static.py` reproduces the exactness
tests, `scripts/mg2/run_bench.py` the serving benchmark, `demo/` the GIFs.

Actions are `(keyboard, mouse)` with keyboard in `w s a d q` (q = none) and mouse in `i k j l u` (up, down, left,
right, none), the discrete set the official demo uses; see `fastworld/models/mg2/model.py` to add your own.

## Repository layout

```
fastworld/                 the package
  graph.py                 CUDA-graph capture with state save/restore
  prefix_cache.py          radix-tree prefix cache (model-agnostic)
  common.py                timing, frame hashing, GPU-busy profiling
  paths.py                 where official code and weights live (env-overridable)
  models/mg2/              Matrix-Game-2: official-loop wrapper, static runtime, decoders, workloads
  models/lingbot_v2/       LingBot-World-v2-1.3B
  models/mg3/              Matrix-Game-3.0
  models/longlive2/        LongLive-2.0-5B
scripts/<model>/           exactness tests, profiles, compile baselines, benchmarks, Slurm wrapper
results/<model>/           raw numbers (JSON) behind every table in this repo
docs/                      setup, porting guide, per-model notes
demo/                      GIF builders and the demo page
```

The official model code is never modified or vendored: it is cloned into `third_party/` and imported as is. Each
model has its own environment requirements (torch 2.7-2.10, FlashAttention 2 or 3); see the per-model docs.

## Limitations

- Batch size 1, one session per graph set. Several sessions share the graphs by swapping state (0.8 ms for MG2).
- Exactness is per GPU type: compare eager and fastworld on the same GPU.
- The prefix cache keys on discrete actions; continuous actions need quantisation to share.
- Checkpoints live on the GPU (1.6 GB each for MG2 with the exact decoder, 0.64 GB with TAEHV); the benchmark holds up to
  52 of them on an 80 GB H100. Eviction is not implemented.
- The ports to LingBot, Matrix-Game-3 and LongLive are exact and branchable but only 1.04-1.16x faster; see their docs.

## Acknowledgements

Built on the released code and weights of [Matrix-Game 2 / 3](https://github.com/SkyworkAI/Matrix-Game) (Skywork),
[LingBot-World-v2](https://github.com/Robbyant/lingbot-world-v2) (Robbyant, CC BY-NC-SA 4.0 weights),
[LongLive](https://github.com/NVlabs/LongLive) (NVIDIA), [Wan 2.1 / 2.2](https://github.com/Wan-Video) and
[TAEHV](https://github.com/madebyollin/taehv) (Ollin Boer Bohan). The exact-sharing idea follows prefix caching in LLM
serving (SGLang's RadixAttention), applied to the KV/feature state of causal video diffusion.

## License

MIT for the code in this repository. The models keep their own licenses.
