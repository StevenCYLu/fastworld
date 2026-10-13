# Matrix-Game-2: exact CUDA-graph runtime, decoders and prefix cache

Target: [Matrix-Game-2.0](https://github.com/SkyworkAI/Matrix-Game/tree/main/Matrix-Game-2) universal model
(Wan2.1-1.3B causal DiT with a per-block action module, 3 distilled denoising steps, 3 latent frames = 12 pixel frames
per chunk, 352x640, 880 tokens per latent frame, rolling attention window of 6 latent frames = exactly one chunk of
history), driven as `inference_streaming.py` does. Numbers below: H100 80GB, `results/mg2/*.json`.

## Where the eager time goes

One eager chunk (3 denoising forwards + 1 context forward) issues ~3,600 kernel launches and ~1,400 host syncs per
forward: the rolling KV caches keep their indices in device tensors and read them back with `.item()`, the `wan`
`flash_attention` wrapper builds `cu_seqlens` on the CPU and copies them over for every attention, the action module's
RoPE tables live on the CPU, and `sinusoidal_embedding_1d` runs a CPU `arange` per forward. The GPU is busy 62 % of
the time; the kernels themselves take 227 ms of a 369 ms chunk.

## What `StaticMG` does (`fastworld/models/mg2/static.py`)

- **History buffers instead of rolling caches.** Every attention in MG2 (self, mouse, keyboard) sees one chunk of
  history plus the current chunk, so from chunk 1 on all shapes are constant. Per layer: `hk/hv` `[1, 2640, 12, 128]`
  (the previous chunk's context-forward K/V after RoPE), `mk/mv` `[880, 3, heads, 64]` and `kk/kv` `[1, 3, heads, 64]`
  for the action module. The context forward (t = 0) writes them in place; denoising forwards only read.
  Session state = these buffers: 0.60 GiB (eager: 1.51 GiB for the full rolling caches).
- **Static inputs.** `x_in` (initial noise), 2 `noise` buffers (one per `add_noise`), `cc` (image condition slice),
  `vis` (CLIP context), `gm`/`kc` (the 21-pixel-frame mouse / keyboard windows the action module reads), `fi` (main
  RoPE multipliers, complex128 as in `wan`), `acos/asin` (action RoPE), device `cu_seqlens`, `lat_in` (prefill input).
  `_prep(path, k)` fills them with exactly the values the official `conditional_dict` would carry; the test checks
  this against the official action-module slicing.
- **Per-step constants.** Timestep tensors and their sinusoidal embeddings for the 3 steps and t = 0.
  *Gotcha:* MG2's `denoising_step_list` holds **warped floats** (1000, 908.84, 713.98). The forward takes `int(t)`, but
  `add_noise` must get the float step (`steps[i+1] * ones(long)` in eager) or the outputs are not bitwise.
- **Forward.** `_net` is the official block loop written out with the module's own layers (`q/k/v/o`, norms,
  `ffn`, `modulation`, `action_model.*`), `flash_attn_varlen_func` called directly, no `.item()`, no CPU tensors.
  `_chunk(first)` = 3 x (`_net`, `add_noise`) + context `_net`; `_prefill(first)` = context `_net` only.
- **Graphs.** `(first, steady) x (generate, prefill)` = 4 graphs, captured by `fastworld.graph.capture` in 1.8 s.
  `gen_chunk(path, k)` = `_prep` + replay + clone of the output; `prefill(lat, path, k)` advances the state over a
  known latent (58 ms).

### Exactness (`scripts/mg2/test_static.py`, `results/mg2/static.json`)

| check | result |
|---|---|
| prepared action windows / conditions vs the official action-module slicing | equal, 2 images x 10 chunks |
| single forward, chunk 0 | bitwise |
| 10-chunk rollouts, no graph | bitwise 20/20 |
| 10-chunk rollouts, graphs | bitwise 20/20 |
| save after chunk 5, run another action, restore, continue | bitwise 5/5 |

### Speed

| | ms per chunk (median, warm) | kernel ms | GPU busy |
|---|---|---|---|
| eager (official) | 369 | 227 | 62 % |
| `StaticMG`, no graph | ~300 | 227 | |
| `StaticMG` + graph | **229** | 227 | 99 % |

## Decoders (`fastworld/models/mg2/vae.py`, `results/mg2/vae.json`)

- `StaticVAE`: the official Wan2.1 causal VAE decoder (fp16), chunk 0 eager (its feature cache starts empty), every
  later chunk one graph replay with the 32-tensor causal cache in static buffers. Bitwise 20/20, fork 5/5. 157 -> 153 ms
  per 12-frame chunk: the decoder is compute-bound (kernel 155 ms), the graph is for the forkable state (0.99 GiB per
  session, save/load 0.8 ms), not speed.
- `TinyVAE`: [TAEHV](https://github.com/madebyollin/taehv) `taew2_1` streaming decoder. 10.4 ms per chunk, 36 MiB
  of state, lossy: PSNR 24.7 dB / LPIPS 0.098 against the Wan VAE on MG2 latents.

Frame budget per 12-frame chunk: eager 369 + 157 ms (23 FPS); graph + exact VAE 229 + 153 ms (31 FPS, the decoder is
now 40 % of it); graph + TAEHV 229 + 10 ms (50 FPS).

## Prefix cache (`fastworld/prefix_cache.py`, `scripts/mg2/run_bench.py`, `results/mg2/bench.json`)

Workloads from 2 start images: *explore* (8 players sharing the start, branch factor 2 every 4 chunks, 16 chunks each)
and *rewind* (one 16-chunk playthrough, then rewind to chunks 4, 6, ..., 14 and play 4 new chunks). All 60 requests in
every exact mode hash to the frames of the official pipeline.

| mode | explore FPS | rewind FPS | time to first new frame after a rewind | checkpoint bytes |
|---|---|---|---|---|
| eager DiT + eager Wan VAE (as shipped) | 22.4 | 22.2 | 5.29 s | |
| graph DiT, no sharing | 31.3 (1.40x) | 31.5 (1.42x) | 5.29 s | |
| graph + sharing, checkpoint every 4 chunks | 66.4 (2.97x) | 67.6 (3.05x) | 0.59 s | 1.6 GB |
| graph + sharing, checkpoint every 2 chunks | 66.3 (2.96x) | 73.0 (3.29x) | 0.38 s | 1.6 GB |
| graph + TAEHV, no sharing | 49.8 | 50.4 | | |
| graph + TAEHV + sharing every 2 | 105.4 (4.71x) | 117.3 (5.28x) | 0.24 s | 0.64 GB |
| graph + TAEHV + sharing every 1 | 104.3 | 116.2 | 0.24 s | |

The sharing gains sit at the cost-weighted ideal (graph speedup x fraction of chunks that had to be generated); the
replay of up to *N-1* chunks after a checkpoint costs 58 ms (prefill) + decode each, which is why checkpointing every
2 chunks beats every 4 on rewinds.

## Built-in alternatives (`scripts/mg2/compile_baselines.py`, `results/mg2/compile/`)

See the README table. All `torch.compile` modes hit 65 graph breaks on the official forward (40 `Tensor.item`, 17
data-dependent ops, 4 data-dependent branches), none is bitwise (max abs diff 4.1-4.3 on the latents, growing through
the cache), and `reduce-overhead` skips ~40 cudagraphs because the KV-cache inputs are mutated and `grid_sizes` is a
CPU tensor. A naive `torch.cuda.graph()` around the eager chunk fails during capture (RNG generator construction).

## Caveats

- Batch 1, universal model only (the TempleRun weights use the same code and should work; not tested).
- Actions are the discrete keyboard/mouse set in `model.KEYS` / `model.CAM`; continuous mouse deltas work with the
  runtime but need quantisation to be shareable.
- Checkpoints are kept on the GPU without eviction.
