# LingBot-World-v2-1.3B-causal-fast

Exact static-shape / CUDA-graph runtime for `robbyant/lingbot-world-v2-1.3b-causal-fast`, following the fastworld
method of the Matrix-Game-2 runtime (`fastworld/models/mg2/static.py`).

Code: `fastworld/models/lingbot_v2/` -- `engine.py` (official preamble verbatim + the official chunk-loop body, one
chunk per call; checked bitwise against unmodified `pipe.generate()`), `static.py` (`StaticLingbot`, the runtime).
Scripts: `scripts/lingbot_v2/{test_static,profile_chunk,compile_baseline}.py`, Slurm wrapper `scripts/lingbot_v2/job.sh`.
Numbers: `results/lingbot_v2/*.json`.

## Setup
```sh
# official code (unmodified); fastworld.paths.LINGBOT_DIR, override with FASTWORLD_LINGBOT_DIR
git clone --depth 1 https://github.com/Robbyant/lingbot-world-v2 third_party/lingbot-world-v2
# DiT weights (the 1.3B HF repo ships the DiT only); LINGBOT_CKPT / FASTWORLD_LINGBOT_CKPT
python -c "from huggingface_hub import snapshot_download as s; s('robbyant/lingbot-world-v2-1.3b-causal-fast', local_dir='weights/lingbot-world-v2-1.3b-causal-fast')"
# T5 encoder (umt5-xxl), Wan2.1_VAE.pth and the google/umt5-xxl tokenizer come from a Wan2.1-T2V-1.3B checkout,
# passed to the pipeline as assets_dir; WAN21_DIR / FASTWORLD_WAN21_DIR
python -c "from huggingface_hub import snapshot_download as s; s('Wan-AI/Wan2.1-T2V-1.3B', local_dir='weights/Wan2.1-T2V-1.3B')"
```
Environment: torch 2.7.1 (cu126), flash-attn 2.8.3 (FA2; no FA3), plus the official `requirements.txt`:
```sh
python -m venv .venv-lingbot && . .venv-lingbot/bin/activate
pip install torch==2.7.1 torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r third_party/lingbot-world-v2/requirements.txt
pip install flash-attn==2.8.3 --no-build-isolation
```
Runs (the wrapper sets `PYTHONPATH` to the repo root and `cd`s there; `FASTWORLD_PY` picks the interpreter; pass the
account/partition/QOS of your cluster; optional `scripts/lingbot_v2/env.sh` is sourced for cache variables such as
`HF_HOME`, `TRITON_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR`):
```sh
export FASTWORLD_PY=$PWD/.venv-lingbot/bin/python
sbatch -A <acct> -p <gpu> --gres=gpu:1 -c14 --time=01:00:00 -J lb_test scripts/lingbot_v2/job.sh scripts/lingbot_v2/test_static.py --chunks 12 --decode
sbatch ... scripts/lingbot_v2/job.sh scripts/lingbot_v2/profile_chunk.py          # eager vs static: wall/kernel/launches/syncs of a steady chunk
sbatch ... scripts/lingbot_v2/job.sh scripts/lingbot_v2/compile_baseline.py --mode compile
```
Outputs default to `results/lingbot_v2/static_<gpu>.json`, `profile_<gpu>.json`, `compile/<mode>_<gpu>.json`
(`--out` overrides). The 12-chunk test needs ~48 GB (fits an L40S); `--chunks 6` is the smallest run that fills the
window and exercises all 5 graphs.

## Model facts (verified in code)
- Wan2.1-1.3B I2V DiT (30 layers, 12 heads x 128, in_dim 36) + per-block camera injector (4 extra 1536x1536 linears
  per block + a Plücker patch embedding). Conditioning in the fast path = image (VAE latent + mask), one T5 prompt,
  and camera poses -> Plücker. `wasd_action` is hard-coded `None` in `_generate_causal_fast` (actions/text events are
  not used by the released 1.3B fast path).
- `generate.py` defaults: chunk_size = **4** latent frames (not 3), 4 steps with `timesteps_index=[0,250,500,750]` of a
  1000-step shift-5 schedule, `--local_attn_size 18 --sink_size 6`. The examples are 1280x720 -> latent 58x104,
  1508 tokens/frame, 6032 tokens/chunk.
- KV cache per layer = 18 frames *including* the 6 sink frames, i.e. steady attention keys = [6 sink | 8 rolled | 4 current]
  = 27,144 keys. Eager rolls with `.item()` bookkeeping and `clone()`: chunk 4 shifts [8:16]->[6:14] (2 frames evicted),
  chunks >= 5 shift [10:18]->[6:14]. Keys are stored **already RoPE'd at their absolute frame index** and are moved
  without re-roping; queries use the absolute index `4*chunk` (RoPE table has 1024 frames => eager and ours both stop at
  chunk 256).
- Global camera normalization: `compute_relative_poses(normalize_trans=True)` divides by the max translation over the
  *whole* trajectory, so per-chunk Plücker inputs depend on the full (fixed) path. We take the official per-chunk
  Plücker tensors as inputs.

## What the static runtime does (`fastworld/models/lingbot_v2/static.py`)
- Same per-layer K/V buffers as eager ([1, 18*1508, 12, 128]). A host-side simulation of eager's index bookkeeping
  gives per chunk `(write_offset, key_len, post_roll)`; 5 distinct plans (chunks 0,1,2,3 and steady). The shift eager
  does at the start of chunk c+1 is done at the end of chunk c, inside the graph, as non-overlapping in-place copies
  (half the traffic of eager's clone+copy). Key order is identical to eager (needed for bitwise FA2 output).
- One CUDA graph per plan (captured with `fastworld.graph.capture`: warm-up on a side stream, one shared pool):
  [hoisted camera features] + 4 x (DiT forward, x0, add_noise) + KV-update forward + shift.
- Static input buffers (filled outside the graph): latent noise slice, image-condition slice, Plücker slice, RoPE
  multiplier table, the 3 add_noise noises (session generator, eager order). Device cu_seqlens; flash_attn varlen
  called directly (wan's wrapper builds cu_seqlens on CPU and H2D-copies them per call).
- Constants computed once with eager's exact ops: time embeddings for the 5 timesteps, x0/add_noise sigmas
  (eager does `nonzero().item()` + CPU->GPU copies per step), prompt cross-attn K/V.
- Dead work dropped: per-forward `text_embedding` (unused once the cross cache is filled), head+unpatchify of the
  KV-update forward (output discarded), and the float64 `cat`/`stack` copies in `causal_rope_apply`.
- `hoist=True` (optional): the camera branch depends only on the chunk's camera, not on t, so it is computed once per
  chunk instead of 5 times (~4 x 9.4 M of the ~51 M params/block of GEMM work per token). Bitwise identical.
- Session state = live prefix of the K/V buffers (14 of 18 frames in steady state) + generator state.

## Results
### Exactness (L40S, `results/lingbot_v2/static_L40_smoke.json`; 2 sessions x 12 chunks, examples 00 and 02, camera
varies: still -> forward -> up to 40 deg/chunk yaw; window rolls from chunk 4, sink exercised)
| check | result |
|---|---|
| engine vs unmodified `pipe.generate()` latents (16x48x58x104) | bitwise |
| eager vs eager | bitwise 12/12 |
| RoPE table vs `causal_rope_apply` | bitwise |
| single forward chunk 0 | bitwise |
| static no-graph, hoist off / on | bitwise 24/24 / 24/24 |
| static CUDA graph, hoist off / on | bitwise 24/24 / 24/24 |
| save state after chunk 6, run chunk 6 with another camera, restore, continue | bitwise 6/6 (alt chunk differs) |

H100 (`results/lingbot_v2/static_H100.json`): identical picture -- engine vs `pipe.generate()`, eager vs eager, RoPE,
fwd0, no-graph and graph (hoist off/on, 24/24 each) and branch restore (6/6) all bitwise.

### Speed (steady chunk = window full; DiT + scheduler, median, warm)
L40S (48 GB), `results/lingbot_v2/profile_L40S.json`, `results/lingbot_v2/static_L40_smoke.json`,
`results/lingbot_v2/compile/compile_L40S.json`:

| variant | ms/chunk (wall) | CUDA kernel ms | GPU busy | launches | host syncs | speedup | bitwise |
|---|---|---|---|---|---|---|---|
| eager (official loop) | 2056 (test run: 2071) | 2016 | 98.1 % | 14,132 | 580 (562 `.item()`) | 1.00x | ref |
| static, no graph | 1891 | 1889 | 99.9 % | 10,965 | 0 | 1.09x | yes |
| static + CUDA graph | 1892 (test run: 1885) | 1895 | 100 % | 1 graph | 0 | 1.09x (1.10x) | yes |
| static + hoist, no graph | 1805 | 1797 | 99.6 % | 10,101 | 0 | 1.14x | yes |
| static + hoist + CUDA graph | 1786 (test run: 1780) | 1788 | 100 % | 1 graph | 0 | 1.15x (1.16x) | yes |
| `torch.compile(model)` default (eager loop) | 1822 | - | - | - | - | 1.13x | **no** (max diff 0.5 -> 4.1, grows over chunks) |

torch.compile: 26 graph breaks (18 `Tensor.item` in the cache bookkeeping, data-dependent shape in
`causal_rope_apply`), compile overhead 31 s on the first 12 chunks. Warm-up chunks 0-3 (shorter key windows):
eager 1485/1645/1812/1980 ms, static+hoist+graph 1199/1361/1526/1697 ms. Capture of the 5 graphs: 9.5 s.

Where the eager time goes (steady chunk, L40S): FA2 self-attention 755 ms (37 %; 27k keys x 6k queries x 5
forwards), GEMMs ~500 ms, ~640 ms of elementwise / LayerNorm / dtype-cast kernels (fp32 modulation, float64 RoPE), and
**127 ms of device-to-device memcpys** (1,030/chunk) that the static runtime does not issue: float64 `cat`/`stack`
copies in `causal_rope_apply` (2 x 74 MB per q/k per layer per forward) and the clone-based roll. Eager kernel time
minus static kernel time = 127 ms, i.e. the whole no-hoist gain is removed work, not launch overhead: static with and
without a graph run at the same speed (1891 vs 1892 ms) because the GPU is already ~98 % busy in eager.

H100 80GB HBM3 (`results/lingbot_v2/static_H100.json`, `results/lingbot_v2/profile_H100.json`,
`results/lingbot_v2/compile/compile_H100.json`):

| variant | ms/chunk (test run / profile run) | CUDA kernel ms | GPU busy | launches / syncs | speedup | bitwise |
|---|---|---|---|---|---|---|
| eager (official loop) | 975 / 982 | 920 | 93.7 % | 14,132 / 580 | 1.00x | ref |
| static, no graph | - / 883 | 869 | 98.4 % | 10,965 / 0 | 1.11x | yes |
| static + CUDA graph | 871 / 873 | 872 | 99.9 % | 1 graph / 0 | 1.12x | yes |
| static + hoist, no graph | - / 860 | 844 | 98.2 % | 10,101 / 0 | 1.14x | yes |
| static + hoist + CUDA graph | 843 / 847 | 844 | 99.7 % | 1 graph / 0 | 1.16x | yes |
| `torch.compile(model)` default | 816 (its eager ref 976) | - | - | 26 graph breaks | 1.20x | **no** (max diff 0.54 -> 3.7) |

VAE decode (Wan2.1, whole 189-frame video at once, as the official code does): 7.27 s = 606 ms per chunk amortized.
End-to-end per chunk incl. decode: eager ~1581 ms vs static+hoist+graph ~1449 ms (1.09x).
Warm-up chunks 0-3: eager 635/730/825/922 ms, static+hoist+graph 496/591/692/791 ms. Capture 5.2-5.4 s.
Save/load 3-8 ms. Breakdown: eager wall - kernel = 62 ms of CPU overhead (hidden partly); eager kernel - static kernel
= 51 ms of removed device work (RoPE float64 copies 37 ms DtoD + dead work); the graph itself adds ~1 % (883 -> 873 ms).

### Session state
Eager: 4.75 GiB (self K/V 18 frames x 30 layers + cross K/V). Ours: 3.62 GiB (live 14 frames; cross K/V is
per-prompt and shared). Save/load 12 ms (L40S), 3-8 ms (H100).

## Verdict
Exact and works: bitwise-identical to the official loop (and to unmodified `pipe.generate()`) for every chunk of two
12-chunk sessions with a rolling window + sink, with and without CUDA graphs, including state save/restore/branch.

But this model is **compute-bound**, so CUDA graphs themselves buy little. On L40S eager is 98 % GPU-busy and
the graph adds nothing (1891 vs 1892 ms); on H100 eager is 93.7 % busy (14k launches, 580 `.item()` syncs hidden behind
0.92 s of kernels; FA2 over 27k keys is ~49 % of kernel time) and the graph adds ~1 % (883 -> 873 ms). Most of the
gain is deleted device work (float64 RoPE copies, clone-based roll, dead text-embedding/head work: 1.11x on H100,
1.09x on L40S); hoisting the t-independent camera branch out of the 4+1 forwards adds ~3-6 % (1.16x total on both GPUs,
bitwise). torch.compile on the unmodified loop is faster on H100 (1.20x, 18 s compile overhead) but not bitwise
(drift up to 3.7) and has 26 graph breaks; with VAE decode included, our end-to-end gain is ~1.09x on H100.
Bigger wins need fewer FLOPs, not fewer launches: FP8 GEMMs / Sage attention (as NVIDIA flashdreams does, not
exact), or compiling the static forward for fused elementwise (not tried; would give up bitwise equality).

Caveats: the 1.3B HF repo ships DiT only (fp32 safetensors, cast to bf16 on load); T5/VAE/tokenizer come from the
Wan2.1-T2V-1.3B folder (`WAN21_DIR`). The official `wan` package initialises CUDA at import time, so the package can only
be imported on a GPU host. FA2 is what `wan.modules.attention` picks when FA3 is absent (numbers above are FA2).
flashdreams' LingBot-v2-1.3B config uses `use_cuda_graph=False`, `torch.compile(dynamic=True)`, Sage attention and
row-wise FP8 linears with a Python-int `BlockKVCache` (no `.item()`); not run here (not exact, extra deps).
