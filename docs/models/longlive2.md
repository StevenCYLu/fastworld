# LongLive-2.0-5B: exact static-shape CUDA-graph runtime

Target: [NVlabs LongLive 2.0](https://github.com/NVlabs/LongLive) (`LongLive2.0` subdir, commit 290be9b), checkpoint
`Efficient-Large-Model/LongLive-2.0-5B` (`model_bf16.pt`, merged DMD LoRA), BF16 path of `configs/inference.yaml`:
Wan2.2-TI2V-5B causal DiT (30 layers, dim 3072, 24 heads), 704x1280 (44x80 latent, 880 tokens/frame), 8 latent frames
= 7040 tokens per chunk, 4 UniPC steps, local attention 32 frames (28160-token KV cache), sink 8 frames,
`multi_shot_sink`, multi-shot RoPE offset 8, guidance 1.0. NVFP4 not used (H100).

Code: `fastworld/models/longlive2/` -- `model.py` (the official pipeline built from `fastworld.paths`, prompt schedule,
noise, per-chunk timing hooks around the unmodified `CausalDiffusionInferencePipeline.inference` loop), `static.py`
(`StaticLL`, the runtime; `StaticUniPC`). Scripts: `scripts/longlive2/{test_static,eager_profile,compile_baseline,vae_time}.py`,
Slurm wrapper `scripts/longlive2/job.sh` (+ optional `env.sh` for cache variables). Numbers: `results/longlive2/*.json`.

## Setup

Locations (all overridable, see `fastworld/paths.py`): `LONGLIVE_DIR` = `third_party/LongLive` (`FASTWORLD_LONGLIVE_DIR`),
`LONGLIVE_CKPT` = `weights/LongLive-2.0-5B` (`FASTWORLD_LONGLIVE_CKPT`), `WAN22_DIR` = `weights/Wan2.2-TI2V-5B`
(`FASTWORLD_WAN22_DIR`).

```bash
git clone https://github.com/NVlabs/LongLive.git third_party/LongLive                      # code (LongLive/LongLive2.0)
uv venv --python 3.10 .venv-longlive2
uv pip install --python .venv-longlive2/bin/python torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv-longlive2/bin/python \
  https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp310-cp310-linux_x86_64.whl \
  diffusers==0.31.0 "transformers<5" accelerate tqdm imageio imageio-ffmpeg easydict ftfy omegaconf einops av==13.1.0 \
  opencv-python-headless peft torchao==0.13.0 sentencepiece "numpy<2"
hf download Efficient-Large-Model/LongLive-2.0-5B model_bf16.pt --local-dir weights/LongLive-2.0-5B      # 10 GB
hf download Wan-AI/Wan2.2-TI2V-5B Wan2.2_VAE.pth config.json models_t5_umt5-xxl-enc-bf16.pth \
   --include "google/umt5-xxl/*" --local-dir weights/Wan2.2-TI2V-5B                                     # 2.8 GB + umT5 11 GB
```

`WAN22_DIR` must hold `config.json`, `Wan2.2_VAE.pth`, `models_t5_umt5-xxl-enc-bf16.pth` and `google/umt5-xxl/` (the umT5
encoder/tokenizer are the same files as in Wan2.1-T2V-1.3B; symlinks are fine). The 20 GB base Wan2.2 DiT safetensors
are NOT needed: `model.py` builds `CausalWanModel` from `config.json` and loads `model_bf16.pt` strictly. The official
loaders open `wan_models/Wan2.2-TI2V-5B/...` relative to the working directory; `model.official_cwd()` points that at
`WAN22_DIR` through a throwaway symlink directory while a model is built, so no particular cwd is required.

Runs (the wrapper sets `PYTHONPATH` to the repo root and `cd`s there; `FASTWORLD_PY` picks the interpreter; pass the
account/partition of your cluster; `scripts/longlive2/env.sh` is sourced if present and sets `TRITON_CACHE_DIR`,
`TORCHINDUCTOR_CACHE_DIR`, `HF_HUB_OFFLINE`, each only if not already exported):

```bash
export FASTWORLD_PY=$PWD/.venv-longlive2/bin/python
sbatch -A <acct> -p <gpu> --gres=gpu:1 -c14 --time=01:00:00 -J ll2_static scripts/longlive2/job.sh \
       scripts/longlive2/test_static.py --chunks 12 [--recache 1] --out results/longlive2/static_h100.json
sbatch ... scripts/longlive2/job.sh scripts/longlive2/eager_profile.py --chunks 12 --profile_chunk 6 --out results/longlive2/eager_profile_h100.json
sbatch ... scripts/longlive2/job.sh scripts/longlive2/compile_baseline.py --chunks 12 --runs 3 --out results/longlive2/compile_h100.json
sbatch ... scripts/longlive2/job.sh scripts/longlive2/vae_time.py --chunks 12 --out results/longlive2/vae_h100.json
```

Outputs default to `results/longlive2/{static,eager_profile,compile,vae}.json`. The 12-chunk test fits a 48 GB L40S
(~11 min there, including the three eager reference runs); `--chunks 6` is the smallest run that fills the window,
takes a scene cut and checks every graph bucket.

## Test trajectory

12 chunks (96 latent frames, ~16 s of video), 3 prompts in the official multi-shot convention: chunks 0-3 prompt A,
chunk 4 `"The scene transitions. " + B`, 5-7 B, chunk 8 cut to C, 9-11 C. This exercises: growing cache (chunks 0-3,
4 shapes), rolling with sink, scene cuts (RoPE temporal offset, pinned shot-sink, floating pin shifted by eviction,
pin merging into the protected prefix, a second cut replacing the first pin), prompt changes. Fixed seed 0 noise.
Also run with `shot_clean_recache: true` (`--recache 1`).

## Numbers

All runs: 12 chunks, prompt switches at chunks 4 and 8; "steady" = median over chunks 4-11 (full 28160-token window)
of all timed runs. DiT-only time per chunk = 4 denoising forwards + UniPC + context forward (what the official loop does
per chunk; text encoding happens once before the loop and is excluded for both).

### H100 80GB HBM3 (eager and static in the same process; `results/longlive2/static_h100.json`, `results/longlive2/eager_profile_h100.json`)

| | eager (official) | static, no graph | **static + CUDA graph** | torch.compile (official setting) |
|---|---|---|---|---|
| steady ms/chunk (DiT only) | 1757.5 | 1692.9 | **1688.1** | see L40S (H100 run pending) |
| speedup vs eager | 1.00x | 1.038x | **1.041x** | |
| chunk 0 / 1 / 2 / 3 ms (window 1-3 chunks) | 1025 / 1265 / 1505 / 1751 | 987 / 1205 / 1445 / 1688 | 963 / 1199 / 1440 / 1683 | |
| CUDA kernel time per steady chunk | 1731 ms | | 1689 ms | |
| GPU busy (kernel / wall) | **97.5 %** | | ~100 % | |
| kernel launches / host syncs per chunk | 11,457 / 59 (+633 pageable H2D, 24 D2H) | | 1 graph launch + 11 small copies / 0 | |
| bitwise vs eager (12 chunks) | eager==eager: 12/12 | 12/12, all 48 flow preds | 12/12 (x2 rollouts) | no (max abs diff 6-10) on L40 |
| branch save / restore / continue | | | 5/5 bitwise, save 9 ms, load 7 ms | |
| graph capture (4 buckets) | | | 12.0 s | |
| session state | 21.1 GB allocated (pos 10.4 + unused neg 10.4 + cross caches); 10.6 GB w/o neg | | 10.4 GB resident buffers; 7.8 GB live state (evicted slot is dead) | |

Eager profile of steady chunk 6 (H100): flash-attn fwd 986 ms (300 calls; self-attn 7040x28160 dominates), GEMMs
~396 ms, cache clones `Memcpy DtoD` 75 ms (1470 copies), fp32 RMSNorm/elementwise ~200 ms, RoPE 27 ms, adaLN 10 ms.
FLOPs: ~70 TFLOP linear + ~73 TFLOP attention per forward, x5 per chunk.

VAE (official default: one batch `decode_to_pixel` of the whole 96-frame video at the end, identical for both paths;
`scripts/longlive2/vae_time.py`, `results/longlive2/vae_{h100,l40}.json`): H100 31.4 s for the whole video, 3.57 s amortized per
chunk (the two repeat decodes took 42.9 / 49.7 s); L40S 57.7 s, 4.81 s per chunk. Not included in the table above.

### L40S 48GB (debug GPU; `results/longlive2/static_l40_smoke.json`, `static_l40_recache.json`, `compile_l40.json`)

| | eager | static no graph | static + graph | torch.compile |
|---|---|---|---|---|
| steady ms/chunk | 4102 | 3933 (1.043x) | 3897 (1.053x) | 4128 warm-median (0.99x); chunks 0-3 2.37/2.94/3.43/3.94 s vs eager 2.71/3.11/3.59/4.04 s |
| bitwise | eager==eager 10/10 | 10/10 | 10/10 x2 | no, max abs diff 5.8-10.3 |
| `shot_clean_recache: true` variant | | 10/10 | 10/10 x2 (cut chunks run graph-less) | |
| compile time | | | 27 s capture | first run 831 s (+109 s second run) |

## What the official eager path does per chunk (and what changed)

Per chunk: 4 denoising forwards + 1 clean context forward (timestep 0) that rewrites the cache slot.

| official eager | static runtime |
|---|---|
| before every forward: 4 `.item()` reads of cache indices (wrapper `_call_model`), more in `_pin_current_chunk`, UniPC `index_for_timestep`, `linalg.solve` | index bookkeeping replayed on the host with Python ints (`StaticLL._insert`, incl. sink / pinned shot-sink / floating-pin shift / `shot_clean_recache`); verified equal to the eager cache indices at every chunk |
| every forward, every layer: `kv_cache["k"].clone()` + `["v"].clone()` (2 x 173 MB), roll/insert into the clone, attend over clone[:, :L], then `_apply_cache_updates` repeats roll/insert on the real cache | per-layer buffers with the identical layout; roll applied once per chunk (in-place left shift, chunked to avoid overlap); each forward writes K/V straight into slot [L-7040, L) and attends over buf[:, :L] (contiguous, same values, same FA2 call) |
| `flash_attention` wrapper builds cu_seqlens on the CPU + H2D copy (2 per attention, 300 attentions/chunk) | `flash_attn_varlen_func` with precomputed device cu_seqlens |
| every forward recomputes `text_embedding` (512x4096->3072) and 30 x (k, v) cross-attention projections (crossattn cache disabled upstream for cudagraph experiments) | cross K/V computed once per prompt change into static buffers (same ops) |
| timestep sinusoid: `torch.arange` on CPU + H2D each forward | precomputed per step |
| UniPC step: `torch.tensor([...], device=cuda)` per step (H2D), CUDA `linalg.solve` (sync) | device constants computed once per step index by the same code (`StaticUniPC`), rest of the step identical |
| wrapper `_convert_flow_pred_to_x0` in fp64 (result discarded by the pipeline) | not executed |
| RoPE tables: LRU-cached complex128->fp32 cos/sin keyed by start frame / offset | computed per chunk with the same functions into static buffers |

Since `local_attn_size * 880 == max_attention_size == cache size`, the attended window is always buf[:, :local_end]
(the prepend-sink / prepend-pinned branches are unreachable), so only 4 attention shapes exist
(L = 7040, 14080, 21120, 28160). One CUDA graph per L captures a whole chunk (4 forwards + 4 UniPC steps + context
forward), recorded with `fastworld.graph.capture` (shared pool; the warm-up/capture scribble on the KV buffers is
zeroed afterwards so a fresh session starts from zeros). Scene cuts only change static inputs and the host-side roll.
With `shot_clean_recache` the cut chunk's context forward runs at a different L than its denoising forwards; that chunk
runs through the same static code without a graph (bitwise as well).

Validation layers (all in `scripts/longlive2/test_static.py`, same process, same GPU): (1) prepared inputs: RoPE tables
vs `causal_rope_apply` (incl. offsets 8, 16), host index plan vs the eager cache indices at each chunk start;
(2) single forward chunk 0 step 0; (3) 12-chunk rollout without graph, plus every one of the 48 denoising flow
predictions vs the eager ones; (4) with graphs (2 rollouts); (5) state save after chunk 6, a different scene-cut prompt
+ different noise at chunk 7 and an extra chunk, restore, continue the original trajectory.

## Blockers / caveats

* **Compute-bound.** Eager is already 97.5 % GPU-busy on H100 (99-100 % on L40S). Launch overhead / host syncs cost
  only ~2.5 % of wall time; the rest of the 4 % gain is GPU work the official code does but does not need (whole-cache
  clones every forward, cross-attention K/V + text_embedding recomputed every forward, fp64 x0 conversion). The
  remaining time is FA2 attention (~57 %) and GEMMs (~23 %); graphs cannot touch that. Exactly-equal outputs forbid the
  real levers (FA3 on Hopper, FP8/NVFP4, fused norms), which all change numerics.
* **torch.compile (official `max-autotune-no-cudagraphs`, dynamic=False)** is not a usable streaming baseline:
  `_CURRENT_GRID_META` publishes the cache indices as Python ints, so every new chunk position (global_end_index etc.)
  is a new guard -> recompile (~2-4 min each with max-autotune); after 8 recompiles dynamo hits `recompile_limit` and
  silently runs the rest eagerly. On L40S the first 12-chunk run took 831 s, steady chunks are no faster than eager, and
  outputs are not bitwise (max abs latent diff 6-10). The release README itself says it is validated only for a single
  8-frame block. The H100 compile run (`compile_baseline.py` -> `results/longlive2/compile_h100.json`) is pending;
  numbers above are from L40S.
* `inference.streaming_vae: true` is broken in this release (`WanVAE_` has no `cached_decode`), so only the default
  batch decode at the end was timed (`vae_time.py`).
* `_adaln_modulate_kernel` is `@triton.autotune`d; the chosen config is per process, so bitwise equality is guaranteed
  for eager vs static in the same process (as tested). Across processes eager itself could in principle differ if the
  autotuner picks a different num_warps.
* `shot_clean_recache: true`: the cut chunk's context forward attends a different window (14080) than its denoising
  forwards; that chunk runs through the static code without a graph (bitwise). Default config (false) is fully graphed.
* Scope: T2V, guidance 1.0, BF16, batch 1, `use_relative_rope=False`, no KV quantization (asserted). I2V first-frame
  clamp, CFG (neg cache), NVFP4 KV are not ported.

## Verdict

Exactness: done. The static runtime reproduces the official eager pipeline bitwise on H100 and L40S for 12 chunks
with two scene cuts (also with `shot_clean_recache`), every intermediate flow prediction matches, graph replay is
bitwise, and save/branch/restore/continue is bitwise. Speed: honest answer is "little to gain": **1.041x** DiT time per
steady chunk on H100 (1757 -> 1688 ms; with 32 output frames per chunk, 18.2 -> 19.0 FPS DiT-only), of which ~1.038x comes from
the static rewrite alone (no redundant cache clones / recomputation) and only ~0.3 % from CUDA graphs, because eager is
97.5 % GPU-busy. This is the expected outcome for a 5B, 720p, 7040x28160-attention model and the opposite regime of
Matrix-Game-2 (62 % busy, 1.60x). The static runtime is still useful as an exact, branchable, sync-free session engine
(7.8 GB live state, 9 ms save / 7 ms restore, no host syncs per chunk), but not as a speedup.
