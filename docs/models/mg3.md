# Matrix-Game-3.0: exact static-shape CUDA-graph runtime

Target model: Matrix-Game-3.0 distilled 5B (Wan2.2-TI2V-5B base, 30 layers, dim 3072). Settings are the official README
defaults: 704x1280, 3 denoising steps, INT8 q/k/v/o (`--use_int8`), FlashAttention-3, MG-LightVAE v2, single GPU,
no Ulysses/async VAE.

## Setup
Locations come from `fastworld/paths.py`: `MG3_DIR` (official code, default `third_party/Matrix-Game-3`) and `MG3_CKPT`
(HF snapshot, default `weights/Matrix-Game-3.0`); override with `FASTWORLD_MG3_DIR` / `FASTWORLD_MG3_CKPT`.
```
# official code (unmodified; the Matrix-Game-3 subdir of the SkyworkAI/Matrix-Game repo)
git clone https://github.com/SkyworkAI/Matrix-Game third_party/Matrix-Game && ln -s Matrix-Game/Matrix-Game-3 third_party/Matrix-Game-3
# environment: the official stack (torch 2.10 + cu128 + a prebuilt FlashAttention-3 wheel), Python 3.12
uv venv --python 3.12 .venv-mg3
uv pip install --python .venv-mg3/bin/python torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv-mg3/bin/python \
  "https://github.com/windreamer/flash-attention3-wheels/releases/download/2026.03.19-850211f/flash_attn_3-3.0.0%2B20260318.cu128torch2100cxx11abitrue.8afc61-cp39-abi3-linux_x86_64.whl" \
  opencv-python==4.10.0.84 "diffusers>=0.36.0" transformers==4.57.3 accelerate tqdm "imageio[ffmpeg]" easydict ftfy \
  numpy==2.2.6 scipy einops pandas safetensors==0.7.0 huggingface_hub trimesh regex requests
# weights (~41 GB): base_distilled_model/, Wan2.2_VAE.pth (encoder), MG-LightVAE_v2.pth, T5 + tokenizer
.venv-mg3/bin/python -c "from huggingface_hub import snapshot_download as s; s('Skywork/Matrix-Game-3.0', local_dir='weights/Matrix-Game-3.0', allow_patterns=['base_distilled_model/*','Wan2.2_VAE.pth','MG-LightVAE_v2.pth','models_t5_umt5-xxl-enc-bf16.pth','google/*'])"
```
(If the snapshot already lives in an HF cache, symlink `weights/Matrix-Game-3.0` to the snapshot directory instead.)

Runs go through `scripts/mg3/job.sh`, which puts the repo root on `PYTHONPATH`, sources the optional
`scripts/mg3/env.sh` (cache dirs) and `cd`s into `MG3_DIR` so the official code's relative `assets/` paths resolve.
`FASTWORLD_PY` selects the interpreter; relative `--out` paths are taken relative to the repo root.
```
export FASTWORLD_PY=$PWD/.venv-mg3/bin/python
sbatch -A <acct> -p <debug-gpu> --gres=gpu:1 -c14 --time=01:00:00 scripts/mg3/job.sh scripts/mg3/check_official.py --iters 4 --out results/mg3/check_official_l40.json
sbatch -A <acct> -p <h100>      --gres=gpu:1 -c14 --time=00:25:00 scripts/mg3/job.sh scripts/mg3/test_static.py --iters 8 --out results/mg3/static_h100.json
sbatch -A <acct> -p <h100>      --gres=gpu:1 -c14 --time=00:25:00 scripts/mg3/job.sh scripts/mg3/compile_baseline.py --iters 8 --out results/mg3/compile_h100.json
```
FA3: the official code really uses FA3 (`flash_attn_interface`, `--fa_version 3`); a prebuilt FA3 wheel for torch 2.10/cu128
exists (windreamer index), so the official stack is used as-is. This FA3 build also runs on L40S (sm89 path).

## Files
- `fastworld/models/mg3/engine.py` - the official `generate()` loop body transcribed into prep / denoise / commit / decode
  (`Engine`). `scripts/mg3/check_official.py` verifies it is **bitwise identical to the official
  `MatrixGame3Pipeline.generate()`** (4/4 iterations) and eager-vs-eager deterministic.
- `fastworld/models/mg3/static.py` - static runtime + CUDA graphs (`StaticMG3`, `CachedUniPC`); capture uses
  `fastworld.graph.capture`. `scripts/mg3/test_static.py` - exactness, profile, ablation, speed, branch test.
- `scripts/mg3/compile_baseline.py` - `torch.compile(model)` defaults. `results/mg3/*.json` raw numbers.

## How MG3 runs (and why it is easy to make static)
No KV cache: each iteration runs the DiT on `[5 retrieved memory latents | 4 past latents | 10 new latents]` = 19 latent
frames = 16,720 tokens, and the official code already pads every forward to `max_seq_len` = 17,600. The memory count is
**always 5** (5 FOV queries with `selected_index[-1] = 4` forced; duplicates allowed), so no padding/bucketing of memory
was needed: from iteration 1 on all shapes are constant; iteration 0 (15 latents, no memory) is the second shape.
Two graphs: `first` and `steady`, each = one whole iteration (RoPE tables + 3 DiT forwards + 3 UniPC steps + img_cond
re-injection). Retrieval (`select_memory_idx_fov`, GPU, 5 `.item()`), plucker embedding and the noise draw stay eager
host-side prep in the same RNG order (prep = ~23 ms/iter on L40S, retrieval itself ~6.5 ms).
The action trajectory is the official random one (`Bench_actions_universal`, seed 42): keyboard/mouse change every
12 frames; retrieved memory indices vary per iteration (e.g. `[251, 102, 102, 250, 4]` at iteration 7).

## What had to change (all value-preserving; verified bitwise)
1. `flash_attention`: cu_seqlens built on CPU + H2D every call -> cached device tensors.
2. `rope_apply_with_indices`: per call it H2D-copies the index list and rebuilds a [12320,24,64] complex128 table
   (4x per layer x 30 layers x 3 steps = 360 builds/iter) -> 2 tables/iter built inside the graph from static device
   index buffers (same ops -> same complex64 values). **This is the biggest win** (memory traffic, not launches).
3. `time_embedding`/`time_projection` (fp32 GEMMs 17600x3072 -> 17600x18432 under fp32 autocast): the per-token
   timestep vector is identical every steady iteration for a given step -> output cached per step (4.5 GB).
4. `sinusoidal_embedding_1d` CPU arange+H2D -> cached device inverse frequencies; `ActionModule.get_rotary_pos_embed`
   (constant, ~50 launches/call) -> cached.
5. Pipeline `timestep = latents.new_full(.., t)` (`.item()` sync) -> precomputed per-step tensors.
6. UniPC `step()`: `torch.tensor(list, device=cuda)` H2D copies + a 2x2 `linalg.solve` (sync) on tiny coefficient
   tensors -> computed once by the identical code and reused (`CachedUniPC`); big-tensor math textually unchanged.

## Results
### H100 80GB (final) — see `results/mg3/static_h100.json`, `results/mg3/compile_h100.json`
| H100, 704x1280, INT8+FA3, 8 iterations | eager (official) | ours: static, no graph | ours: static + CUDA graph | torch.compile default |
|---|---|---|---|---|
| DiT ms / steady iteration (median, warm) | 2800 | 2531 (1.106x) | **2509 (1.116x)** | 2103 (1.34x vs 2822 eager in that job) |
| iteration 0 (no memory) ms | 2570 | 2474 | 2459 | 1908 |
| GPU busy (CUDA kernel time / wall) | **98.2%** (2749 / 2800 ms) | - | 100% (2523 ms kernels) | - |
| kernel launches / host syncs per iteration | 26,680 / 384 (+2,259 memcpy) | - | 1 graph launch | 45 graph breaks |
| bitwise vs eager | eager==eager 8/8 | 8/8 (+ single fwd) | **8/8**, branch+restore 4/4 | 0/8 (cos 0.999 at it 0 -> 0.78 at it 7) |
| capture / compile time | - | - | 11.1 s (2 graphs) | 103 s (+ recompiles at it 0,1) |
| end-to-end incl. LightVAE-v2 decode (0.29 s) | 3.107 s/iter (12.9 fps) | - | 2.821 s/iter (14.2 fps, 1.10x) | - |

Ablation on one steady iteration (H100): eager 2.800 s -> cu_seqlens / RoPE-table / rotary / scheduler hoists 2.667 s
(-4.7%) -> + time-embedding cache 2.532 s (-5.1%) -> + CUDA graph 2.509 s (-0.9%).
Host prep per iteration 13.5 ms (memory retrieval 3.1 ms of it) - unchanged, outside the graph.
Peak GPU memory 22.8 GB allocated (weights + 4.5 GB temb cache + 0.39 GB static inputs + graph pool).

### L40S (debug GPU, 8 iterations) — `results/mg3/static_l40_8iter.json`, `results/mg3/compile_l40_8iter.json`
| | eager | ours: static, no graph | ours: static + CUDA graph | torch.compile default |
|---|---|---|---|---|
| DiT ms / steady iteration (median) | 8853 | 8181 | **8139 (1.088x)** | 6569 (1.34x) |
| iteration 0 (no memory) ms | 8241 | - | 7810 | - |
| GPU busy (kernel/wall) | 99.3% | - | 99.9% | - |
| kernel launches / syncs per iter | 26,680 / 384 | - | 1 graph launch | - |
| bitwise vs eager (8 iters) | eager==eager 8/8 | 8/8 | **8/8** (+ branch/restore 4/4) | 0/8 (cos 0.998 at it 0 -> 0.78 at it 7) |
| capture / compile time | - | - | 33 s | 46-100 s (45 graph breaks) |

Ablation on one steady iteration (L40S): eager 8.82 s -> + cu_seqlens/RoPE-table/rotary/scheduler hoists 8.39 s
(-4.8%) -> + time-embedding cache 8.19 s (-2.4%) -> + CUDA graph 8.14 s (-0.5%).
End-to-end incl. MG-LightVAE-v2 decode (40 frames, 0.77 s): eager 9.65 s/iter, ours 8.95 s/iter.

Session state (per session, both runtimes identical by construction): latent history 3.4 MB/iteration (+1.35 MB img_cond;
30 MB after 8 iterations) + VAE stream cache 0.80 GB. Ours adds shared (not per-session) static input buffers 0.39 GB,
the time-embedding cache 4.5 GB and the graph pool (peak allocated 22.8 GB total on L40S incl. weights).
State save/load (clone latents + generator state) is <1 ms; branch-from-saved-state is bitwise.

Other result files: `results/mg3/check_official_l40.json` (engine vs official `generate()`), `static_l40.json` /
`compile_l40.json` (earlier 4-iteration runs), `static_l40_v1_noTembCache.json` (before the time-embedding cache).

## Verdict
MG3 at 720p is **compute/memory-bandwidth bound, not launch bound**: eager is already 98.2% GPU-busy on H100 (99.3% on
L40S) despite 26.7k launches and 384 syncs per iteration, because each iteration is 3 forwards over 17,600 tokens.
The exact static runtime is bitwise identical to the official pipeline on H100 (8/8 iterations with varying actions and
FOV memory retrieval, plus branch/restore) and gives **1.116x on DiT (1.10x end-to-end)**, but almost all of that comes
from removing redundant *work* that the static rewrite exposes (360 RoPE-table rebuilds and 3 constant fp32
time-embedding GEMM chains per iteration); the CUDA graph itself adds only ~0.9%, matching the ~2% idle time.
Non-exact `torch.compile` is clearly faster (1.34x) because the remaining cost is unfused fp32 elementwise/dtype work,
but its rollouts diverge from eager (cos 0.78 by iteration 8). If exactness is required, the static runtime is the
better option; if not, compile (possibly on top of the static runtime) is where the remaining speed is.

## Open issues
- Remaining eager inefficiencies are memory-bound fp32 elementwise work (per-block `modulation + e` over a 17600x18432
  fp32 tensor, dtype conversions, fp32 autocast). Inductor fuses these (hence 1.34x) but changes numerics; an exact
  version would need hand rewrites (e.g. compute modulation on the 2 distinct timestep rows and broadcast).
- torch.compile default hits 45 graph breaks (`.item()` on grid sizes, data-dependent `repeat(S)`, list-index RoPE)
  and needed one environment fix (register the dynamically loaded `triton_kernels` module in `sys.modules`, otherwise
  dynamo crashes on Int8Linear and falls back to eager).
- Importing the official `wan` package calls `torch.cuda.current_device()` at class-definition time, so
  `fastworld.models.mg3` can only be imported on a machine with a GPU.
