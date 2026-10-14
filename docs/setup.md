# Setup

fastworld imports the official model code unmodified and loads the official weights. Nothing is vendored: the model
repo is cloned into `third_party/` and the weights are downloaded next to it (or anywhere, with the `FASTWORLD_*`
environment variables from `fastworld/paths.py` pointing at them).

## Jetson AGX Thor (the main target)

Tested on a Thor developer kit: L4T R38.4 (JetPack 7.x), CUDA 13.0, Python 3.12, torch 2.14.1+cu130, flash-attn 2.8.3
(built from source), TensorRT 10.13.3 (from JetPack).

```sh
git clone https://github.com/StevenCYLu/fastworld && cd fastworld
scripts/jetson/setup.sh               # venv + torch cu130 + deps, Matrix-Game-2 code, 11.8 GB of weights, TAEHV, TensorRT link
scripts/jetson/build_flash_attn.sh    # FlashAttention 2 for sm_110 (~2 h)
sudo nvpmodel -m 0 && sudo jetson_clocks
```

Check (each prints its result; JSON goes to `results/mg2/`):

```sh
export PYTHONPATH=$PWD HF_HUB_OFFLINE=1
.venv/bin/python scripts/mg2/test_static.py --chunks 10 --out results/mg2/thor/static.json   # exact runtime: all True
.venv/bin/python -m fastworld.models.mg2.trt_vae                                             # TensorRT VAE engine, ~15 min
.venv/bin/python scripts/mg2/thor_vae.py                                                     # decoders: speed + PSNR
.venv/bin/python scripts/mg2/thor_e2e.py                                                     # end-to-end FPS + side-by-side mp4
```

The first `FastMG(compile=True)` run compiles the DiT with Inductor (11-13 min on Thor); later runs hit the on-disk
cache. Disk: ~12 GB of weights, ~1 GB for the flash-attn build tree, ~150 MB each for the VAE ONNX export and its TensorRT engine.

Expected layout afterwards (what `fastworld.models.mg2.Model` reads):

```
third_party/Matrix-Game-2/
  configs/inference_yaml/inference_universal.yaml
  demo_images/universal/0000.png ...
  ckpt/base_distilled_model/base_distill.safetensors
  ckpt/Wan2.1_VAE.pth
  ckpt/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth, xlm-roberta-large/
third_party/taehv/taew2_1.pth
build/trt/vae_dec.onnx, vae_dec_fp16.plan        (after building the TensorRT VAE)
```

Already have the official repo and weights elsewhere? Point at them instead of cloning:

```sh
export FASTWORLD_MG2_DIR=/path/to/Matrix-Game/Matrix-Game-2      # ckpt/ inside it, or FASTWORLD_MG2_CKPT separately
export FASTWORLD_TAEHV_DIR=/path/to/taehv
export FASTWORLD_TRT_DIR=/path/to/engines                        # where TRTVAE reads/writes its engine
```

## Datacenter GPUs

### Matrix-Game-2

Environment: Python 3.10+, torch >= 2.4 with CUDA, FlashAttention 2. Tested with torch 2.7.1+cu126, flash-attn 2.8.3,
diffusers 0.31.0 on H100 and L40S.

```sh
git clone https://github.com/StevenCYLu/fastworld && cd fastworld
python -m venv .venv && . .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -e ".[mg2]"              # diffusers, omegaconf, safetensors, lpips, imageio
pip install flash-attn --no-build-isolation   # or a prebuilt wheel matching your torch

git clone --depth 1 https://github.com/SkyworkAI/Matrix-Game third_party/Matrix-Game-src
ln -s Matrix-Game-src/Matrix-Game-2 third_party/Matrix-Game-2
pip install -r third_party/Matrix-Game-2/requirements.txt
huggingface-cli download Skywork/Matrix-Game-2.0 --local-dir third_party/Matrix-Game-2/ckpt     # ~10 GB
```

Expected layout afterwards (what `fastworld.models.mg2.Model` reads):

```
third_party/Matrix-Game-2/
  configs/inference_yaml/inference_universal.yaml
  demo_images/universal/0000.png ...
  ckpt/base_distilled_model/base_distill.safetensors
  ckpt/Wan2.1_VAE.pth
  ckpt/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth, xlm-roberta-large/
```

Optional lossy decoder (`fastworld.models.mg2.TinyVAE`):

```sh
git clone --depth 1 https://github.com/madebyollin/taehv third_party/taehv      # ships taew2_1.pth
```

Check:

```sh
python scripts/mg2/test_static.py --chunks 10        # eager vs static vs graphs vs branch: all lines must say True
python scripts/mg2/example.py                        # writes results/mg2/example_*.mp4
```

Already have the official repo and weights elsewhere? Point at them instead of cloning:

```sh
export FASTWORLD_MG2_DIR=/path/to/Matrix-Game/Matrix-Game-2      # ckpt/ inside it, or FASTWORLD_MG2_CKPT separately
export FASTWORLD_TAEHV_DIR=/path/to/taehv
```

### Other models

See `docs/models/lingbot_v2.md`, `docs/models/mg3.md`, `docs/models/longlive2.md` for their environments (they differ:
LongLive needs torch 2.8, Matrix-Game-3 torch 2.10 + FlashAttention 3), weights and the exact commands used for the
numbers in the README.

### Running on a cluster

`scripts/<model>/job.sh` is a Slurm wrapper: it sets `PYTHONPATH` to the repo, `HF_HUB_OFFLINE=1`, and runs the
script with `$FASTWORLD_PY` (default `python`). Partition/account/QOS go on the command line:

```sh
FASTWORLD_PY=.venv/bin/python sbatch -A <account> -p <partition> --gres=gpu:1 -c14 --time=00:30:00 \
    scripts/mg2/job.sh scripts/mg2/test_static.py --chunks 10
```

All scripts write their JSON incrementally or are idempotent (`run_bench.py` skips finished entries), so they survive
preemption and requeueing.
