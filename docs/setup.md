# Setup

fastworld imports the official model code unmodified and loads the official weights. Nothing is vendored: you clone
the model repo into `third_party/` and download weights into `weights/` (or anywhere, with the `FASTWORLD_*`
environment variables from `fastworld/paths.py` pointing at them). Each model has its own environment requirements,
so use one virtual environment per model.

## Matrix-Game-2 (the main target)

Environment: Python 3.10+, torch >= 2.4 with CUDA, FlashAttention 2. Tested with torch 2.7.1+cu126, flash-attn 2.8.3,
diffusers 0.31.0 on H100 and L40S.

```sh
git clone https://github.com/StevenCYLu/fastworld && cd fastworld
python -m venv .venv && . .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -e ".[mg2]"              # einops, diffusers, omegaconf, safetensors, lpips (+ flash-attn: prebuilt wheel recommended)

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

## Other models

See `docs/models/lingbot_v2.md`, `docs/models/mg3.md`, `docs/models/longlive2.md` for their environments (they differ:
LongLive needs torch 2.8, Matrix-Game-3 torch 2.10 + FlashAttention 3), weights and the exact commands used for the
numbers in the README.

## Running on a cluster

`scripts/<model>/job.sh` is a Slurm wrapper: it sets `PYTHONPATH` to the repo, `HF_HUB_OFFLINE=1`, and runs the
script with `$FASTWORLD_PY` (default `python`). Partition/account/QOS go on the command line:

```sh
FASTWORLD_PY=.venv/bin/python sbatch -A <account> -p <partition> --gres=gpu:1 -c14 --time=00:30:00 \
    scripts/mg2/job.sh scripts/mg2/test_static.py --chunks 10
```

All scripts write their JSON incrementally or are idempotent (`run_bench.py` skips finished entries), so they survive
preemption and requeueing.
