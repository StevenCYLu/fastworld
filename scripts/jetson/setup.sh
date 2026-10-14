#!/bin/bash
# One-shot environment for Matrix-Game-2 on Jetson AGX Thor (JetPack 7.x / L4T R38, CUDA 13, aarch64, Python 3.12).
# Run from the repo root. Steps (each skipped if already done):
#   1. .venv with torch/torchvision from the cu130 index + fastworld[mg2,jetson]
#   2. the official Matrix-Game code (third_party/Matrix-Game-2) and the part of its requirements.txt the inference path
#      imports (its TensorRT/pycuda/onnx/wandb/flask entries are for other tools and are skipped)
#   3. only the weights MG2 needs (~11.8 GB; the full HF repo is 21.5 GB) and the TAEHV decoder
#   4. JetPack's TensorRT Python bindings linked into the venv (they only ship for the system Python)
# FlashAttention is built separately (no sm_110 wheel, ~2 h): scripts/jetson/build_flash_attn.sh
# Then: sudo nvpmodel -m 0 && sudo jetson_clocks   (the README numbers were measured in MAXN with the GPU at its max clock)
set -euo pipefail
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
PY=.venv/bin/python

[ -x "$PY" ] || uv venv .venv --python 3.12
uv pip install --python $PY torch torchvision --index-url https://download.pytorch.org/whl/cu130
uv pip install --python $PY -e ".[mg2,jetson]" pip
uv pip install --python $PY opencv-python "transformers>=4.49.0" tokenizers accelerate tqdm easydict ftfy av \
    open_clip_torch sentencepiece scikit-image "git+https://github.com/openai/CLIP.git"

mkdir -p third_party
[ -d third_party/Matrix-Game-src ] || git clone --depth 1 https://github.com/SkyworkAI/Matrix-Game third_party/Matrix-Game-src
[ -e third_party/Matrix-Game-2 ] || ln -s Matrix-Game-src/Matrix-Game-2 third_party/Matrix-Game-2
[ -d third_party/taehv ] || git clone --depth 1 https://github.com/madebyollin/taehv third_party/taehv
.venv/bin/hf download Skywork/Matrix-Game-2.0 --local-dir third_party/Matrix-Game-2/ckpt \
    --include "base_distilled_model/*" "Wan2.1_VAE.pth" "models_clip_*" "xlm-roberta-large/*" "*.json"

SP=$($PY -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
SYS=/usr/lib/python3.12/dist-packages
if [ -d $SYS/tensorrt ]; then
    for d in $SYS/tensorrt $SYS/tensorrt-*.dist-info $SYS/tensorrt_lean $SYS/tensorrt_dispatch; do
        [ -e "$d" ] && ln -sfn "$d" "$SP/$(basename "$d")"
    done
else
    echo "TensorRT Python bindings not found in $SYS (apt package python3-libnvinfer); TRTVAE will be unavailable"
fi
$PY -c "import torch; print('torch', torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
$PY -c "import tensorrt; print('tensorrt', tensorrt.__version__)" || true
$PY -c "import flash_attn" 2>/dev/null && echo "flash_attn ok" || echo "next: scripts/jetson/build_flash_attn.sh"
