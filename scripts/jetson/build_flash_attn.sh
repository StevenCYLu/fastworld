#!/bin/bash
# Build FlashAttention 2 from source for Jetson AGX Thor (sm_110, aarch64, CUDA 13). There is no prebuilt wheel.
# Two patches to v2.8.3's setup.py: add the sm_110 gencode, and compile with C++20 (the torch 2.14 headers require it).
# Took ~2 h on a Thor dev kit with MAX_JOBS=8. Usage, from the repo root after scripts/jetson/setup.sh:
#   scripts/jetson/build_flash_attn.sh [path/to/python]
set -euo pipefail
PY=${1:-.venv/bin/python}
PY=$(realpath "$PY")
SRC=${FLASH_ATTN_SRC:-build/flash-attention}
VER=${FLASH_ATTN_VERSION:-v2.8.3}

if [ ! -d "$SRC" ]; then
    git clone --depth 1 --branch "$VER" https://github.com/Dao-AILab/flash-attention "$SRC"
    git -C "$SRC" submodule update --init --depth 1 csrc/cutlass
fi
cd "$SRC"
if ! grep -q "compute_110" setup.py; then
    python3 - <<'EOF'
s = open("setup.py").read()
anchor = '            cc_flag.append("arch=compute_120,code=sm_120")\n'
assert anchor in s, "setup.py layout changed; add the sm_110 gencode by hand"
s = s.replace(anchor, anchor + '''        if bare_metal_version >= Version("13.0") and "110" in cuda_archs():
            cc_flag.append("-gencode")
            cc_flag.append("arch=compute_110,code=sm_110")
''', 1)
open("setup.py", "w").write(s)
EOF
fi
sed -i 's/-std=c++17/-std=c++20/g' setup.py

export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH="$CUDA_HOME/bin:$PATH"
export FLASH_ATTN_CUDA_ARCHS=110 FLASH_ATTENTION_FORCE_BUILD=TRUE
export MAX_JOBS=${MAX_JOBS:-8} NVCC_THREADS=${NVCC_THREADS:-2}
"$PY" -m pip install --no-build-isolation -v . 2>&1 | tail -5
"$PY" - <<'EOF'
import torch, flash_attn
from flash_attn import flash_attn_func
q = torch.randn(1, 1024, 12, 128, device="cuda", dtype=torch.bfloat16)
o = flash_attn_func(q, q, q)
r = torch.nn.functional.scaled_dot_product_attention(*(t.transpose(1, 2) for t in (q, q, q))).transpose(1, 2)
print("flash_attn", flash_attn.__version__, "max diff vs SDPA", (o - r).abs().max().item())
EOF
