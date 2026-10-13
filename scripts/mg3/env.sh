# Optional environment for the MG3 scripts; sourced by scripts/mg3/job.sh (ROOT = repo root). Edit to taste.
# Everything here has a default under the repo, so nothing needs to be set on a fresh checkout.
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$ROOT/.inductor_cache/mg3}"   # compile_baseline.py only
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$ROOT/.inductor_cache/mg3/triton}"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
# T5 / VAE / DiT weights are read from MG3_CKPT (fastworld/paths.py), so no HF_HOME is required at run time.
