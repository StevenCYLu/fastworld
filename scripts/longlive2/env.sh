# Cache / runtime environment for the LongLive-2.0 scripts (sourced by scripts/longlive2/job.sh; `source` it yourself for
# interactive runs). Every value is a default: export the variable before sbatch to override (sbatch forwards the env).
# Keep the Triton/Inductor caches off a small home quota: default under the repo's gitignored .inductor_cache/.
ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$ROOT/.inductor_cache/triton}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$ROOT/.inductor_cache/longlive2}   # only compile_baseline.py uses it
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}        # everything is loaded from local files (fastworld/paths.py)
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
