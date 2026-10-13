#!/bin/bash
# Slurm wrapper for the MG3 scripts. Pass partition/account/QOS on the command line, e.g.
#   sbatch -A <acct> -p <gpu-partition> --gres=gpu:1 -c14 --time=00:30:00 scripts/mg3/job.sh scripts/mg3/test_static.py --iters 8
# FASTWORLD_PY selects the interpreter (default: python on PATH); model locations via FASTWORLD_* (see fastworld/paths.py).
# The official Matrix-Game-3 code loads `assets/...` relative to its own directory, so the script runs from MG3_DIR;
# the script path is resolved against the repo root first. Optional scripts/mg3/env.sh (cache dirs etc.) is sourced.
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --requeue
#SBATCH -o logs/%x_%j.out
# sbatch runs a copy of this file from its spool dir, so $0 is useless there: use the submit dir (submit from the repo root)
ROOT=${FASTWORLD_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}}
cd "$ROOT"
mkdir -p logs
export PYTHONPATH="$ROOT:$PYTHONPATH"
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
[ -f "$ROOT/scripts/mg3/env.sh" ] && source "$ROOT/scripts/mg3/env.sh"
PY="${FASTWORLD_PY:-python}"
SCRIPT="$1"; shift
[[ "$SCRIPT" = /* ]] || SCRIPT="$ROOT/$SCRIPT"
MG3_DIR=$("$PY" -c "from fastworld.paths import MG3_DIR; print(MG3_DIR)")
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "start $(date)  cwd $MG3_DIR"
cd "$MG3_DIR"
exec "$PY" -u "$SCRIPT" "$@"
