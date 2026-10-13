#!/bin/bash
# Slurm wrapper for the LingBot-World-v2 scripts. Pass partition/account/QOS on the command line, e.g.
#   sbatch -A <acct> -p <gpu-partition> --gres=gpu:1 -c14 --time=01:00:00 scripts/lingbot_v2/job.sh scripts/lingbot_v2/test_static.py --chunks 12
# FASTWORLD_PY selects the interpreter (default: python on PATH); model locations via FASTWORLD_* (see fastworld/paths.py).
# Optional: scripts/lingbot_v2/env.sh (not committed) is sourced if present, e.g. for HF_HOME / TRITON_CACHE_DIR /
# TORCHINDUCTOR_CACHE_DIR (the latter two matter for compile_baseline.py only).
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH -o logs/%x_%j.out
# sbatch runs a copy of this file from its spool dir, so $0 is useless there: use the submit dir (submit from the repo root)
ROOT=${FASTWORLD_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}}
cd "$ROOT"
mkdir -p logs
[ -f "$ROOT/scripts/lingbot_v2/env.sh" ] && source "$ROOT/scripts/lingbot_v2/env.sh"
export PYTHONPATH="$ROOT:$PYTHONPATH"
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export PYTHONUNBUFFERED=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
exec "${FASTWORLD_PY:-python}" "$@"
