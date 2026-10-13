#!/bin/bash
# Slurm wrapper for the LongLive-2.0 scripts. Pass partition/account/QOS on the command line, e.g.
#   FASTWORLD_PY=/path/to/venv/bin/python sbatch -A <acct> -p <gpu-partition> --gres=gpu:1 -c14 --time=01:00:00 \
#       scripts/longlive2/job.sh scripts/longlive2/test_static.py --chunks 12
# FASTWORLD_PY selects the interpreter (default: python on PATH); model locations via FASTWORLD_* (see fastworld/paths.py);
# caches via scripts/longlive2/env.sh (sourced if present). The official loaders need no particular cwd: model.py
# resolves their `wan_models/Wan2.2-TI2V-5B` lookups to WAN22_DIR itself.
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH -o logs/%x_%j.out
# sbatch runs a copy of this file from its spool dir, so $0 is useless there: use the submit dir (submit from the repo root)
ROOT=${FASTWORLD_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}}
cd "$ROOT"
mkdir -p logs
export PYTHONPATH="$ROOT:$PYTHONPATH"
[ -f "$ROOT/scripts/longlive2/env.sh" ] && source "$ROOT/scripts/longlive2/env.sh"
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
exec "${FASTWORLD_PY:-python}" "$@"
