#!/bin/bash
# Slurm wrapper for the MG2 scripts. Pass partition/account/QOS on the command line, e.g.
#   sbatch -A <acct> -p <gpu-partition> --gres=gpu:1 -c14 --time=00:30:00 scripts/mg2/job.sh scripts/mg2/test_static.py
# FASTWORLD_PY selects the interpreter (default: python on PATH); model locations via FASTWORLD_* (see fastworld/paths.py).
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --requeue
#SBATCH -o logs/%x_%j.out
#SBATCH --open-mode=append
# sbatch runs a copy of this file from its spool dir, so $0 is useless there: use the submit dir (submit from the repo root)
ROOT=${FASTWORLD_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}}
cd "$ROOT"
mkdir -p logs
export PYTHONPATH="$ROOT:$PYTHONPATH"
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
exec "${FASTWORLD_PY:-python}" "$@"
