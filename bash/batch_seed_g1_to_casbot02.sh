#!/usr/bin/env bash
# Retarget every Bones Seed G1 CSV onto casbot02.
# Output filenames are unchanged, under output_data/robot_motion/seed/.
set -uo pipefail

PYBIN="${PYBIN:-$HOME/miniconda3/envs/robot_retargeter/bin/python}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

exec "${PYBIN}" scripts/batch_seed_to_casbot02.py "$@"
