#!/usr/bin/env bash
# 批量将 dataset/lafan1_g1 下全部 CSV 重定向到 CASBOT02
# 输出: output_data/robot_motion/<name>_from_g1_CASBOT02.csv
set -uo pipefail

PYBIN="${PYBIN:-$HOME/miniconda3/envs/robot_retargeter/bin/python}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

SRC_DIR="dataset/lafan1_g1"
FAILED=()

shopt -s nullglob
FILES=("${SRC_DIR}"/*.csv)
echo "[info] total files: ${#FILES[@]}"

for f in "${FILES[@]}"; do
  name="$(basename "${f}" .csv)"
  kp_name="${name}_from_g1"
  out_csv="output_data/robot_motion/${kp_name}_CASBOT02.csv"

  echo "=== ${name} ==="
  if [[ -f "${out_csv}" ]]; then
    echo "[skip] already exists: ${out_csv}"
    continue
  fi

  if ! "${PYBIN}" scripts/robot_replay.py --no-viewer \
      --motion-file "${f}" \
      --source-robot-config config/robot/g1.yaml \
      --target-robot-config config/robot/CASBOT02.yaml \
      --fps 30 >/dev/null 2>&1; then
    echo "[fail] robot_replay: ${name}"
    FAILED+=("${name} (replay)")
    continue
  fi

  if ! "${PYBIN}" scripts/robot_retarget.py \
      --config config/robot/CASBOT02.yaml \
      --keypoints-name "${kp_name}" \
      --no-render-debug >/dev/null 2>&1; then
    echo "[fail] robot_retarget: ${name}"
    FAILED+=("${name} (retarget)")
    continue
  fi
  echo "[ok] ${out_csv}"
done

echo
echo "[summary] failed: ${#FAILED[@]}"
for x in "${FAILED[@]:-}"; do [[ -n "${x}" ]] && echo "  - ${x}"; done
