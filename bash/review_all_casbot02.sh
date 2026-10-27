#!/usr/bin/env bash
# 逐个播放全部 lafan1_g1 -> CASBOT02 重定向结果（G1 与 CASBOT02 并排对比）。
# 关闭当前 MuJoCo 窗口即播放下一个；全部播完自动退出。
set -uo pipefail

PYBIN="${PYBIN:-$HOME/miniconda3/envs/robot_retargeter/bin/python}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

shopt -s nullglob
FILES=(output_data/robot_motion/*_from_g1_CASBOT02.csv)
TOTAL=${#FILES[@]}
echo "[info] ${TOTAL} motions to review. Close the viewer window to advance."

idx=0
for f in "${FILES[@]}"; do
  base="$(basename "${f}" .csv)"          # <name>_from_g1_CASBOT02
  motion="${base%_CASBOT02}"              # <name>_from_g1
  idx=$((idx + 1))
  echo "=== [${idx}/${TOTAL}] ${motion} ==="
  "${PYBIN}" scripts/multi_robot_visualize.py \
    --motion "${motion}" \
    --robots g1 CASBOT02 \
    --source_fps 30 --render_fps 30 --loop 2>/dev/null | grep -v Playback || true
done

echo "[done] all motions reviewed."
