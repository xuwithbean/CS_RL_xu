#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/xu/anaconda3/envs/condacommon/bin/python}"

INPUT="${INPUT:-}"
OUTPUT="${OUTPUT:-${ROOT_DIR}/cropped_images}"
ROI="${ROI:-0.00,0.08,1.00,0.84}"
RECURSIVE="${RECURSIVE:-1}"
PRESERVE_TREE="${PRESERVE_TREE:-1}"
SUFFIX="${SUFFIX:-_crop}"
OVERWRITE="${OVERWRITE:-0}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[crop] PYTHON_BIN not found or not executable: $PYTHON_BIN" >&2
  exit 1
fi

if [[ -z "$INPUT" ]]; then
  echo "[crop] INPUT is empty." >&2
  echo "[crop] Example: INPUT=path/to/images OUTPUT=path/to/out bash crop_images.sh" >&2
  exit 1
fi

CMD=(
  "$PYTHON_BIN" "$ROOT_DIR/crop_images.py"
  --input "$INPUT"
  --output "$OUTPUT"
  --roi "$ROI"
  --suffix "$SUFFIX"
)

if [[ "$RECURSIVE" == "1" ]]; then
  CMD+=(--recursive)
fi
if [[ "$PRESERVE_TREE" == "1" ]]; then
  CMD+=(--preserve-tree)
fi
if [[ "$OVERWRITE" == "1" ]]; then
  CMD+=(--overwrite)
fi

echo "[crop] Running: ${CMD[*]}"
exec "${CMD[@]}"
