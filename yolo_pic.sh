#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/xu/anaconda3/envs/condacommon/bin/python}"

WEIGHTS="${WEIGHTS:-visual_recognition/runs/ct_t_yolo/weights/best.pt}"
SOURCE="${SOURCE:-screenshots/latest.png}"
CONF="${CONF:-0.25}"
IMGSZ="${IMGSZ:-640}"
DEVICE="${DEVICE:-0}"
RUN_NAME="${RUN_NAME:-yolo_pic}"
SHOW_WINDOW="${SHOW_WINDOW:-1}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[yolo_pic] PYTHON_BIN not found or not executable: $PYTHON_BIN" >&2
  exit 1
fi

if [[ ! -f "$ROOT_DIR/$SOURCE" ]]; then
  echo "[yolo_pic] SOURCE image not found: $ROOT_DIR/$SOURCE" >&2
  echo "[yolo_pic] Use: SOURCE=path/to/image.png bash yolo_pic.sh" >&2
  exit 1
fi

CMD=(
  "$PYTHON_BIN" "$ROOT_DIR/visual_recognition/yolor.py"
  --weights "$WEIGHTS"
  --source "$ROOT_DIR/$SOURCE"
  --conf "$CONF"
  --imgsz "$IMGSZ"
  --device "$DEVICE"
  --name "$RUN_NAME"
)

if [[ "$SHOW_WINDOW" == "1" ]]; then
  CMD+=(--show)
fi

echo "[yolo_pic] Running: ${CMD[*]}"
exec "${CMD[@]}"
