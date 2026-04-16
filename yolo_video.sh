#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/xu/anaconda3/envs/condacommon/bin/python}"

WEIGHTS="${WEIGHTS:-visual_recognition/runs/ct_t_yolo/weights/best.pt}"
SOURCE="${SOURCE:-udp://192.168.221.36:1234}"
CONF="${CONF:-0.25}"
IMGSZ="${IMGSZ:-640}"
DEVICE="${DEVICE:-0}"
FPS="${FPS:-60}"
RUN_NAME="${RUN_NAME:-yolo_video}"
SHOW_WINDOW="${SHOW_WINDOW:-1}"
SAVE_VIDEO="${SAVE_VIDEO:-1}"
OUT_STREAM="${OUT_STREAM:-}"
STREAM_FPS="${STREAM_FPS:-60}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[yolo_video] PYTHON_BIN not found or not executable: $PYTHON_BIN" >&2
  exit 1
fi

CMD=(
  "$PYTHON_BIN" "$ROOT_DIR/visual_recognition/yolor.py"
  --weights "$WEIGHTS"
  --source "$SOURCE"
  --conf "$CONF"
  --imgsz "$IMGSZ"
  --device "$DEVICE"
  --fps "$FPS"
  --name "$RUN_NAME"
  --stream-fps "$STREAM_FPS"
)

if [[ "$SHOW_WINDOW" == "1" ]]; then
  CMD+=(--show)
fi
if [[ "$SAVE_VIDEO" == "1" ]]; then
  CMD+=(--save-video)
fi
if [[ -n "$OUT_STREAM" ]]; then
  CMD+=(--out-stream "$OUT_STREAM")
fi

echo "[yolo_video] Running: ${CMD[*]}"
exec "${CMD[@]}"
