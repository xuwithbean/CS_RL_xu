#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/xu/anaconda3/envs/condacommon/bin/python}"

# 可按需修改的默认参数
WEIGHTS="${WEIGHTS:-visual_recognition/runs/ct_t_yolo/weights/best.pt}"
IN_STREAM="${IN_STREAM:-udp://192.168.221.36:1234}"
OUT_STREAM="${OUT_STREAM:-udp://127.0.0.1:2234}"
MONITOR="${MONITOR:-2}"
FPS="${FPS:-60}"
BITRATE="${BITRATE:-2500k}"
CONF="${CONF:-0.25}"
IMGSZ="${IMGSZ:-640}"
DEVICE="${DEVICE:-0}"
RUN_NAME="${RUN_NAME:-ct_t_realtime}"

PREVIEW_FLAG=""
SKIP_STREAM_FLAG=""
SHOW_FLAG=""

if [[ "${PREVIEW:-1}" == "1" ]]; then
  PREVIEW_FLAG="--preview"
fi

if [[ "${SKIP_WIN_STREAM:-0}" == "1" ]]; then
  SKIP_STREAM_FLAG="--skip-win-stream"
fi

if [[ "${SHOW_WINDOW:-0}" == "1" ]]; then
  SHOW_FLAG="--show"
fi

cd "$ROOT_DIR"

exec "$PYTHON_BIN" "$ROOT_DIR/visual_recognition/realtime_pipeline.py" \
  --weights "$WEIGHTS" \
  --in-stream "$IN_STREAM" \
  --out-stream "$OUT_STREAM" \
  --monitor "$MONITOR" \
  --framerate "$FPS" \
  --bitrate "$BITRATE" \
  --conf "$CONF" \
  --imgsz "$IMGSZ" \
  --device "$DEVICE" \
  --name "$RUN_NAME" \
  $PREVIEW_FLAG \
  $SKIP_STREAM_FLAG \
  $SHOW_FLAG
