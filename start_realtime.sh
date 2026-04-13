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
DETECT_ROI="${DETECT_ROI:-0.00,0.08,1.00,0.84}"
PRINT_YOLO="${PRINT_YOLO:-0}"
YOLO_INFO_JSONL="${YOLO_INFO_JSONL:-}"

PREVIEW_FLAG=""
SKIP_STREAM_FLAG=""
SHOW_FLAG=""
PRINT_YOLO_FLAG=""

if [[ "${PREVIEW:-1}" == "1" ]]; then
  PREVIEW_FLAG="--preview"
fi

if [[ "${SKIP_WIN_STREAM:-0}" == "1" ]]; then
  SKIP_STREAM_FLAG="--skip-win-stream"
fi

if [[ "${SHOW_WINDOW:-0}" == "1" ]]; then
  SHOW_FLAG="--show"
fi

if [[ "$PRINT_YOLO" == "1" ]]; then
  PRINT_YOLO_FLAG="--print-yolo"
fi

YOLO_JSONL_ARGS=()
if [[ -n "$YOLO_INFO_JSONL" ]]; then
  YOLO_JSONL_ARGS=(--yolo-info-jsonl "$YOLO_INFO_JSONL")
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
  --detect-roi "$DETECT_ROI" \
  --name "$RUN_NAME" \
  $PRINT_YOLO_FLAG \
  $PREVIEW_FLAG \
  $SKIP_STREAM_FLAG \
  $SHOW_FLAG \
  "${YOLO_JSONL_ARGS[@]}"
