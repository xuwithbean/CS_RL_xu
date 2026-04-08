#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/xu/anaconda3/envs/condacommon/bin/python}"

SOURCE="${SOURCE:-udp://192.168.221.36:1234}"
OCR_ENGINE="${OCR_ENGINE:-pytesseract}"
OCR_ROI="${OCR_ROI:-0.00,0.78,0.42,0.22}"
OCR_MIN_CONF="${OCR_MIN_CONF:-0.20}"
RUN_NAME="${RUN_NAME:-ocr_video}"
SHOW_WINDOW="${SHOW_WINDOW:-1}"
SAVE_VIDEO="${SAVE_VIDEO:-1}"
FPS="${FPS:-60}"
OUT_STREAM="${OUT_STREAM:-}"
STREAM_FPS="${STREAM_FPS:-60}"
DEVICE="${DEVICE:-0}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[ocr_video] PYTHON_BIN not found or not executable: $PYTHON_BIN" >&2
  exit 1
fi

CMD=(
  "$PYTHON_BIN" "$ROOT_DIR/visual_recognition/ocrr.py"
  --source "$SOURCE"
  --ocr-engine "$OCR_ENGINE"
  --ocr-roi "$OCR_ROI"
  --ocr-min-conf "$OCR_MIN_CONF"
  --name "$RUN_NAME"
  --fps "$FPS"
  --stream-fps "$STREAM_FPS"
  --device "$DEVICE"
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

echo "[ocr_video] Running: ${CMD[*]}"
exec "${CMD[@]}"
