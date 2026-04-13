#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/xu/anaconda3/envs/condacommon/bin/python}"

WEIGHTS="${WEIGHTS:-visual_recognition/runs/ct_t_yolo_fix_test/weights/best.pt}"
SOURCE="${SOURCE:-udp://192.168.221.36:1234}"
PROJECT="${PROJECT:-visual_recognition/runs}"
NAME="${NAME:-recognize}"
CONF="${CONF:-0.25}"
IMGSZ="${IMGSZ:-640}"
DEVICE="${DEVICE:-0}"
HEAD_RATIO="${HEAD_RATIO:-0.30}"
HEAD_WIDTH_RATIO="${HEAD_WIDTH_RATIO:-0.45}"
LINE_WIDTH="${LINE_WIDTH:-2}"
FPS="${FPS:-60}"
STREAM_FPS="${STREAM_FPS:-60}"
SHOW_WINDOW="${SHOW_WINDOW:-1}"
SAVE_VIDEO="${SAVE_VIDEO:-1}"
OCR="${OCR:-1}"
OCR_ENGINE="${OCR_ENGINE:-pytesseract}"
OCR_ROI="${OCR_ROI:-0.00,0.78,0.42,0.22}"
OCR_MIN_CONF="${OCR_MIN_CONF:-0.20}"
OCR_WHITELIST="${OCR_WHITELIST:-0123456789/%:HPARMOABULLET}"
PRINT_OCR="${PRINT_OCR:-1}"
DETECT_ROI="${DETECT_ROI:-0.00,0.08,1.00,0.84}"
PRINT_YOLO="${PRINT_YOLO:-0}"
YOLO_INFO_JSONL="${YOLO_INFO_JSONL:-}"
OUT_STREAM="${OUT_STREAM:-}"
OCR_INFO_JSONL="${OCR_INFO_JSONL:-}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[recognize] PYTHON_BIN not found or not executable: $PYTHON_BIN" >&2
  echo "[recognize] Hint: set PYTHON_BIN or activate your conda env first." >&2
  exit 1
fi

if [[ -z "$SOURCE" ]]; then
  echo "[recognize] SOURCE is empty." >&2
  exit 1
fi

CMD=(
  "$PYTHON_BIN" "$ROOT_DIR/visual_recognition/predict.py"
  --weights "$WEIGHTS"
  --source "$SOURCE"
  --conf "$CONF"
  --imgsz "$IMGSZ"
  --device "$DEVICE"
  --project "$PROJECT"
  --name "$NAME"
  --head-ratio "$HEAD_RATIO"
  --head-width-ratio "$HEAD_WIDTH_RATIO"
  --line-width "$LINE_WIDTH"
  --fps "$FPS"
  --stream-fps "$STREAM_FPS"
  --detect-roi "$DETECT_ROI"
)

if [[ "$SHOW_WINDOW" == "1" ]]; then
  CMD+=(--show)
fi
if [[ "$SAVE_VIDEO" == "1" ]]; then
  CMD+=(--save-video)
fi
if [[ "$OCR" == "1" ]]; then
  CMD+=(
    --ocr
    --ocr-engine "$OCR_ENGINE"
    --ocr-roi "$OCR_ROI"
    --ocr-min-conf "$OCR_MIN_CONF"
    --ocr-whitelist "$OCR_WHITELIST"
  )
fi
if [[ "$PRINT_OCR" == "1" ]]; then
  CMD+=(--print-ocr)
fi
if [[ "$PRINT_YOLO" == "1" ]]; then
  CMD+=(--print-yolo)
fi
if [[ -n "$OUT_STREAM" ]]; then
  CMD+=(--out-stream "$OUT_STREAM")
fi
if [[ -n "$OCR_INFO_JSONL" ]]; then
  CMD+=(--ocr-info-jsonl "$OCR_INFO_JSONL")
fi
if [[ -n "$YOLO_INFO_JSONL" ]]; then
  CMD+=(--yolo-info-jsonl "$YOLO_INFO_JSONL")
fi

echo "[recognize] Running: ${CMD[*]}"
exec "${CMD[@]}"
