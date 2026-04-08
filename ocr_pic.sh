#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/xu/anaconda3/envs/condacommon/bin/python}"

SOURCE="${SOURCE:-screenshots/latest.png}"
OCR_ENGINE="${OCR_ENGINE:-pytesseract}"
OCR_ROI="${OCR_ROI:-0.00,0.78,0.42,0.22}"
OCR_MIN_CONF="${OCR_MIN_CONF:-0.20}"
RUN_NAME="${RUN_NAME:-ocr_pic}"
SHOW_WINDOW="${SHOW_WINDOW:-1}"
DEVICE="${DEVICE:-0}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[ocr_pic] PYTHON_BIN not found or not executable: $PYTHON_BIN" >&2
  exit 1
fi

if [[ ! -f "$ROOT_DIR/$SOURCE" ]]; then
  echo "[ocr_pic] SOURCE image not found: $ROOT_DIR/$SOURCE" >&2
  echo "[ocr_pic] Use: SOURCE=path/to/image.png bash ocr_pic.sh" >&2
  exit 1
fi

CMD=(
  "$PYTHON_BIN" "$ROOT_DIR/visual_recognition/ocrr.py"
  --source "$ROOT_DIR/$SOURCE"
  --ocr-engine "$OCR_ENGINE"
  --ocr-roi "$OCR_ROI"
  --ocr-min-conf "$OCR_MIN_CONF"
  --name "$RUN_NAME"
  --device "$DEVICE"
)

if [[ "$SHOW_WINDOW" == "1" ]]; then
  CMD+=(--show)
fi

echo "[ocr_pic] Running: ${CMD[*]}"
exec "${CMD[@]}"
