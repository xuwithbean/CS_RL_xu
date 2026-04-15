#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/xu/anaconda3/envs/condacommon/bin/python}"

# Windows 游戏与推流参数
GAME_EXE="${GAME_EXE:-E:\\steam\\steamapps\\common\\Counter-Strike Global Offensive\\game\\bin\\win64\\cs2.exe}"
WINDOW_TITLE="${WINDOW_TITLE:-auto}"
WAIT_GAME="${WAIT_GAME:-6.0}"
AUTO_LINUX_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
LINUX_IP="${LINUX_IP:-${AUTO_LINUX_IP:-127.0.0.1}}"
PORT="${PORT:-12345}"
IN_STREAM="${IN_STREAM:-udp://${LINUX_IP}:${PORT}}"
OUT_STREAM="${OUT_STREAM:-}"

# YOLO 参数
WEIGHTS="${WEIGHTS:-}"
CONF="${CONF:-0.30}"
IMGSZ="${IMGSZ:-256}"
FRAME_DRAIN="${FRAME_DRAIN:-4}"
WORK_SIZE="${WORK_SIZE:-704x396}"
PREVIEW_SIZE="${PREVIEW_SIZE:-800x450}"
INFER_EVERY="${INFER_EVERY:-1}"
HALF="${HALF:-1}"
DEVICE="${DEVICE:-0}"
UDP_FIFO_SIZE="${UDP_FIFO_SIZE:-65536}"

if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L >/dev/null 2>&1; then
  echo "[yolo_ffplay] NVIDIA GPU not available. This mode requires GPU and will not fallback to CPU." >&2
  exit 1
fi
RUN_NAME="${RUN_NAME:-ct_t_yolo_ffplay}"
DETECT_ROI="${DETECT_ROI:-0.00,0.08,1.00,0.84}"

# 可选扩展
SHOW_WINDOW="${SHOW_WINDOW:-0}"
PREVIEW="${PREVIEW:-1}"
PRINT_YOLO="${PRINT_YOLO:-0}"
YOLO_INFO_JSONL="${YOLO_INFO_JSONL:-}"
OCR="${OCR:-0}"
OCR_ENGINE="${OCR_ENGINE:-pytesseract}"
OCR_ROI="${OCR_ROI:-}"
OCR_MIN_CONF="${OCR_MIN_CONF:-0.20}"
OCR_WHITELIST="${OCR_WHITELIST:-0123456789/%:HPARMOABULLET}"
OCR_INFO_JSONL="${OCR_INFO_JSONL:-}"
SKIP_WIN_STREAM="${SKIP_WIN_STREAM:-0}"
FFPLAY_BIN="${FFPLAY_BIN:-ffplay}"

DEFAULT_WEIGHTS_CANDIDATES=(
  "$ROOT_DIR/visual_recognition/runs/xu/weights/best.pt"
  "$ROOT_DIR/visual_recognition/runs/xu/weights/last.pt"
  "$ROOT_DIR/yolo11n.pt"
)

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[yolo_ffplay] PYTHON_BIN not found or not executable: $PYTHON_BIN" >&2
  echo "[yolo_ffplay] Hint: set PYTHON_BIN or activate conda env first." >&2
  exit 1
fi

if [[ -z "$WEIGHTS" ]]; then
  for candidate in "${DEFAULT_WEIGHTS_CANDIDATES[@]}"; do
    if [[ -f "$candidate" ]]; then
      WEIGHTS="$candidate"
      break
    fi
  done
fi

if [[ ! -f "$WEIGHTS" ]]; then
  echo "[yolo_ffplay] weights not found: $WEIGHTS" >&2
  echo "[yolo_ffplay] available candidates:" >&2
  for candidate in "${DEFAULT_WEIGHTS_CANDIDATES[@]}"; do
    echo "[yolo_ffplay]   - $candidate" >&2
  done
  echo "[yolo_ffplay] hint: set WEIGHTS=/path/to/best.pt before running" >&2
  exit 1
fi

CMD=(
  "$PYTHON_BIN" "$ROOT_DIR/visual_recognition/stream_ffplay_pipeline.py"
  --game-exe "$GAME_EXE"
  --window-title "$WINDOW_TITLE"
  --wait-game "$WAIT_GAME"
  --weights "$WEIGHTS"
  --in-stream "$IN_STREAM"
  --out-stream "$OUT_STREAM"
  --conf "$CONF"
  --imgsz "$IMGSZ"
  --device "$DEVICE"
  --name "$RUN_NAME"
  --detect-roi "$DETECT_ROI"
)

if [[ "$SHOW_WINDOW" == "1" ]]; then
  CMD+=(--show)
fi
if [[ "$PREVIEW" != "0" ]]; then
  CMD+=(--preview)
fi
if [[ "$PRINT_YOLO" == "1" ]]; then
  CMD+=(--print-yolo)
fi
if [[ -n "$YOLO_INFO_JSONL" ]]; then
  CMD+=(--yolo-info-jsonl "$YOLO_INFO_JSONL")
fi
if [[ "$OCR" == "1" ]]; then
  CMD+=(--ocr --ocr-engine "$OCR_ENGINE")
  if [[ -n "$OCR_ROI" ]]; then
    CMD+=(--ocr-roi "$OCR_ROI")
  fi
  CMD+=(--ocr-min-conf "$OCR_MIN_CONF" --ocr-whitelist "$OCR_WHITELIST")
  if [[ -n "$OCR_INFO_JSONL" ]]; then
    CMD+=(--ocr-info-jsonl "$OCR_INFO_JSONL")
  fi
fi
if [[ "$SKIP_WIN_STREAM" == "1" ]]; then
  CMD+=(--skip-win-stream)
fi
CMD+=(--frame-drain "$FRAME_DRAIN")
CMD+=(--work-size "$WORK_SIZE")
CMD+=(--preview-size "$PREVIEW_SIZE")
CMD+=(--infer-every "$INFER_EVERY")
CMD+=(--udp-fifo-size "$UDP_FIFO_SIZE")
if [[ "$HALF" == "1" ]]; then
  CMD+=(--half)
fi
CMD+=(--ffplay "$FFPLAY_BIN")
if [[ -n "$OUT_STREAM" ]]; then
  CMD+=(--out-stream "$OUT_STREAM")
fi

echo "[yolo_ffplay] Running: ${CMD[*]}"
exec "${CMD[@]}"
