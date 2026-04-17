#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/xu/anaconda3/envs/condacommon/bin/python}"

# 兼容保留：advisor 已不再直接拉流。
SOURCE="${SOURCE:-${OUT_STREAM:-udp://@:2234?fifo_size=32768&overrun_nonfatal=1}}"
WEIGHTS="${WEIGHTS:-$ROOT_DIR/visual_recognition/runs/xu/weights/best.pt}"
SHARED_FRAME_PATH="${SHARED_FRAME_PATH:-${CSRL_SHARED_FRAME_PATH:-/tmp/cs_rl_latest_frame.jpg}}"
SHARED_STATE_PATH="${SHARED_STATE_PATH:-${CSRL_SHARED_STATE_PATH:-/tmp/cs_rl_runtime_state.json}}"
POLL_INTERVAL_SEC="${POLL_INTERVAL_SEC:-0.10}"

CONF="${CONF:-0.30}"
IMGSZ="${IMGSZ:-128}"
DEVICE="${DEVICE:-0}"
HALF="${HALF:-1}"
INFER_EVERY="${INFER_EVERY:-3}"
DETECT_ROI="${DETECT_ROI:-0.00,0.08,1.00,0.84}"

OCR_ENGINE="${OCR_ENGINE:-pytesseract}"
OCR_ROI="${OCR_ROI:-}"
OCR_WHITELIST="${OCR_WHITELIST:-0123456789/%:HPARMOABULLET}"
OCR_MIN_CONF="${OCR_MIN_CONF:-0.20}"

LOCATION_ROI="${LOCATION_ROI:-0.00,0.0,0.150,0.346}"
QWEN_MODEL="${QWEN_MODEL:-qwen3.6-plus}"
API_KEY="${API_KEY:-${DASHSCOPE_API_KEY:-${QWEN_API_KEY:-${OPENAI_API_KEY:-}}}}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[decision_advisor] PYTHON_BIN 不存在或不可执行: $PYTHON_BIN" >&2
  exit 1
fi

if [[ ! -f "$WEIGHTS" ]]; then
  echo "[decision_advisor] 权重文件不存在: $WEIGHTS" >&2
  exit 1
fi

CMD=(
  "$PYTHON_BIN" "$ROOT_DIR/decision_advisor.py"
  --source "$SOURCE"
  --weights "$WEIGHTS"
  --shared-frame-path "$SHARED_FRAME_PATH"
  --shared-state-path "$SHARED_STATE_PATH"
  --poll-interval-sec "$POLL_INTERVAL_SEC"
  --conf "$CONF"
  --imgsz "$IMGSZ"
  --device "$DEVICE"
  --infer-every "$INFER_EVERY"
  --detect-roi "$DETECT_ROI"
  --ocr-engine "$OCR_ENGINE"
  --ocr-whitelist "$OCR_WHITELIST"
  --ocr-min-conf "$OCR_MIN_CONF"
  --location-roi "$LOCATION_ROI"
  --qwen-model "$QWEN_MODEL"
)

if [[ "$HALF" == "1" || "$HALF" == "true" || "$HALF" == "TRUE" ]]; then
  CMD+=(--half)
fi

if [[ -n "$OCR_ROI" ]]; then
  IFS=';' read -r -a OCR_ROI_ARR <<< "$OCR_ROI"
  for item in "${OCR_ROI_ARR[@]}"; do
    if [[ -n "${item// }" ]]; then
      CMD+=(--ocr-roi "$item")
    fi
  done
fi

if [[ -n "$API_KEY" ]]; then
  export DASHSCOPE_API_KEY="$API_KEY"
fi

echo "[decision_advisor] 启动命令:"
echo "${CMD[*]}"
if [[ -n "$API_KEY" ]]; then
  echo "[decision_advisor] API key: loaded from env (masked)"
fi
echo "[decision_advisor] 共享帧路径: $SHARED_FRAME_PATH"
echo "[decision_advisor] 共享状态路径: $SHARED_STATE_PATH"
echo "[decision_advisor] 当前 SOURCE(兼容参数，可忽略): $SOURCE"
echo "[decision_advisor] 运行后可在终端输入: ocr / pos / help / quit"

exec "${CMD[@]}"
