#!/usr/bin/env bash
set -euo pipefail

# 实时识别启动器：输入流 -> YOLO -> 输出流/预览。
# 示例：
#   bash yolorun.sh
#   IN_STREAM=udp://192.168.221.1:12345 OUT_STREAM=udp://127.0.0.1:2234 bash yolorun.sh

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/xu/anaconda3/envs/condacommon/bin/python}"

if [[ "${1:-}" == "--help" ]] || [[ "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage:
  bash yolorun.sh

Common overrides:
  IN_STREAM=udp://192.168.221.1:12345
  OUT_STREAM=udp://127.0.0.1:2234
  WEIGHTS=/path/to/best.pt
  DEVICE=0
  HALF=1
  INFER_EVERY=1
  WORK_SIZE=704x396
  STREAM_FPS=60
  OUT_VCODEC=mpeg2video

Examples:
  bash yolorun.sh
  IN_STREAM=udp://192.168.221.1:12345 PREVIEW=1 OUT_STREAM=udp://127.0.0.1:2234 bash yolorun.sh
EOF
  exit 0
fi

AUTO_LINUX_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
LINUX_IP="${LINUX_IP:-${AUTO_LINUX_IP:-127.0.0.1}}"
PORT="${PORT:-12345}"

IN_STREAM="${IN_STREAM:-udp://${LINUX_IP}:${PORT}}"
OUT_STREAM="${OUT_STREAM:-}"

WEIGHTS="${WEIGHTS:-}"
CONF="${CONF:-0.30}"
IMGSZ="${IMGSZ:-320}"
DEVICE="${DEVICE:-0}"
HALF="${HALF:-1}"
INFER_EVERY="${INFER_EVERY:-1}"

WORK_SIZE="${WORK_SIZE:-704x396}"
DETECT_ROI="${DETECT_ROI:-0.00,0.08,1.00,0.84}"
LINE_WIDTH="${LINE_WIDTH:-2}"
STREAM_FPS="${STREAM_FPS:-60}"

PREVIEW="${PREVIEW:-1}"
SHOW_WINDOW="${SHOW_WINDOW:-0}"

CAPTURE_DRAIN="${CAPTURE_DRAIN:-0}"
CAPTURE_RECONNECT_SEC="${CAPTURE_RECONNECT_SEC:-2.0}"
CAPTURE_TIMEOUT_MS="${CAPTURE_TIMEOUT_MS:-4000}"
UDP_FIFO_SIZE="${UDP_FIFO_SIZE:-1048576}"
CAPTURE_PROBESIZE="${CAPTURE_PROBESIZE:-131072}"
CAPTURE_ANALYZEDURATION="${CAPTURE_ANALYZEDURATION:-2000000}"

OUT_VCODEC="${OUT_VCODEC:-mpeg2video}"
OUT_BITRATE="${OUT_BITRATE:-2200k}"
FFMPEG_BIN="${FFMPEG_BIN:-ffmpeg}"
FFPLAY_BIN="${FFPLAY_BIN:-ffplay}"

DEFAULT_WEIGHTS_CANDIDATES=(
  "$ROOT_DIR/visual_recognition/runs/xu/weights/best.pt"
  "$ROOT_DIR/visual_recognition/runs/xu/weights/last.pt"
  "$ROOT_DIR/yolo11n.pt"
)

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[yolorun] PYTHON_BIN not found: $PYTHON_BIN" >&2
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
  echo "[yolorun] weights not found: $WEIGHTS" >&2
  for candidate in "${DEFAULT_WEIGHTS_CANDIDATES[@]}"; do
    echo "[yolorun]   - $candidate" >&2
  done
  exit 1
fi

CMD=(
  "$PYTHON_BIN" "$ROOT_DIR/visual_recognition/stream_ffplay_pipeline.py"
  --weights "$WEIGHTS"
  --in-stream "$IN_STREAM"
  --out-stream "$OUT_STREAM"
  --conf "$CONF"
  --imgsz "$IMGSZ"
  --device "$DEVICE"
  --infer-every "$INFER_EVERY"
  --work-size "$WORK_SIZE"
  --detect-roi "$DETECT_ROI"
  --line-width "$LINE_WIDTH"
  --stream-fps "$STREAM_FPS"
  --capture-drain "$CAPTURE_DRAIN"
  --capture-reconnect-sec "$CAPTURE_RECONNECT_SEC"
  --capture-timeout-ms "$CAPTURE_TIMEOUT_MS"
  --udp-fifo-size "$UDP_FIFO_SIZE"
  --capture-probesize "$CAPTURE_PROBESIZE"
  --capture-analyzeduration "$CAPTURE_ANALYZEDURATION"
  --out-vcodec "$OUT_VCODEC"
  --out-bitrate "$OUT_BITRATE"
  --ffmpeg "$FFMPEG_BIN"
  --ffplay "$FFPLAY_BIN"
)

if [[ "$HALF" == "1" ]]; then
  CMD+=(--half)
fi

if [[ "$PREVIEW" == "1" ]]; then
  CMD+=(--preview)
fi

if [[ "$SHOW_WINDOW" == "1" ]]; then
  CMD+=(--show)
fi

if [[ -z "$OUT_STREAM" ]]; then
  # 避免传空字符串导致解析歧义。
  for i in "${!CMD[@]}"; do
    if [[ "${CMD[$i]}" == "--out-stream" ]]; then
      unset 'CMD[i]'
      unset 'CMD[i+1]'
      break
    fi
  done
  CMD=("${CMD[@]}")
fi

echo "[yolorun] Running: ${CMD[*]}"
exec "${CMD[@]}"
