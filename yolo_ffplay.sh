#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/xu/anaconda3/envs/condacommon/bin/python}"

# Windows 游戏与推流参数
GAME_EXE="${GAME_EXE:-E:\\steam\\steamapps\\common\\Counter-Strike Global Offensive\\game\\bin\\win64\\cs2.exe}"
WINDOW_TITLE="${WINDOW_TITLE:-auto}"
WAIT_GAME="${WAIT_GAME:-6.0}"

is_bad_windows_ip() {
  local ip="$1"
  [[ -z "$ip" ]] && return 0
  [[ "$ip" =~ ^127\. ]] && return 0
  [[ "$ip" =~ ^169\.254\. ]] && return 0
  [[ "$ip" =~ ^100\.(6[4-9]|[7-9][0-9]|1[0-1][0-9]|12[0-7])\. ]] && return 0
  return 1
}

check_ip_reachable() {
  local ip="$1"
  ping -c 1 -W 1 "$ip" >/dev/null 2>&1
}

detect_windows_ip() {
  local candidates=()
  local ns_ip=""
  local gw_ip=""
  local ps_ips=""
  local ip=""

  ns_ip="$(awk '/^nameserver /{print $2; exit}' /etc/resolv.conf 2>/dev/null || true)"
  gw_ip="$(ip route show default 2>/dev/null | awk '/default/ {print $3; exit}')"
  ps_ips="$(powershell.exe -NoProfile -Command "Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue | Where-Object { $_.IPAddress -and $_.IPAddress -notlike '127.*' -and $_.IPAddress -notlike '169.254.*' } | Select-Object -ExpandProperty IPAddress" 2>/dev/null | tr -d '\r' || true)"

  [[ -n "$ns_ip" ]] && candidates+=("$ns_ip")
  [[ -n "$gw_ip" ]] && candidates+=("$gw_ip")
  while IFS= read -r ip; do
    [[ -n "$ip" ]] && candidates+=("$ip")
  done <<< "$ps_ips"

  # 去重并校验可达性，优先选择可达的候选地址。
  local -A seen=()
  for ip in "${candidates[@]}"; do
    [[ -n "${seen[$ip]:-}" ]] && continue
    seen["$ip"]=1
    if is_bad_windows_ip "$ip"; then
      echo "[yolo_ffplay] Windows IP check: skip candidate=$ip" >&2
      continue
    fi
    if check_ip_reachable "$ip"; then
      echo "[yolo_ffplay] Windows IP check: selected reachable candidate=$ip" >&2
      echo "$ip"
      return 0
    fi
    echo "[yolo_ffplay] Windows IP check: candidate not reachable=$ip" >&2
  done

  # 没有可达结果时，回退到非保留地址候选。
  for ip in "${candidates[@]}"; do
    if ! is_bad_windows_ip "$ip"; then
      echo "[yolo_ffplay] Windows IP check: fallback candidate=$ip" >&2
      echo "$ip"
      return 0
    fi
  done

  echo "127.0.0.1"
  return 0
}

AUTO_LINUX_IP="$(ip -4 route get 1.1.1.1 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="src"){print $(i+1); exit}}')"
if [[ -z "$AUTO_LINUX_IP" ]]; then
  AUTO_LINUX_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
fi
LINUX_IP="${LINUX_IP:-${AUTO_LINUX_IP:-127.0.0.1}}"
PORT="${PORT:-12345}"
IN_STREAM="${IN_STREAM:-udp://${LINUX_IP}:${PORT}}"
OUT_STREAM="${OUT_STREAM:-}"
FRAMERATE="${FRAMERATE:-30}"
BITRATE="${BITRATE:-4000k}"
AUTO_WINDOWS_IP="$(detect_windows_ip)"
WINDOWS_IP="${WINDOWS_IP:-${AUTO_WINDOWS_IP:-127.0.0.1}}"
if [[ -z "$OUT_STREAM" ]]; then
  OUT_STREAM="udp://${WINDOWS_IP}:2234?pkt_size=1316&buffer_size=1048576"
fi

# YOLO 参数
WEIGHTS="${WEIGHTS:-}"
ACCEL_MODE="${ACCEL_MODE:-auto}"
CONF="${CONF:-0.30}"
IMGSZ_SET="${IMGSZ+x}"
WORK_SIZE_SET="${WORK_SIZE+x}"
INFER_EVERY_SET="${INFER_EVERY+x}"
OUTPUT_MAX_SIZE_SET="${OUTPUT_MAX_SIZE+x}"
FRAME_DRAIN_SET="${FRAME_DRAIN+x}"
CAPTURE_DRAIN_SET="${CAPTURE_DRAIN+x}"
STREAM_FPS_SET="${STREAM_FPS+x}"
BOXES_TTL_MS_SET="${BOXES_TTL_MS+x}"
OUT_BITRATE_SET="${OUT_BITRATE+x}"
STREAM_VCODEC_SET="${STREAM_VCODEC+x}"
IMGSZ="${IMGSZ:-128}"
FRAME_DRAIN="${FRAME_DRAIN:-2}"
CAPTURE_DRAIN="${CAPTURE_DRAIN:-2}"
WORK_SIZE="${WORK_SIZE:-288x162}"
OUTPUT_MAX_SIZE="${OUTPUT_MAX_SIZE:-768x432}"
PREVIEW_SIZE="${PREVIEW_SIZE:-800x450}"
INFER_EVERY="${INFER_EVERY:-2}"
HALF="${HALF:-1}"
DEVICE="${DEVICE:-0}"
OUT_BITRATE="${OUT_BITRATE:-4000k}"
STREAM_FPS="${STREAM_FPS:-30}"
BOXES_TTL_MS="${BOXES_TTL_MS:-350}"
UDP_FIFO_SIZE="${UDP_FIFO_SIZE:-32768}"
CAPTURE_RECONNECT_SEC="${CAPTURE_RECONNECT_SEC:-3.0}"
CAPTURE_TIMEOUT_MS="${CAPTURE_TIMEOUT_MS:-5000}"
FIRST_FRAME_TIMEOUT_SEC="${FIRST_FRAME_TIMEOUT_SEC:-20.0}"
CAPTURE_PROBESIZE="${CAPTURE_PROBESIZE:-32768}"
CAPTURE_ANALYZEDURATION="${CAPTURE_ANALYZEDURATION:-300000}"
SENDER_UDP_PKT_SIZE="${SENDER_UDP_PKT_SIZE:-1316}"
SENDER_UDP_BUFFER_SIZE="${SENDER_UDP_BUFFER_SIZE:-262144}"
STREAM_VCODEC="${STREAM_VCODEC:-mpeg2video}"

if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L >/dev/null 2>&1; then
  echo "[yolo_ffplay] NVIDIA GPU not available. This mode requires GPU and will not fallback to CPU." >&2
  exit 1
fi
RUN_NAME="${RUN_NAME:-ct_t_yolo_ffplay}"
DETECT_ROI="${DETECT_ROI:-0.00,0.08,1.00,0.84}"

# 可选扩展
PREVIEW="${PREVIEW:-1}"
PRINT_YOLO="${PRINT_YOLO:-0}"
YOLO_INFO_JSONL="${YOLO_INFO_JSONL:-}"
OCR="${OCR:-1}"
OCR_ENGINE="${OCR_ENGINE:-pytesseract}"
OCR_LANG="${OCR_LANG:-eng}"
OCR_CN_LANG="${OCR_CN_LANG:-chi_sim+eng}"
OCR_ROI="${OCR_ROI:-}"
DRAW_OCR_ROI="${DRAW_OCR_ROI:-1}"
OCR_MIN_CONF="${OCR_MIN_CONF:-0.20}"
OCR_WHITELIST="${OCR_WHITELIST:-0123456789}"
OCR_INFO_JSONL="${OCR_INFO_JSONL:-}"
OCR_EVERY="${OCR_EVERY:-10}"
LOCATION_DETECT="${LOCATION_DETECT:-1}"
LOCATION_EVERY="${LOCATION_EVERY:-5}"
LOCATION_ROI="${LOCATION_ROI:-0.00,0.0,0.150,0.346}"
LOCATION_MODEL="${LOCATION_MODEL:-qwen3.6-plus}"
if [[ "$LOCATION_MODEL" == "deepseek-chat" || "$LOCATION_MODEL" == "deepseek-reasoner" || "$LOCATION_MODEL" == "qwen-vl-plus" ]]; then
  echo "[yolo_ffplay] location model migrated from $LOCATION_MODEL to qwen3.6-plus"
  LOCATION_MODEL="qwen3.6-plus"
fi
SKIP_WIN_STREAM="${SKIP_WIN_STREAM:-0}"
FFPLAY_BIN="${FFPLAY_BIN:-F:\\source\\Go\\ffmpeg-6.1.1-full_build\\ffmpeg-6.1.1-full_build\\bin\\ffplay.exe}"

DEFAULT_WEIGHTS_CANDIDATES=(
  "$ROOT_DIR/visual_recognition/runs/xu/weights/best.pt"
  "$ROOT_DIR/visual_recognition/runs/xu/weights/last.pt"
  "$ROOT_DIR/yolo11n.pt"
)
TRT_ENGINE="${TRT_ENGINE:-$ROOT_DIR/visual_recognition/runs/xu/weights/best.engine}"

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

if [[ "$ACCEL_MODE" == "trt" || "$ACCEL_MODE" == "auto" ]]; then
  if [[ "$WEIGHTS" == *.pt && -f "$TRT_ENGINE" ]]; then
    echo "[yolo_ffplay] accel: use TensorRT engine -> $TRT_ENGINE"
    WEIGHTS="$TRT_ENGINE"
  elif [[ "$ACCEL_MODE" == "trt" && "$WEIGHTS" != *.engine ]]; then
    echo "[yolo_ffplay] accel warning: TensorRT mode requested but engine not found: $TRT_ENGINE" >&2
    echo "[yolo_ffplay] accel fallback: keep PT weights -> $WEIGHTS" >&2
  fi
fi

if [[ "$WEIGHTS" == *.engine && -z "$INFER_EVERY_SET" ]]; then
  INFER_EVERY="3"
  echo "[yolo_ffplay] accel: engine detected, auto set INFER_EVERY=3"
fi

if [[ "$WEIGHTS" == *.engine && -z "$IMGSZ_SET" ]]; then
  IMGSZ="640"
  echo "[yolo_ffplay] accel: engine detected, auto set IMGSZ=640"
fi

if [[ "$WEIGHTS" == *.engine && -z "$WORK_SIZE_SET" ]]; then
  WORK_SIZE="512x288"
  echo "[yolo_ffplay] accel: engine detected, auto set WORK_SIZE=512x288"
fi

if [[ "$WEIGHTS" == *.engine && -z "$OUTPUT_MAX_SIZE_SET" ]]; then
  OUTPUT_MAX_SIZE="448x252"
  echo "[yolo_ffplay] accel: engine detected, auto set OUTPUT_MAX_SIZE=448x252"
fi

if [[ "$WEIGHTS" == *.engine && -z "$FRAME_DRAIN_SET" ]]; then
  FRAME_DRAIN="1"
  echo "[yolo_ffplay] accel: engine detected, auto set FRAME_DRAIN=1"
fi

if [[ "$WEIGHTS" == *.engine && -z "$CAPTURE_DRAIN_SET" ]]; then
  CAPTURE_DRAIN="1"
  echo "[yolo_ffplay] accel: engine detected, auto set CAPTURE_DRAIN=1"
fi

if [[ "$WEIGHTS" == *.engine && -z "$STREAM_FPS_SET" ]]; then
  STREAM_FPS="18"
  echo "[yolo_ffplay] accel: engine detected, auto set STREAM_FPS=18"
fi

if [[ "$WEIGHTS" == *.engine && -z "$OUT_BITRATE_SET" ]]; then
  OUT_BITRATE="2200k"
  echo "[yolo_ffplay] accel: engine detected, auto set OUT_BITRATE=2200k"
fi

if [[ "$WEIGHTS" == *.engine && -z "$BOXES_TTL_MS_SET" ]]; then
  BOXES_TTL_MS="350"
  echo "[yolo_ffplay] accel: engine detected, auto set BOXES_TTL_MS=350"
fi

CMD=(
  "$PYTHON_BIN" "$ROOT_DIR/visual_recognition/stream_ffplay_pipeline.py"
  --game-exe "$GAME_EXE"
  --window-title "$WINDOW_TITLE"
  --wait-game "$WAIT_GAME"
  --weights "$WEIGHTS"
  --in-stream "$IN_STREAM"
  --out-stream "$OUT_STREAM"
  --linux-ip "$LINUX_IP"
  --port "$PORT"
  --framerate "$FRAMERATE"
  --bitrate "$BITRATE"
  --conf "$CONF"
  --imgsz "$IMGSZ"
  --accel-mode "$ACCEL_MODE"
  --device "$DEVICE"
  --name "$RUN_NAME"
  --detect-roi "$DETECT_ROI"
)

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
  if [[ -z "$OCR_INFO_JSONL" ]]; then
    OCR_INFO_JSONL="$ROOT_DIR/visual_recognition/runs/$RUN_NAME/ocr_info.jsonl"
  fi
  CMD+=(--ocr --ocr-engine "$OCR_ENGINE")
  CMD+=(--ocr-lang "$OCR_LANG" --ocr-cn-lang "$OCR_CN_LANG")
  CMD+=(--ocr-every "$OCR_EVERY")
  if [[ -n "$OCR_ROI" ]]; then
    CMD+=(--ocr-roi "$OCR_ROI")
  fi
  CMD+=(--ocr-min-conf "$OCR_MIN_CONF" --ocr-whitelist "$OCR_WHITELIST")
  if [[ -n "$OCR_INFO_JSONL" ]]; then
    CMD+=(--ocr-info-jsonl "$OCR_INFO_JSONL")
  fi
fi
if [[ "$DRAW_OCR_ROI" == "1" ]]; then
  CMD+=(--draw-ocr-roi)
fi
if [[ "$LOCATION_DETECT" == "1" ]]; then
  CMD+=(--location-detect)
  CMD+=(--location-every "$LOCATION_EVERY")
  CMD+=(--location-roi "$LOCATION_ROI")
  CMD+=(--location-model "$LOCATION_MODEL")
fi
if [[ "$SKIP_WIN_STREAM" == "1" ]]; then
  CMD+=(--skip-win-stream)
fi
CMD+=(--frame-drain "$FRAME_DRAIN")
CMD+=(--capture-drain "$CAPTURE_DRAIN")
CMD+=(--work-size "$WORK_SIZE")
CMD+=(--output-max-size "$OUTPUT_MAX_SIZE")
CMD+=(--preview-size "$PREVIEW_SIZE")
CMD+=(--infer-every "$INFER_EVERY")
CMD+=(--boxes-ttl-ms "$BOXES_TTL_MS")
CMD+=(--stream-fps "$STREAM_FPS")
CMD+=(--udp-fifo-size "$UDP_FIFO_SIZE")
CMD+=(--capture-reconnect-sec "$CAPTURE_RECONNECT_SEC")
CMD+=(--capture-timeout-ms "$CAPTURE_TIMEOUT_MS")
CMD+=(--first-frame-timeout-sec "$FIRST_FRAME_TIMEOUT_SEC")
CMD+=(--capture-probesize "$CAPTURE_PROBESIZE")
CMD+=(--capture-analyzeduration "$CAPTURE_ANALYZEDURATION")
CMD+=(--sender-udp-pkt-size "$SENDER_UDP_PKT_SIZE")
CMD+=(--sender-udp-buffer-size "$SENDER_UDP_BUFFER_SIZE")
CMD+=(--out-bitrate "$OUT_BITRATE")
CMD+=(--out-vcodec "$STREAM_VCODEC")
CMD+=(--win-vcodec "$STREAM_VCODEC")
if [[ "$HALF" == "1" ]]; then
  CMD+=(--half)
fi
CMD+=(--ffplay "$FFPLAY_BIN")
if [[ -n "$OUT_STREAM" ]]; then
  CMD+=(--out-stream "$OUT_STREAM")
fi

echo "[yolo_ffplay] Running: ${CMD[*]}"
echo "[yolo_ffplay] Endpoint: IN_STREAM=${IN_STREAM} OUT_STREAM=${OUT_STREAM} LINUX_IP=${LINUX_IP} WINDOWS_IP=${WINDOWS_IP} PORT=${PORT}"
exec "${CMD[@]}"
