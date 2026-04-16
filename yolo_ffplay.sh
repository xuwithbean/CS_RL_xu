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
FRAMERATE="${FRAMERATE:-60}"
BITRATE="${BITRATE:-2500k}"
AUTO_WINDOWS_IP="$(detect_windows_ip)"
WINDOWS_IP="${WINDOWS_IP:-${AUTO_WINDOWS_IP:-127.0.0.1}}"
if [[ -z "$OUT_STREAM" ]]; then
  OUT_STREAM="udp://${WINDOWS_IP}:2234"
fi

# YOLO 参数
WEIGHTS="${WEIGHTS:-}"
CONF="${CONF:-0.30}"
IMGSZ="${IMGSZ:-256}"
FRAME_DRAIN="${FRAME_DRAIN:-0}"
CAPTURE_DRAIN="${CAPTURE_DRAIN:-0}"
WORK_SIZE="${WORK_SIZE:-704x396}"
PREVIEW_SIZE="${PREVIEW_SIZE:-800x450}"
INFER_EVERY="${INFER_EVERY:-1}"
HALF="${HALF:-1}"
DEVICE="${DEVICE:-0}"
UDP_FIFO_SIZE="${UDP_FIFO_SIZE:-1048576}"
CAPTURE_RECONNECT_SEC="${CAPTURE_RECONNECT_SEC:-3.0}"
CAPTURE_TIMEOUT_MS="${CAPTURE_TIMEOUT_MS:-5000}"
FIRST_FRAME_TIMEOUT_SEC="${FIRST_FRAME_TIMEOUT_SEC:-20.0}"
CAPTURE_PROBESIZE="${CAPTURE_PROBESIZE:-32768}"
CAPTURE_ANALYZEDURATION="${CAPTURE_ANALYZEDURATION:-1000000}"
SENDER_UDP_PKT_SIZE="${SENDER_UDP_PKT_SIZE:-1316}"
SENDER_UDP_BUFFER_SIZE="${SENDER_UDP_BUFFER_SIZE:-1048576}"
STREAM_VCODEC="${STREAM_VCODEC:-mpeg1video}"

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
OCR="${OCR:-0}"
OCR_ENGINE="${OCR_ENGINE:-pytesseract}"
OCR_ROI="${OCR_ROI:-}"
OCR_MIN_CONF="${OCR_MIN_CONF:-0.20}"
OCR_WHITELIST="${OCR_WHITELIST:-0123456789/%:HPARMOABULLET}"
OCR_INFO_JSONL="${OCR_INFO_JSONL:-}"
SKIP_WIN_STREAM="${SKIP_WIN_STREAM:-0}"
FFPLAY_BIN="${FFPLAY_BIN:-F:\\source\\Go\\ffmpeg-6.1.1-full_build\\ffmpeg-6.1.1-full_build\\bin\\ffplay.exe}"

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
  --linux-ip "$LINUX_IP"
  --port "$PORT"
  --framerate "$FRAMERATE"
  --bitrate "$BITRATE"
  --conf "$CONF"
  --imgsz "$IMGSZ"
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
CMD+=(--capture-drain "$CAPTURE_DRAIN")
CMD+=(--work-size "$WORK_SIZE")
CMD+=(--preview-size "$PREVIEW_SIZE")
CMD+=(--infer-every "$INFER_EVERY")
CMD+=(--udp-fifo-size "$UDP_FIFO_SIZE")
CMD+=(--capture-reconnect-sec "$CAPTURE_RECONNECT_SEC")
CMD+=(--capture-timeout-ms "$CAPTURE_TIMEOUT_MS")
CMD+=(--first-frame-timeout-sec "$FIRST_FRAME_TIMEOUT_SEC")
CMD+=(--capture-probesize "$CAPTURE_PROBESIZE")
CMD+=(--capture-analyzeduration "$CAPTURE_ANALYZEDURATION")
CMD+=(--sender-udp-pkt-size "$SENDER_UDP_PKT_SIZE")
CMD+=(--sender-udp-buffer-size "$SENDER_UDP_BUFFER_SIZE")
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
