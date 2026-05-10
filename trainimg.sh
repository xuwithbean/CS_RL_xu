#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/xu/anaconda3/envs/condacommon/bin/python}"

# 共享数据路径与 `visual_recognition/stream_ffplay_pipeline.py` 保持一致。
SHARED_FRAME_PATH="${SHARED_FRAME_PATH:-${CSRL_SHARED_FRAME_PATH:-/tmp/cs_rl_latest_frame.jpg}}"
SHARED_STATE_PATH="${SHARED_STATE_PATH:-${CSRL_SHARED_STATE_PATH:-/tmp/cs_rl_runtime_state.json}}"

# 默认输出到 Windows 主机 UDP 端口，便于在 Windows 上直接用 ffplay 查看。
OUT_STREAM="${OUT_STREAM:-}"
FPS="${FPS:-20}"
POINT_RADIUS="${POINT_RADIUS:-4}"
OUT_BITRATE="${OUT_BITRATE:-800k}"
OUT_VCODEC="${OUT_VCODEC:-mpeg2video}"
FFMPEG_BIN="${FFMPEG_BIN:-ffmpeg}"
AUTO_START_WINDOWS_FFPLAY="${AUTO_START_WINDOWS_FFPLAY:-1}"
FFPLAY_BIN_WIN="${FFPLAY_BIN_WIN:-}"

PREVIEW_PORT="${PREVIEW_PORT:-23001}"
WINDOWS_PLAY_URL="udp://0.0.0.0:${PREVIEW_PORT}?fifo_size=50000000&overrun_nonfatal=1"

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
  if command -v powershell.exe >/dev/null 2>&1; then
    ps_ips="$(powershell.exe -NoProfile -Command "Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue | Where-Object { $_.IPAddress -and $_.IPAddress -notlike '127.*' -and $_.IPAddress -notlike '169.254.*' } | Select-Object -ExpandProperty IPAddress" 2>/dev/null | tr -d '\r' || true)"
  fi

  [[ -n "$ns_ip" ]] && candidates+=("$ns_ip")
  [[ -n "$gw_ip" ]] && candidates+=("$gw_ip")
  while IFS= read -r ip; do
    [[ -n "$ip" ]] && candidates+=("$ip")
  done <<< "$ps_ips"

  local -A seen=()
  for ip in "${candidates[@]}"; do
    [[ -n "${seen[$ip]:-}" ]] && continue
    seen["$ip"]=1
    if is_bad_windows_ip "$ip"; then
      echo "[trainimg] Windows IP check: skip candidate=$ip" >&2
      continue
    fi
    if check_ip_reachable "$ip"; then
      echo "[trainimg] Windows IP check: selected reachable candidate=$ip" >&2
      echo "$ip"
      return 0
    fi
    echo "[trainimg] Windows IP check: candidate not reachable=$ip" >&2
  done

  for ip in "${candidates[@]}"; do
    if ! is_bad_windows_ip "$ip"; then
      echo "[trainimg] Windows IP check: fallback candidate=$ip" >&2
      echo "$ip"
      return 0
    fi
  done

  echo "127.0.0.1"
  return 0
}

get_windows_ffplay_path() {
  if [[ -n "$FFPLAY_BIN_WIN" ]]; then
    echo "$FFPLAY_BIN_WIN"
    return 0
  fi

  if ! command -v powershell.exe >/dev/null 2>&1; then
    return 1
  fi

  local found=""
  found="$(powershell.exe -NoProfile -Command "(Get-Command ffplay.exe -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty Source)" 2>/dev/null | tr -d '\r' || true)"
  if [[ -n "$found" ]]; then
    echo "$found"
    return 0
  fi

  return 1
}

start_windows_ffplay() {
  local ffplay_path="$1"
  if ! command -v powershell.exe >/dev/null 2>&1; then
    echo "[trainimg] 未找到 powershell.exe，跳过自动启动 Windows ffplay。" >&2
    return 1
  fi

  powershell.exe -NoProfile -Command "Start-Process -FilePath '$ffplay_path' -ArgumentList '-f','mpegts','-fflags','+discardcorrupt','-framedrop','-strict','-1','-probesize','262144','-analyzeduration','100000','-sync','ext','-i','$WINDOWS_PLAY_URL'" >/dev/null 2>&1 || return 1
  return 0
}

# 仅在共享帧不可用时才会用到的兜底尺寸。
WIDTH="${WIDTH:-0}"
HEIGHT="${HEIGHT:-0}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[trainimg] PYTHON_BIN 不存在或不可执行: $PYTHON_BIN" >&2
  exit 1
fi

CMD=(
  "$PYTHON_BIN" "$ROOT_DIR/trainimg.py"
  --shared-frame-path "$SHARED_FRAME_PATH"
  --shared-state-path "$SHARED_STATE_PATH"
  --fps "$FPS"
  --point-radius "$POINT_RADIUS"
  --ffmpeg "$FFMPEG_BIN"
  --out-bitrate "$OUT_BITRATE"
  --out-vcodec "$OUT_VCODEC"
)

if [[ -z "$OUT_STREAM" ]]; then
  WIN_IP="$(detect_windows_ip)"
  if [[ -z "$WIN_IP" || "$WIN_IP" == "127.0.0.1" ]]; then
    echo "[trainimg] 无法推断 Windows IP，请手动设置 OUT_STREAM。" >&2
    exit 1
  fi
  OUT_STREAM="udp://${WIN_IP}:${PREVIEW_PORT}?pkt_size=1316&buffer_size=262144"
fi

if [[ "$WIDTH" != "0" ]]; then
  CMD+=(--width "$WIDTH")
fi
if [[ "$HEIGHT" != "0" ]]; then
  CMD+=(--height "$HEIGHT")
fi

CMD+=(--out-stream "$OUT_STREAM")

FFPLAY_HINT="ffplay -f mpegts -fflags +discardcorrupt -framedrop -strict -1 -probesize 262144 -analyzeduration 100000 -sync ext -i ${WINDOWS_PLAY_URL}"

if [[ "$AUTO_START_WINDOWS_FFPLAY" == "1" || "$AUTO_START_WINDOWS_FFPLAY" == "true" || "$AUTO_START_WINDOWS_FFPLAY" == "TRUE" ]]; then
  FFPLAY_PATH="$(get_windows_ffplay_path || true)"
  if [[ -n "$FFPLAY_PATH" ]]; then
    if start_windows_ffplay "$FFPLAY_PATH"; then
      echo "[trainimg] 已尝试自动启动 Windows ffplay: $FFPLAY_PATH"
    else
      echo "[trainimg] 自动启动 Windows ffplay 失败，请手动运行: $FFPLAY_HINT" >&2
    fi
  else
    echo "[trainimg] 未检测到 Windows ffplay.exe，请手动运行: $FFPLAY_HINT" >&2
  fi
fi

echo "[trainimg] 启动命令:"
echo "${CMD[*]}"
echo "[trainimg] 共享帧路径: $SHARED_FRAME_PATH"
echo "[trainimg] 共享状态路径: $SHARED_STATE_PATH"
echo "[trainimg] 模式: windows-stream-only"
echo "[trainimg] 输出流: $OUT_STREAM"
echo "[trainimg] 在 Windows 上运行: $FFPLAY_HINT"

exec "${CMD[@]}"
