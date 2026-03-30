#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/xu/code/CS_RL_xu"
PYTHON_BIN="/home/xu/anaconda3/envs/condacommon/bin/python"

# 可按需改为你的实际路径
GAME_EXE="${GAME_EXE:-E:\\steam\\steamapps\\common\\Counter-Strike Global Offensive\\game\\bin\\win64\\cs2.exe}"
GAME_ARG1="${GAME_ARG1:--applaunch}"
GAME_ARG2="${GAME_ARG2:-730}"

# Linux 接收地址/端口（Windows ffmpeg 会推送到这里）
# 默认自动取 WSL 当前 IP，避免错误地推到 Windows 自己的 127.0.0.1
AUTO_LINUX_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
LINUX_IP="${LINUX_IP:-${AUTO_LINUX_IP:-127.0.0.1}}"
PORT="${PORT:-12345}"

# 捕获窗口标题：默认自动探测游戏窗口标题，失败时再退回桌面采集
WINDOW_TITLE="${WINDOW_TITLE:-auto}"
STREAM_OUTPUT="${STREAM_OUTPUT:-}"
VIEWER_SOURCE="${VIEWER_SOURCE:-}"
VIEW_W="${VIEW_W:-800}"
VIEW_H="${VIEW_H:-450}"
NO_VIEWER="${NO_VIEWER:-0}"
# Windows 侧 ffplay 绝对路径（用于 winffplay 模式）
FFPLAY_BIN="${FFPLAY_BIN:-F:\\source\\Go\\ffmpeg-6.1.1-full_build\\ffmpeg-6.1.1-full_build\\bin\\ffplay.exe}"

args=(
	"$PYTHON_BIN" "$ROOT_DIR/opengame.py"
	--game-exe "$GAME_EXE"
	--game-arg="$GAME_ARG1"
	--game-arg="$GAME_ARG2"
	--linux-ip "$LINUX_IP"
	--port "$PORT"
	--window-title "$WINDOW_TITLE"
	--view-width "$VIEW_W"
	--view-height "$VIEW_H"
	--ffplay "$FFPLAY_BIN"
)

if [[ -n "$STREAM_OUTPUT" ]]; then
	args+=(--stream-output "$STREAM_OUTPUT")
fi
if [[ -n "$VIEWER_SOURCE" ]]; then
	args+=(--viewer-source "$VIEWER_SOURCE")
fi
if [[ "$NO_VIEWER" == "1" ]]; then
	args+=(--no-viewer)
fi

exec "${args[@]}"