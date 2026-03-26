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
VIEWER_MODE="${VIEWER_MODE:-winffplay}"
CAPTURE_BACKEND="${CAPTURE_BACKEND:-gdigrab}"
MONITOR="${MONITOR:-2}"
VIEW_W="${VIEW_W:-800}"
VIEW_H="${VIEW_H:-600}"
ALLOW_DESKTOP_FALLBACK="${ALLOW_DESKTOP_FALLBACK:-0}"
NO_VIEWER="${NO_VIEWER:-0}"
# Windows 侧 ffplay 绝对路径（用于 winffplay 模式）
FFPLAY_BIN="${FFPLAY_BIN:-F:\\source\\Go\\ffmpeg-6.1.1-full_build\\ffmpeg-6.1.1-full_build\\bin\\ffplay.exe}"

# 若用户手动提供 DISPLAY 则尊重；否则保留当前环境值。
# 是否可用取决于 Windows 侧是否运行 X server / WSLg。
DISPLAY_VALUE="${DISPLAY_VALUE:-${DISPLAY:-}}"
if [[ -n "$DISPLAY_VALUE" ]]; then
	export DISPLAY="$DISPLAY_VALUE"
fi

exec "$PYTHON_BIN" "$ROOT_DIR/opengame.py" \
	--game-exe "$GAME_EXE" \
	--game-arg="$GAME_ARG1" \
	--game-arg="$GAME_ARG2" \
	--linux-ip "$LINUX_IP" \
	--port "$PORT" \
	--window-title "$WINDOW_TITLE" \
	--capture-backend "$CAPTURE_BACKEND" \
	--monitor "$MONITOR" \
	--view-width "$VIEW_W" \
	--view-height "$VIEW_H" \
	--viewer-mode "$VIEWER_MODE" \
	--ffplay "$FFPLAY_BIN" \
	$( [[ "$NO_VIEWER" == "1" ]] && echo "--no-viewer" ) \
	$( [[ "$ALLOW_DESKTOP_FALLBACK" == "1" ]] && echo "--allow-desktop-fallback" )