#!/usr/bin/env bash
conda activate condacommon
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
VIEWER_MODE="${VIEWER_MODE:-web}"
WEB_PORT="${WEB_PORT:-18080}"

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
	--viewer-mode "$VIEWER_MODE" \
	--web-port "$WEB_PORT"