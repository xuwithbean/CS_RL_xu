#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/xu/anaconda3/envs/condacommon/bin/python}"

# 白底点流共享状态，和 trainimg.py / trainimg.sh 保持一致。
SHARED_FRAME_PATH="${SHARED_FRAME_PATH:-${CSRL_SHARED_FRAME_PATH:-/tmp/cs_rl_latest_frame.jpg}}"
SHARED_STATE_PATH="${SHARED_STATE_PATH:-${CSRL_SHARED_STATE_PATH:-/tmp/cs_rl_runtime_state.json}}"

# 训练模型保存路径
SAVE_PATH="${SAVE_PATH:-point_aim_net.pt}"
LOAD_PATH="${LOAD_PATH:-}"

# 动作控制参数
MOVE_GAIN="${MOVE_GAIN:-300}"
MAX_STEP="${MAX_STEP:-400}"
POLL_SEC="${POLL_SEC:-0.03}"
SHOOT_CENTER_ERROR="${SHOOT_CENTER_ERROR:-0.04}"
BATCH_SIZE="${BATCH_SIZE:-64}"
BUFFER_SIZE="${BUFFER_SIZE:-1024}"
LR="${LR:-1e-3}"
HIDDEN_DIM="${HIDDEN_DIM:-64}"
LOG_EVERY="${LOG_EVERY:-50}"
SAVE_EVERY="${SAVE_EVERY:-500}"
DEVICE="${DEVICE:-cpu}"

# 是否只训练不动鼠标：默认 false，实际执行鼠标控制
TRAIN_ONLY="${TRAIN_ONLY:-0}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[point-aim-train] PYTHON_BIN 不存在或不可执行: $PYTHON_BIN" >&2
  exit 1
fi

TRAINIMG_SCRIPT="$ROOT_DIR/trainimg.sh"
TRAINER_SCRIPT="$ROOT_DIR/point_aim_trainer.py"

if [[ ! -f "$TRAINIMG_SCRIPT" ]]; then
  echo "[point-aim-train] 未找到 trainimg.sh: $TRAINIMG_SCRIPT" >&2
  exit 1
fi
if [[ ! -f "$TRAINER_SCRIPT" ]]; then
  echo "[point-aim-train] 未找到 point_aim_trainer.py: $TRAINER_SCRIPT" >&2
  exit 1
fi

echo "[point-aim-train] 启动白底点流: $TRAINIMG_SCRIPT"
echo "[point-aim-train] 共享帧: $SHARED_FRAME_PATH"
echo "[point-aim-train] 共享状态: $SHARED_STATE_PATH"
echo "[point-aim-train] 模型保存: $SAVE_PATH"
echo "[point-aim-train] 实际动鼠标: $([[ "$TRAIN_ONLY" == "1" || "$TRAIN_ONLY" == "true" || "$TRAIN_ONLY" == "TRUE" ]] && echo "否" || echo "是")"

cleanup() {
  if [[ -n "${TRAINIMG_PID:-}" ]]; then
    kill "$TRAINIMG_PID" >/dev/null 2>&1 || true
    wait "$TRAINIMG_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

# 启动白底点流，保持共享状态持续更新
SHARED_FRAME_PATH="$SHARED_FRAME_PATH" SHARED_STATE_PATH="$SHARED_STATE_PATH" bash "$TRAINIMG_SCRIPT" >/tmp/point_aim_trainimg.log 2>&1 &
TRAINIMG_PID=$!

# 给点流一点启动时间
sleep 1

CMD=(
  "$PYTHON_BIN" "$TRAINER_SCRIPT"
  --shared-state "$SHARED_STATE_PATH"
  --save-path "$SAVE_PATH"
  --move-gain "$MOVE_GAIN"
  --max-step "$MAX_STEP"
  --poll-sec "$POLL_SEC"
  --shoot-center-error "$SHOOT_CENTER_ERROR"
  --batch-size "$BATCH_SIZE"
  --buffer-size "$BUFFER_SIZE"
  --lr "$LR"
  --hidden-dim "$HIDDEN_DIM"
  --log-every "$LOG_EVERY"
  --save-every "$SAVE_EVERY"
  --device "$DEVICE"
)

if [[ -n "$LOAD_PATH" ]]; then
  CMD+=(--load-path "$LOAD_PATH")
fi
if [[ "$TRAIN_ONLY" == "1" || "$TRAIN_ONLY" == "true" || "$TRAIN_ONLY" == "TRUE" ]]; then
  CMD+=(--train-only)
fi

echo "[point-aim-train] 启动训练命令:"
echo "${CMD[*]}"

exec "${CMD[@]}"
