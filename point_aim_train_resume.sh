#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/xu/anaconda3/envs/condacommon/bin/python}"

# 白底点流共享状态，和 trainimg.py / trainimg.sh 保持一致。
SHARED_FRAME_PATH="${SHARED_FRAME_PATH:-${CSRL_SHARED_FRAME_PATH:-/tmp/cs_rl_latest_frame.jpg}}"
SHARED_STATE_PATH="${SHARED_STATE_PATH:-${CSRL_SHARED_STATE_PATH:-/tmp/cs_rl_runtime_state.json}}"

# 从已有模型继续训练：默认加载当前模型文件，也可以手动指定 LOAD_PATH。
SAVE_PATH="${SAVE_PATH:-point_aim_net_resume.pt}"
BEST_SAVE_PATH="${BEST_SAVE_PATH:-}"
LOAD_PATH="${LOAD_PATH:-$SAVE_PATH}"

# 动作控制参数
MOVE_GAIN="${MOVE_GAIN:-300}"
POLL_SEC="${POLL_SEC:-0.03}"
SHOOT_CENTER_ERROR="${SHOOT_CENTER_ERROR:-0.04}"
BATCH_SIZE="${BATCH_SIZE:-64}"
BUFFER_SIZE="${BUFFER_SIZE:-1024}"
ACTOR_LR="${ACTOR_LR:-1e-3}"
CRITIC_LR="${CRITIC_LR:-1e-3}"
HIDDEN_DIM="${HIDDEN_DIM:-64}"
LOG_EVERY="${LOG_EVERY:-50}"
SAVE_EVERY="${SAVE_EVERY:-500}"
DEVICE="${DEVICE:-cuda}"

# 是否只训练不动鼠标：默认 false，实际执行鼠标控制
TRAIN_ONLY="${TRAIN_ONLY:-0}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[point-aim-resume] PYTHON_BIN 不存在或不可执行: $PYTHON_BIN" >&2
  exit 1
fi

TRAINER_SCRIPT="$ROOT_DIR/point_aim_trainer.py"

if [[ ! -f "$TRAINER_SCRIPT" ]]; then
  echo "[point-aim-resume] 未找到 point_aim_trainer.py: $TRAINER_SCRIPT" >&2
  exit 1
fi

echo "[point-aim-resume] 使用外部启动的白底点流"
echo "[point-aim-resume] 共享帧: $SHARED_FRAME_PATH"
echo "[point-aim-resume] 共享状态: $SHARED_STATE_PATH"
echo "[point-aim-resume] 加载模型: $LOAD_PATH"
echo "[point-aim-resume] 当前保存: $SAVE_PATH"
echo "[point-aim-resume] 训练设备: $DEVICE"
if [[ -n "$BEST_SAVE_PATH" ]]; then
  echo "[point-aim-resume] 最好模型保存: $BEST_SAVE_PATH"
fi
echo "[point-aim-resume] 实际动鼠标: $([[ "$TRAIN_ONLY" == "1" || "$TRAIN_ONLY" == "true" || "$TRAIN_ONLY" == "TRUE" ]] && echo "否" || echo "是")"

if [[ ! -f "$SHARED_STATE_PATH" ]]; then
  echo "[point-aim-resume] 未找到共享状态文件：$SHARED_STATE_PATH" >&2
  echo "[point-aim-resume] 请先手动启动 trainimg.py / trainimg.sh 生成点视频流。" >&2
  exit 1
fi

CMD=(
  "$PYTHON_BIN" "$TRAINER_SCRIPT"
  --shared-state "$SHARED_STATE_PATH"
  --save-path "$SAVE_PATH"
  --move-gain "$MOVE_GAIN"
  --poll-sec "$POLL_SEC"
  --shoot-center-error "$SHOOT_CENTER_ERROR"
  --batch-size "$BATCH_SIZE"
  --buffer-size "$BUFFER_SIZE"
  --actor-lr "$ACTOR_LR"
  --critic-lr "$CRITIC_LR"
  --hidden-dim "$HIDDEN_DIM"
  --log-every "$LOG_EVERY"
  --save-every "$SAVE_EVERY"
  --device "$DEVICE"
)

if [[ -n "$BEST_SAVE_PATH" ]]; then
  CMD+=(--best-save-path "$BEST_SAVE_PATH")
fi
if [[ -n "$LOAD_PATH" ]]; then
  CMD+=(--load-path "$LOAD_PATH")
fi
if [[ "$TRAIN_ONLY" == "1" || "$TRAIN_ONLY" == "true" || "$TRAIN_ONLY" == "TRUE" ]]; then
  CMD+=(--train-only)
fi

echo "[point-aim-resume] 启动训练命令:"
echo "${CMD[*]}"

exec "${CMD[@]}"
