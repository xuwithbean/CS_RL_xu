#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/xu/anaconda3/envs/condacommon/bin/python}"

ENV_MODE="${ENV_MODE:-shared}"
EPISODES="${EPISODES:-200}"
MAX_STEPS="${MAX_STEPS:-200}"
MANAGER_INTERVAL="${MANAGER_INTERVAL:-10}"
TARGET_DISAPPEAR_SEC="${TARGET_DISAPPEAR_SEC:-1.5}"
STEP_DT_SEC="${STEP_DT_SEC:-0.03}"
GAMMA="${GAMMA:-0.99}"
SAVE_PATH="${SAVE_PATH:-${MODEL_PATH:-$ROOT_DIR/td3_checkpoint.pt}}"
LOAD_PATH="${LOAD_PATH:-$SAVE_PATH}"
RESUME="${RESUME:-1}"
APPLY_ACTIONS="${APPLY_ACTIONS:-1}"
BEST_SAVE_PATH="${BEST_SAVE_PATH:-}"
REWARD_PLOT_PATH="${REWARD_PLOT_PATH:-reward_curve.png}"
BEST_REWARD_PLOT_PATH="${BEST_REWARD_PLOT_PATH:-}"
REWARD_PLOT_EVERY="${REWARD_PLOT_EVERY:-100}"
MOVE_GAIN="${MOVE_GAIN:-1000.0}"
MAX_STEP="${MAX_STEP:-1000}"
BATCH_SIZE="${BATCH_SIZE:-128}"
REPLAY_SIZE="${REPLAY_SIZE:-50000}"
START_STEPS="${START_STEPS:-400}"
UPDATES_PER_STEP="${UPDATES_PER_STEP:-1}"
POLICY_NOISE="${POLICY_NOISE:-0.20}"
NOISE_CLIP="${NOISE_CLIP:-0.50}"
POLICY_DELAY="${POLICY_DELAY:-2}"
TAU="${TAU:-0.005}"
EXPLORATION_NOISE="${EXPLORATION_NOISE:-0.15}"
SHOOT_THRESHOLD="${SHOOT_THRESHOLD:-0.35}"
SHOOT_CENTER_ERROR="${SHOOT_CENTER_ERROR:-0.055}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-10}"
SHARED_FRAME_PATH="${SHARED_FRAME_PATH:-${CSRL_SHARED_FRAME_PATH:-/tmp/cs_rl_latest_frame.jpg}}"
SHARED_STATE_PATH="${SHARED_STATE_PATH:-${CSRL_SHARED_STATE_PATH:-/tmp/cs_rl_runtime_state.json}}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[trainsl] PYTHON_BIN 不存在或不可执行: $PYTHON_BIN" >&2
  exit 1
fi

CMD=(
  "$PYTHON_BIN" "$ROOT_DIR/train.py"
  --env-mode "$ENV_MODE"
  --episodes "$EPISODES"
  --max-steps "$MAX_STEPS"
  --manager-interval "$MANAGER_INTERVAL"
  --shared-state-path "$SHARED_STATE_PATH"
  --shared-frame-path "$SHARED_FRAME_PATH"
  --target-disappear-sec "$TARGET_DISAPPEAR_SEC"
  --step-dt-sec "$STEP_DT_SEC"
  --gamma "$GAMMA"
  --save-path "$SAVE_PATH"
  --best-save-path "$BEST_SAVE_PATH"
  --reward-plot-path "$REWARD_PLOT_PATH"
  --best-reward-plot-path "$BEST_REWARD_PLOT_PATH"
  --reward-plot-every "$REWARD_PLOT_EVERY"
  --move-gain "$MOVE_GAIN"
  --max-step "$MAX_STEP"
  --batch-size "$BATCH_SIZE"
  --replay-size "$REPLAY_SIZE"
  --start-steps "$START_STEPS"
  --updates-per-step "$UPDATES_PER_STEP"
  --policy-noise "$POLICY_NOISE"
  --noise-clip "$NOISE_CLIP"
  --policy-delay "$POLICY_DELAY"
  --tau "$TAU"
  --exploration-noise "$EXPLORATION_NOISE"
  --shoot-threshold "$SHOOT_THRESHOLD"
  --shoot-center-error "$SHOOT_CENTER_ERROR"
  --checkpoint-every "$CHECKPOINT_EVERY"
)

if [[ "$RESUME" == "1" || "$RESUME" == "true" || "$RESUME" == "TRUE" ]]; then
  CMD+=(--resume)
  CMD+=(--load-path "$LOAD_PATH")
fi

if [[ "$APPLY_ACTIONS" == "1" || "$APPLY_ACTIONS" == "true" || "$APPLY_ACTIONS" == "TRUE" ]]; then
  CMD+=(--apply-actions)
fi

if [[ "$#" -gt 0 ]]; then
  CMD+=("$@")
fi

echo "[trainsl] 启动命令:"
echo "${CMD[*]}"
echo "[trainsl] 模型保存路径: $SAVE_PATH"
echo "[trainsl] 加载路径: $LOAD_PATH"
echo "[trainsl] reward图路径: $REWARD_PLOT_PATH"
echo "[trainsl] best模型路径: ${BEST_SAVE_PATH:-<auto>}"
echo "[trainsl] best reward图路径: ${BEST_REWARD_PLOT_PATH:-<auto>}"
echo "[trainsl] reward更新间隔: $REWARD_PLOT_EVERY"
echo "[trainsl] gamma: $GAMMA"
echo "[trainsl] mouse move gain: $MOVE_GAIN"
echo "[trainsl] mouse max step: $MAX_STEP"
echo "[trainsl] batch_size: $BATCH_SIZE replay_size: $REPLAY_SIZE start_steps: $START_STEPS"
echo "[trainsl] updates_per_step: $UPDATES_PER_STEP policy_noise: $POLICY_NOISE noise_clip: $NOISE_CLIP policy_delay: $POLICY_DELAY"
echo "[trainsl] tau: $TAU exploration_noise: $EXPLORATION_NOISE"
echo "[trainsl] shoot_threshold: $SHOOT_THRESHOLD shoot_center_error: $SHOOT_CENTER_ERROR checkpoint_every: $CHECKPOINT_EVERY"
echo "[trainsl] 模式: $ENV_MODE"
echo "[trainsl] 说明: 按 Ctrl+C 可中断并保存当前模型，重新运行时可继续训练。"

exec "${CMD[@]}"
