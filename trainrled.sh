#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

# 继续训练已有模型：默认开启。
RESUME="${RESUME:-1}"

# 优先使用已存在的主模型；可通过 LOAD_PATH / SAVE_PATH 覆盖。
DEFAULT_MODEL_PATH="$ROOT_DIR/td3_checkpoint.pt"

LOAD_PATH="${LOAD_PATH:-$DEFAULT_MODEL_PATH}"
SAVE_PATH="${SAVE_PATH:-$LOAD_PATH}"
BEST_SAVE_PATH="${BEST_SAVE_PATH:-}"
REWARD_PLOT_PATH="${REWARD_PLOT_PATH:-reward_curve.png}"
BEST_REWARD_PLOT_PATH="${BEST_REWARD_PLOT_PATH:-}"
REWARD_PLOT_EVERY="${REWARD_PLOT_EVERY:-100}"
GAMMA="${GAMMA:-0.99}"
MOVE_GAIN="${MOVE_GAIN:-120.0}"
MAX_STEP="${MAX_STEP:-160}"
STEP_DT_SEC="${STEP_DT_SEC:-0.03}"
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

# 透传到 trainsl.sh 的环境变量。
export RESUME
export LOAD_PATH
export SAVE_PATH
export BEST_SAVE_PATH
export REWARD_PLOT_PATH
export BEST_REWARD_PLOT_PATH
export REWARD_PLOT_EVERY
export GAMMA
export MOVE_GAIN
export MAX_STEP
export STEP_DT_SEC
export BATCH_SIZE
export REPLAY_SIZE
export START_STEPS
export UPDATES_PER_STEP
export POLICY_NOISE
export NOISE_CLIP
export POLICY_DELAY
export TAU
export EXPLORATION_NOISE
export SHOOT_THRESHOLD
export SHOOT_CENTER_ERROR
export CHECKPOINT_EVERY

echo "[trainrled] resume=$RESUME"
echo "[trainrled] load_path=$LOAD_PATH"
echo "[trainrled] save_path=$SAVE_PATH"
echo "[trainrled] reward_plot_path=$REWARD_PLOT_PATH"
echo "[trainrled] reward_plot_every=$REWARD_PLOT_EVERY"
echo "[trainrled] gamma=$GAMMA"
echo "[trainrled] mouse move gain=$MOVE_GAIN"
echo "[trainrled] mouse max step=$MAX_STEP"
echo "[trainrled] step_dt_sec=$STEP_DT_SEC"
echo "[trainrled] batch_size=$BATCH_SIZE replay_size=$REPLAY_SIZE start_steps=$START_STEPS"
echo "[trainrled] updates_per_step=$UPDATES_PER_STEP policy_noise=$POLICY_NOISE noise_clip=$NOISE_CLIP policy_delay=$POLICY_DELAY"
echo "[trainrled] tau=$TAU exploration_noise=$EXPLORATION_NOISE"
echo "[trainrled] shoot_threshold=$SHOOT_THRESHOLD shoot_center_error=$SHOOT_CENTER_ERROR checkpoint_every=$CHECKPOINT_EVERY"

exec "$ROOT_DIR/trainsl.sh" "$@"
