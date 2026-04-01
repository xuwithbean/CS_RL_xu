#!/usr/bin/env bash
set -euo pipefail

# YOLO training launcher for CT/T detection.
# Usage:
#   bash yolorun.sh
# Optional overrides:
#   PYTHON_BIN=/home/xu/anaconda3/envs/condacommon/bin/python \
#   DATA=visual_recognition/data_ct_t.yaml \
#   MODEL=yolo11n.pt \
#   EPOCHS=100 \
#   IMGSZ=640 \
#   BATCH=16 \
#   DEVICE=0 \
#   PROJECT=visual_recognition/runs \
#   NAME=ct_t_yolo \
#   WORKERS=4 \
#   PATIENCE=50 \
#   SEED=42 \
#   CACHE=false \
#   AMP=0 \
#   EXIST_OK=0 \
#   FORCE_PIL_IMREAD=0 \
#   bash yolorun.sh

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/xu/anaconda3/envs/condacommon/bin/python}"

DATA="${DATA:-visual_recognition/data_ct_t.yaml}"
MODEL="${MODEL:-yolo11n.pt}"
EPOCHS="${EPOCHS:-100}"
IMGSZ="${IMGSZ:-640}"
BATCH="${BATCH:-16}"
DEVICE="${DEVICE:-0}"
PROJECT="${PROJECT:-visual_recognition/runs}"
NAME="${NAME:-ct_t_yolo}"
WORKERS="${WORKERS:-4}"
PATIENCE="${PATIENCE:-50}"
SEED="${SEED:-42}"
CACHE="${CACHE:-false}"

# Switch-like options: 1 enables, 0 disables.
AMP="${AMP:-0}"
EXIST_OK="${EXIST_OK:-0}"
FORCE_PIL_IMREAD="${FORCE_PIL_IMREAD:-0}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[yolorun] PYTHON_BIN not found or not executable: $PYTHON_BIN" >&2
  echo "[yolorun] Hint: set PYTHON_BIN or activate conda env first." >&2
  exit 1
fi

if [[ ! -f "$ROOT_DIR/$DATA" ]]; then
  echo "[yolorun] DATA yaml not found: $ROOT_DIR/$DATA" >&2
  exit 1
fi

# If the default full dataset is missing, fallback to smoke dataset automatically.
DEFAULT_FULL_DATA="visual_recognition/data_ct_t.yaml"
DEFAULT_SMOKE_DATA="visual_recognition/data_ct_t_smoke.yaml"
FULL_VAL_DIR="$ROOT_DIR/visual_recognition/datasets/cs_ct_t/images/val"

if [[ "$DATA" == "$DEFAULT_FULL_DATA" ]] && [[ ! -d "$FULL_VAL_DIR" ]]; then
  if [[ -f "$ROOT_DIR/$DEFAULT_SMOKE_DATA" ]]; then
    echo "[yolorun] Full dataset not found: $FULL_VAL_DIR" >&2
    echo "[yolorun] Fallback to smoke dataset: $DEFAULT_SMOKE_DATA" >&2
    DATA="$DEFAULT_SMOKE_DATA"
  else
    echo "[yolorun] Full dataset missing and smoke yaml not found." >&2
    echo "[yolorun] Please prepare datasets/cs_ct_t or set DATA=<your_yaml>." >&2
    exit 1
  fi
fi

CMD=(
  "$PYTHON_BIN" "$ROOT_DIR/visual_recognition/train.py"
  --data "$DATA"
  --model "$MODEL"
  --epochs "$EPOCHS"
  --imgsz "$IMGSZ"
  --batch "$BATCH"
  --device "$DEVICE"
  --project "$PROJECT"
  --name "$NAME"
  --workers "$WORKERS"
  --patience "$PATIENCE"
  --seed "$SEED"
  --cache "$CACHE"
)

if [[ "$AMP" == "1" ]]; then
  CMD+=(--amp)
fi

if [[ "$EXIST_OK" == "1" ]]; then
  CMD+=(--exist-ok)
fi

if [[ "$FORCE_PIL_IMREAD" == "1" ]]; then
  CMD+=(--force-pil-imread)
fi

echo "[yolorun] Running: ${CMD[*]}"
exec "${CMD[@]}"
