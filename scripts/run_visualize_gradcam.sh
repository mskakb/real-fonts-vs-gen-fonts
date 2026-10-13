#!/usr/bin/env bash
# Grad-CAM による Scorer の判断根拠可視化
#
# このスクリプトは例外的に独立ランチャ config を持たない。--config は「学習済み Scorer の
# config.yaml」（データ/SDF kwarg 再現用）を指すため。実行設定はコマンドに直書きし、単発の
# 上書きは "$@"（--layer/--n-samples/data.* 等）で渡す。恒久的に変えるときはこのコマンドを編集する。
#
# Usage:
#   bash scripts/run_visualize_gradcam.sh
#   bash scripts/run_visualize_gradcam.sh --layer layer2   # 単発の上書き
set -euo pipefail

MODEL="models/2026-06-19-21-23_scorer_classifier_uppercase"

uv run python scripts/visualize_gradcam.py \
  --checkpoint "${MODEL}/checkpoints/last.ckpt" \
  --config     "${MODEL}/config.yaml" \
  --output-dir "${MODEL}/grad_cam" \
  --split test \
  --n-samples 100 \
  "$@"
