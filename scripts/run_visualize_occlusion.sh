#!/usr/bin/env bash
# Occlusion sensitivity による Scorer の判断根拠可視化
#
# このスクリプトは例外的に独立ランチャ config を持たない。--config は「学習済み Scorer の
# config.yaml」（データ/SDF kwarg 再現用）を指すため。実行設定はコマンドに直書きし、単発の
# 上書きは "$@"（--patch/--stride/--baseline/data.* 等）で渡す。恒久的に変えるときはこのコマンドを編集する。
#
# Usage:
#   bash scripts/run_visualize_occlusion.sh
#   bash scripts/run_visualize_occlusion.sh --patch 12 --stride 6   # 単発の上書き
set -euo pipefail

MODEL="models/2026-06-20-23-41_scorer_uppercase"
CKPT="checkpoints/step148820-auroc0.8812.ckpt"

uv run python scripts/visualize_occlusion.py \
  --checkpoint "${MODEL}/${CKPT}" \
  --config     "${MODEL}/config.yaml" \
  --output-dir "/${MODEL}/occlusion" \
  --split test \
  --n-samples 100 \
  "$@"
