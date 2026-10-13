#!/usr/bin/env bash
# analyze_occlusion.py: per-patch 絶対スコア掃引 + 遮蔽バリアントの特徴空間写像
#
# このスクリプトは例外的に独立ランチャ config を持たない。--config は「学習済み Scorer の
# config.yaml」（データ/SDF kwarg 再現用）を指すため。--features は同一 checkpoint 由来の
# extract_scorer_features.py 出力 parquet（母集団 real/gen 特徴）を指す。実行設定はコマンドに
# 直書きし、単発の上書きは "$@"（--char/--label/--n-images/--patch/--stride/--embedding-layer/
# --reducer 等）で渡す。恒久的に変えるときはこのコマンドを編集する。
#
# Usage:
#   bash scripts/run_analyze_occlusion.sh --char H --label real
#   bash scripts/run_analyze_occlusion.sh --char A --label gen --n-images 3 --embedding-layer layer2
set -euo pipefail

MODEL="models/2026-06-19-21-23_scorer_classifier_uppercase"
CKPT="checkpoints/last.ckpt"
FEATURES="features_ema/scorer_features.parquet"
N=10

uv run python scripts/analyze_occlusion.py \
  --checkpoint "${MODEL}/${CKPT}" \
  --config     "${MODEL}/config.yaml" \
  --features   "${MODEL}/${FEATURES}" \
  --output-dir "${MODEL}/occlusion_analysis" \
  --split test \
  --reducer tsne \
  --n-images $N \
  "$@"
