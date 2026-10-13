#!/usr/bin/env bash
# Scorer 中間特徴（GAP 後 512 次元）を全スプリットで抽出して parquet に保存する
#
# このスクリプトは例外的に独立ランチャ config を持たない。--config は「学習済み Scorer の
# config.yaml」（モデル再構築用）を指すため。実行設定はコマンドに直書きし、データパス等は
# data.* 上書きで渡す。設定を恒久的に変えたいときはこのコマンドを直接編集する。
#
# Usage:
#   bash scripts/run_extract_scorer_features.sh
#   bash scripts/run_extract_scorer_features.sh --split val   # 単発の上書き
set -euo pipefail

uv run python scripts/extract_scorer_features.py \
  --checkpoint "models/2026-05-29-16-07_scorer_H_honly/checkpoints/last.ckpt" \
  --config "models/2026-05-29-16-07_scorer_H_honly/config.yaml" \
  --split all \
  --batch-size 64 \
  --output-dir "results/socorer_features/2026-05-29-16-07_scorer_H_honly/checkpoints/last" \
  data.real_data_dir=data/datasets/font2dataset_ascii_google \
  data.gen_data_dir=results/2026-05-26_diffusion_H_honly/checkpoints/last \
  data.charset=H \
  data.split_file=config/splits/split_ascii_seed42.yaml \
  "$@"
