#!/usr/bin/env bash
# データセット内の全画像に realness_score を付与
#
# 設定の真実は config/eval/score_images.yaml。設定を変えたいときは YAML を直接編集する。
#
# Usage:
#   bash scripts/run_score_images.sh
#   bash scripts/run_score_images.sh overwrite=true batch_size=256  # 単発の上書き
set -euo pipefail

uv run python scripts/score_images.py --config config/eval/score_images.yaml "$@"
