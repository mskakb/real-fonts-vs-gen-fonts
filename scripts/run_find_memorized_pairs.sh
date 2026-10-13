#!/usr/bin/env bash
# 生成画像の記憶（memorization）検証: 生成画像ごとの最近傍実フォント画像を探す
#
# 設定の真実は config/eval/find_memorized_pairs.yaml。設定を変えたいときは YAML を直接編集する。
#
# Usage:
#   bash scripts/run_find_memorized_pairs.sh
#   bash scripts/run_find_memorized_pairs.sh charset=H top_k=32  # 単発の上書き
set -euo pipefail

uv run python scripts/find_memorized_pairs.py --config config/eval/find_memorized_pairs.yaml "$@"
