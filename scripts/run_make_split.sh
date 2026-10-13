#!/usr/bin/env bash
# 実フォントの train/val/test スプリットを計算して保存する
#
# 設定の真実は config/splits/make_split.yaml。設定を変えたいときは YAML を直接編集する。
# （output パスは charset/seed に合わせて YAML 側で更新すること）
#
# Usage:
#   bash scripts/run_make_split.sh
#   bash scripts/run_make_split.sh seed=43 output=config/splits/split_ascii_seed43.yaml  # 単発の上書き
set -euo pipefail

uv run python scripts/make_split.py --config config/splits/make_split.yaml "$@"
