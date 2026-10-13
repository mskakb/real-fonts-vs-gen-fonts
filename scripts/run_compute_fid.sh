#!/usr/bin/env bash
# FID 計算: 実フォント vs 生成フォント
#
# 設定の真実は config/eval/compute_fid.yaml。設定を変えたいときは YAML を直接編集する。
#
# Usage:
#   bash scripts/run_compute_fid.sh
#   bash scripts/run_compute_fid.sh split=val batch_size=32  # 単発の上書き
set -euo pipefail

uv run python scripts/compute_fid.py --config config/eval/compute_fid.yaml "$@"
