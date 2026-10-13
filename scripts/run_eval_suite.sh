#!/usr/bin/env bash
# 複数 Scorer チェックポイント横断で評価・解析スイート一式（feature 抽出/特徴解析/
# 周波数解析/occlusion 解析/per-char 可視化/attention 忠実性検証）を一括実行する。
#
# 設定の真実は config/eval/eval_suite.yaml（チェックポイントのリスト + ステップ別パラメータ）。
# --checkpoints/--steps で対象を絞り込み、--stop-on-error でエラー時に即中断できる。
#
# Usage:
#   bash scripts/run_eval_suite.sh
#   bash scripts/run_eval_suite.sh --checkpoints ranking --steps extract_features,analyze_features
#   bash scripts/run_eval_suite.sh --stop-on-error
set -euo pipefail

uv run python scripts/run_eval_suite.py --config config/eval/eval_suite.yaml "$@"
