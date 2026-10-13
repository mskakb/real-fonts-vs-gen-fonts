#!/usr/bin/env bash
# E2: スコアランドスケープ（s(x̂_0) vs. t の可視化）
#
# 設定の真実は config/eval/score_landscape.yaml。設定を変えたいときは YAML を直接編集する。
#
# Usage:
#   bash scripts/run_score_landscape.sh
#   bash scripts/run_score_landscape.sh n_samples=32 cond_scales=[1.0,6.0]  # 単発の上書き
set -euo pipefail

uv run python scripts/score_landscape.py --config config/eval/score_landscape.yaml "$@"
