#!/usr/bin/env bash
# 実画像の forward→reverse diffusion ラウンドトリップ検証（verify_roundtrip.py）
#
# 設定の真実は config/eval/verify_roundtrip.yaml。設定を変えたいときは YAML を直接編集する。
#
# Usage:
#   bash scripts/run_verify_roundtrip.sh
#   bash scripts/run_verify_roundtrip.sh noise_steps=[100,500] ddim_reverse_steps=[20]  # 単発の上書き
set -euo pipefail

uv run python scripts/verify_roundtrip.py --config config/eval/verify_roundtrip.yaml "$@"
