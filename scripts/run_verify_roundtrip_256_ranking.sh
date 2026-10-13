#!/usr/bin/env bash
# 256px 拡散モデル × ranking Scorer のラウンドトリップ検証（verify_roundtrip.py）
#
# 設定の真実は config/eval/verify_roundtrip_256_ranking.yaml。設定を変えたいときは YAML を直接編集する。
#
# Usage:
#   bash scripts/run_verify_roundtrip_256_ranking.sh
#   bash scripts/run_verify_roundtrip_256_ranking.sh noise_steps=[500] n_samples=4  # 単発の上書き
set -euo pipefail

uv run python scripts/verify_roundtrip.py --config config/eval/verify_roundtrip_256_ranking.yaml "$@"
