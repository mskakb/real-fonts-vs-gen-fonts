#!/usr/bin/env bash
# Phase 2: 学習済み Generator でフォント画像を生成
#
# 生成条件の真実は config/generate/generate_fonts.yaml。設定を変えたいときは YAML を直接編集する。
# checkpoint/scorer/use_ema は config キーでも CLI フラグでも指定可（CLI が config を上書き）。
# checkpoint はランごとに変わるので CLI フラグで渡すのが手軽（config: checkpoint も可）。
#
# Usage:
#   bash scripts/run_generate_fonts.sh --checkpoint <last.ckpt>
#   bash scripts/run_generate_fonts.sh --checkpoint <ckpt> --scorer <scorer.ckpt>  # スコア付与
#   bash scripts/run_generate_fonts.sh --checkpoint <ckpt> --no-ema                # 生重みで生成
#   bash scripts/run_generate_fonts.sh --checkpoint <ckpt> n_samples_per_class=50  # 単発の上書き
set -euo pipefail

uv run python scripts/generate_fonts.py --config config/generate/generate_fonts.yaml "$@"
