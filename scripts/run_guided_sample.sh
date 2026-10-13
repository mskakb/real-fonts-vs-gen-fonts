#!/usr/bin/env bash
# E3: 誘導サンプリング（realness_scale スイープ）
#
# 生成条件の真実は config/generate/guided_sample.yaml。設定を変えたいときは YAML を直接編集する。
# diffusion/scorer/scorer_use_ema は config キーでも CLI フラグでも指定可（CLI が config を上書き）。
# checkpoint はランごとに変わるので CLI フラグで渡すのが手軽（config: diffusion/scorer も可）。
#
# Usage:
#   bash scripts/run_guided_sample.sh --diffusion <diff.ckpt> --scorer <scorer.ckpt>
#   bash scripts/run_guided_sample.sh --diffusion <d> --scorer <s> realness_scales=[0.0,2.0]
set -euo pipefail

uv run python scripts/guided_sample.py --config config/generate/guided_sample.yaml "$@"
