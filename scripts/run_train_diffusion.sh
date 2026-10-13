#!/usr/bin/env bash
# Phase 1: Generator (拡散モデル) の学習
#
# 設定の真実は config/diffusion/diffusion_x0_sdf.yaml（パス・charset・ハイパーパラメータすべて）。
# 設定を変えたいときは YAML を直接編集する。このスクリプトは薄いラッパー。
#
# Usage:
#   bash scripts/run_train_diffusion.sh
#   bash scripts/run_train_diffusion.sh --resume <ckpt>          # 前回から再開
#   bash scripts/run_train_diffusion.sh --weights-only <ckpt>    # 重みのみ引き継ぎ
#   bash scripts/run_train_diffusion.sh trainer.max_steps=200000 # 単発の上書き
set -euo pipefail

uv run python scripts/train_diffusion.py --config config/diffusion/diffusion_x0_sdf.yaml "$@"
