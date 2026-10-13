#!/usr/bin/env bash
# Phase 3: Scorer (RealnessScorer) の学習（ranking / RankNet）
#
# 設定の真実は config/scorer/scorer_basic.yaml。設定を変えたいときは YAML を直接編集する。
# classifier (BCE) モードは run_train_scorer_classifier.sh を使う。
#
# Usage:
#   bash scripts/run_train_scorer.sh
#   bash scripts/run_train_scorer.sh --resume <ckpt>          # 前回から再開
#   bash scripts/run_train_scorer.sh --weights-only <ckpt>    # 重みのみ引き継ぎ
#   bash scripts/run_train_scorer.sh trainer.max_steps=30000  # 単発の上書き
set -euo pipefail

uv run python scripts/train_scorer.py --config config/scorer/scorer_basic.yaml "$@"
