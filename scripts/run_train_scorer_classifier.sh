#!/usr/bin/env bash
# Phase 3: Scorer (RealnessScorer) の学習（classifier / BCE モード）
#
# 設定の真実は config/scorer/scorer_classifier_basic.yaml。設定を変えたいときは YAML を直接編集する。
# ranking (RankNet) モードは run_train_scorer.sh を使う。
#
# Usage:
#   bash scripts/run_train_scorer_classifier.sh
#   bash scripts/run_train_scorer_classifier.sh --resume <ckpt>          # 前回から再開
#   bash scripts/run_train_scorer_classifier.sh --weights-only <ckpt>    # 重みのみ引き継ぎ
#   bash scripts/run_train_scorer_classifier.sh trainer.max_steps=30000  # 単発の上書き
set -euo pipefail

uv run python scripts/train_scorer_classifier.py --config config/scorer/scorer_classifier_basic.yaml "$@"
