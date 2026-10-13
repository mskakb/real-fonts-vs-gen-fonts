#!/usr/bin/env bash
# デバッグパイプライン: Generator → Generate → Scorer → Score Landscape → Guided
#
# これはオーケストレータ（多段実行の制御本体）であり、薄ラッパーにはできない。
# モデル・学習ハイパーパラメータは各 config/*/*.yaml が真実（このスクリプトでは上書きしない）。
# このスクリプトが渡すのは phase 由来の値（charset・max_steps）と、ステップ間で受け渡す
# パス（real_data_dir・gen_dir・checkpoint）のみ。
#
# Usage:
#   bash scripts/run_pipeline.sh [phase] [options]
#
# Phase:
#   t              スモークテスト (200/100 steps)
#   a  (default)   charset="A"         1クラス・手法の基本動作確認
#   b              charset="AB"        2クラス・条件付け確認
#   c              charset="uppercase" 26クラス・本番相当
#
# Control options:
#   --skip-diffusion           Step 1 をスキップ
#   --skip-generate            Step 2 をスキップ（生成済みフォントを再利用）
#   --last                     best_model_path.txt を無視して last.ckpt を使用
#   --resume-diffusion=<ckpt>  Step 1 を指定 ckpt から再開（重み+スケジューラ復元）
#   /path/to/ckpt              Step 2–5 で使う Diffusion チェックポイントを直接指定
#   --diffusion-steps=50000    Generator の max_steps（phase 既定の上書き）
#   --scorer-steps=5000        Scorer の max_steps（phase 既定の上書き）
#
# Examples:
#   bash scripts/run_pipeline.sh a
#   bash scripts/run_pipeline.sh a --skip-diffusion --last
#   bash scripts/run_pipeline.sh a --skip-diffusion /path/to/last.ckpt
set -euo pipefail

# 使用する config（モデル・学習ハイパラの真実）
DIFFUSION_CONFIG="config/diffusion/diffusion_x0_sdf.yaml"
SCORER_CONFIG="config/scorer/scorer_basic.yaml"
GENERATE_CONFIG="config/generate/generate_fonts.yaml"
SCORE_LANDSCAPE_CONFIG="config/eval/score_landscape.yaml"
GUIDED_CONFIG="config/generate/guided_sample.yaml"

# ============================================================
# フェーズのデフォルト値
# ============================================================
PHASE="${1:-a}"

case "${PHASE}" in
  t)
    CHARSET="A"; PHASE_TAG="test_A"
    DIFFUSION_MAX_STEPS=200; SCORER_MAX_STEPS=100
    ;;
  a)
    CHARSET="A"; PHASE_TAG="debug_A"
    DIFFUSION_MAX_STEPS=50000; SCORER_MAX_STEPS=5000
    ;;
  b)
    CHARSET="AB"; PHASE_TAG="debug_AB"
    DIFFUSION_MAX_STEPS=50000; SCORER_MAX_STEPS=5000
    ;;
  c)
    CHARSET="uppercase"; PHASE_TAG="debug_uppercase"
    DIFFUSION_MAX_STEPS=200000; SCORER_MAX_STEPS=50000
    ;;
  *)
    echo "ERROR: Unknown phase '${PHASE}'. Use t, a, b, or c."
    exit 1
    ;;
esac

# ============================================================
# 制御フラグの解析
# ============================================================
SKIP_DIFFUSION=false
SKIP_GENERATE=false
USE_LAST_CKPT=false
DIFFUSION_CKPT_OVERRIDE=""
RESUME_DIFFUSION=""

for arg in "$@"; do
  case $arg in
    --skip-diffusion)        SKIP_DIFFUSION=true ;;
    --skip-generate)         SKIP_GENERATE=true ;;
    --last)                  USE_LAST_CKPT=true ;;
    --resume-diffusion=*)    RESUME_DIFFUSION="${arg#--resume-diffusion=}" ;;
    --diffusion-steps=*)     DIFFUSION_MAX_STEPS="${arg#--diffusion-steps=}" ;;
    --scorer-steps=*)        SCORER_MAX_STEPS="${arg#--scorer-steps=}" ;;
    [abct])                  ;;
    --*)                     echo "WARNING: unknown option: $arg" ;;
    *)                       DIFFUSION_CKPT_OVERRIDE="${arg}" ;;
  esac
done

# ============================================================
# パス設定（オーケストレーション固有）
# ============================================================
OUTPUT_ROOT="models"
REAL_DATA_DIR="data/datasets/font2dataset_ascii_google"

# experiment_name は setup_experiment_dir が日時接頭辞を自動付与するため、ここでは付けない。
DIFFUSION_EXP="${PHASE_TAG}_diffusion"
SCORER_EXP="${PHASE_TAG}_scorer"
GEN_DIR="results/${PHASE_TAG}_generated"
GUIDED_DIR="results/${PHASE_TAG}_guided"
SCORE_LANDSCAPE_DIR="results/${PHASE_TAG}_score_landscape"

# 日時接頭辞付きの実際の出力ディレクトリ（最新の一致を採用）
resolve_ckpt_dir() {
  ls -dt "${OUTPUT_ROOT}"/*_"$1"/checkpoints 2>/dev/null | head -1
}

echo "Phase: ${PHASE}  charset='${CHARSET}'  tag=${PHASE_TAG}"
echo "  diffusion_steps=${DIFFUSION_MAX_STEPS}  scorer_steps=${SCORER_MAX_STEPS}"
echo "  diffusion_config=${DIFFUSION_CONFIG}  scorer_config=${SCORER_CONFIG}"

# ============================================================
# Step 1: Generator 学習
# ============================================================
if [ "${SKIP_DIFFUSION}" = false ]; then
  echo ""
  echo "=========================================="
  echo "Step 1: Train Diffusion"
  echo "=========================================="

  RESUME_ARG=()
  [ -n "${RESUME_DIFFUSION}" ] && RESUME_ARG=(--resume "${RESUME_DIFFUSION}")

  uv run python scripts/train_diffusion.py \
    --config "${DIFFUSION_CONFIG}" \
    "${RESUME_ARG[@]}" \
    "experiment_name=${DIFFUSION_EXP}" \
    "output_root=${OUTPUT_ROOT}" \
    "data.real_data_dir=${REAL_DATA_DIR}" \
    "data.charset=${CHARSET}" \
    "trainer.max_steps=${DIFFUSION_MAX_STEPS}"
else
  echo "Step 1 skipped (--skip-diffusion)."
fi

# チェックポイントの決定
DIFFUSION_CKPT_DIR="$(resolve_ckpt_dir "${DIFFUSION_EXP}")"
if [ -n "${DIFFUSION_CKPT_OVERRIDE}" ]; then
  DIFFUSION_CKPT="${DIFFUSION_CKPT_OVERRIDE}"
elif [ "${USE_LAST_CKPT}" = true ]; then
  DIFFUSION_CKPT="${DIFFUSION_CKPT_DIR}/last.ckpt"
elif [ -n "${DIFFUSION_CKPT_DIR}" ] && [ -f "${DIFFUSION_CKPT_DIR}/best_model_path.txt" ]; then
  DIFFUSION_CKPT="$(cat "${DIFFUSION_CKPT_DIR}/best_model_path.txt")"
else
  DIFFUSION_CKPT="${DIFFUSION_CKPT_DIR}/last.ckpt"
fi

if [ ! -f "${DIFFUSION_CKPT}" ]; then
  echo "ERROR: checkpoint not found: ${DIFFUSION_CKPT}"
  exit 1
fi
echo "Using diffusion ckpt: ${DIFFUSION_CKPT}"

# ============================================================
# Step 2: フォント生成
# ============================================================
echo ""
echo "=========================================="
echo "Step 2: Generate Fonts"
echo "=========================================="

if [ "${SKIP_GENERATE}" = false ]; then
  uv run python scripts/generate_fonts.py \
    --config "${GENERATE_CONFIG}" \
    --checkpoint "${DIFFUSION_CKPT}" \
    "output_dir=${GEN_DIR}" \
    "charset=${CHARSET}" \
    "cond_scales=[1.0,3.0,6.0]" \
    "sampling_timesteps=[50,1000]" \
    "n_samples_per_class=10"
else
  echo "Step 2 skipped (--skip-generate). Reusing: ${GEN_DIR}"
fi

# ============================================================
# Step 3: Scorer 学習
# ============================================================
echo ""
echo "=========================================="
echo "Step 3: Train Scorer"
echo "=========================================="

uv run python scripts/train_scorer.py \
  --config "${SCORER_CONFIG}" \
  "experiment_name=${SCORER_EXP}" \
  "output_root=${OUTPUT_ROOT}" \
  "data.real_data_dir=${REAL_DATA_DIR}" \
  "data.gen_data_dir=${GEN_DIR}" \
  "data.charset=${CHARSET}" \
  "data.gen_split_file=null" \
  "trainer.max_steps=${SCORER_MAX_STEPS}"

SCORER_CKPT_DIR="$(resolve_ckpt_dir "${SCORER_EXP}")"
SCORER_CKPT="${SCORER_CKPT_DIR}/last.ckpt"

# ============================================================
# Step 4: Score Landscape (E2)
# ============================================================
echo ""
echo "=========================================="
echo "Step 4: Score Landscape"
echo "=========================================="

uv run python scripts/score_landscape.py \
  --config "${SCORE_LANDSCAPE_CONFIG}" \
  "diffusion_ckpt=${DIFFUSION_CKPT}" \
  "scorer_ckpt=${SCORER_CKPT}" \
  "output_dir=${SCORE_LANDSCAPE_DIR}" \
  "charset=${CHARSET}" \
  "cond_scales=[1.0,3.0,6.0]" \
  "sampling_timesteps=[50,1000]" \
  "n_samples=32"

# ============================================================
# Step 5: Guided Sampling (E3)
# ============================================================
echo ""
echo "=========================================="
echo "Step 5: Guided Sampling"
echo "=========================================="

uv run python scripts/guided_sample.py \
  --config "${GUIDED_CONFIG}" \
  --diffusion "${DIFFUSION_CKPT}" \
  --scorer    "${SCORER_CKPT}" \
  "output_dir=${GUIDED_DIR}" \
  "charset=${CHARSET}" \
  "cond_scales=[3.0,6.0]" \
  "realness_scales=[0.0,1.0,2.0,4.0]" \
  "sampling_timesteps=[50]" \
  "n_samples_per_class=5"

echo ""
echo "=========================================="
echo "Phase ${PHASE} (charset='${CHARSET}') done."
echo "  Diffusion      : ${DIFFUSION_CKPT_DIR}"
echo "  Scorer         : ${SCORER_CKPT_DIR}"
echo "  Generated      : ${GEN_DIR}"
echo "  Score landscape: ${SCORE_LANDSCAPE_DIR}"
echo "  Guided samples : ${GUIDED_DIR}"
echo "=========================================="
