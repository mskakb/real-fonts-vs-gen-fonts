# real-fonts-vs-gen-fonts

実在するフォント画像と拡散モデルが生成したフォント画像の「本物度（realness）」を定量化し、
その判断根拠を解析し、さらにその勾配で生成を誘導することを目指す研究用コードです。

> **Status:** 研究途中の公開版です。API・設定ファイル・ディレクトリ構成は予告なく変わる可能性があります。

## 研究の問い

| | 問い |
|---|---|
| RQ1 識別可能性 | 学習済み分類器は実フォントと生成フォントを識別できるか。その根拠となる視覚特徴は何か |
| RQ2 スコアの連続性 | 本物度スコア `s(x) ∈ [0,1]` はデノイジングのタイムステップ `t` に対して連続的に変化するか |
| RQ3 スコア誘導生成 | `s(x)` の勾配を拡散サンプリングに組み込み、生成フォントの本物度を制御できるか |
| RQ4 決定境界の構造 | 実/生成フォントの決定境界はどのような幾何構造を持つか |

## システム構成

```
Generator              →  Scorer                  →  Guided Sampler
GaussianDiffusion          RealnessScorer             Classifier guidance
(U-Net, DDIM, CFG++)       (ResNet-18/50 or ViT)      μ̃ = μ_CFG++ + λ·σ²·∇s(x̂₀)
文字クラス条件付き           s(x) ∈ [0,1]
```

- **Generator**: 文字クラス（A–Z など）条件付きの拡散モデル。画像を直接、または符号付き距離場（SDF）表現で学習できます。
- **Scorer**: 実フォント=1 / 生成フォント=0 とする本物度スコア。学習方式は 2 通りです。
  - **ranking**（既定）: (real, generated) ペアに対する RankNet 損失
  - **classifier**: BCE による 2 値分類
- **Guided Sampler**: Scorer の勾配を DDIM サンプリングに加える classifier guidance。

## セットアップ

[uv](https://docs.astral.sh/uv/) を使います。Python 3.12、CUDA 12.4 版の PyTorch 2.5 を前提としています。

```bash
git clone https://github.com/mskakb/real-fonts-vs-gen-fonts.git
cd real-fonts-vs-gen-fonts
uv sync          # 依存のインストールと src/ の editable install
```

### データの準備

実フォント画像は [font2dataset](https://github.com/mskakb/font2dataset) で生成します。出力は次の形式です。

```
<real_data_dir>/
├── images/             # グリフ画像
├── sdf/                # 事前計算済み SDF（任意）
├── metadata.parquet    # char, font_family, ファイル名など
└── config.yaml
```

学習・評価・テストへの分割は `font_family` 単位で行います（同じフォントの別の文字が train と test に分かれて混入するのを防ぐため）。

```bash
uv run python scripts/make_split.py --real-dir /path/to/real --charset uppercase \
    --output config/splits/split_uppercase_seed42.yaml
```

### パスの規約

`config/` 以下の YAML に書かれたパスは、**リポジトリのルートを基準とした相対パス**です。
スクリプトは必ずリポジトリのルートから実行してください。置き場所は次のとおりで、いずれも git の管理対象外です。

| パス | 中身 |
|---|---|
| `data/datasets/` | font2dataset で生成した実フォントのデータセット |
| `models/` | 学習済みチェックポイント（Generator と Scorer） |
| `results/` | 生成画像と解析結果 |

データを別のディスクに置いている場合は、シンボリックリンクを張ると便利です。

```bash
mkdir -p data
ln -s /path/to/datasets data/datasets
ln -s /path/to/checkpoints models
ln -s /path/to/results results
```

YAML に書かれたデータセット名や実験名（例: `2026-06-17-16-48_diffusion_uppercase_sdf`）は、
作者の環境で使ったものです。自分の環境で学習した名前に合わせて、YAML を編集するか
コマンドラインの `key=value` で上書きしてください。

## 使い方

設定は `config/<subsystem>/*.yaml` で管理します。どのスクリプトも `--config <yaml>` を受け取り、
`section.key=value` 形式で設定を上書きできます。`scripts/run_*.sh` は既定の config を指定するだけの薄いラッパーです。

### 1. Generator の学習と生成

```bash
uv run python scripts/train_diffusion.py --config config/diffusion/diffusion_x0_sdf.yaml
uv run python scripts/train_diffusion.py model.dim=128 trainer.max_steps=200000   # 上書き例

uv run python scripts/generate_fonts.py --checkpoint outputs/<exp>/checkpoints/last.ckpt
```

### 2. Scorer の学習

Scorer の学習には実フォントと生成フォントの両方が必要です（`data.gen_data_dir`）。

```bash
uv run python scripts/train_scorer.py              # ranking（RankNet）
uv run python scripts/train_scorer_classifier.py   # classifier（BCE）
```

ResNet の代わりに ViT を使う場合は `config/scorer/scorer_vit.yaml` を使います。

### 3. スコアランドスケープと誘導サンプリング

```bash
# s(x̂₀) をタイムステップ t に対してプロット（RQ2）
uv run python scripts/score_landscape.py --config config/eval/score_landscape.yaml

# realness_scale λ をスイープしながら誘導サンプリング（RQ3）
uv run python scripts/guided_sample.py \
    --diffusion outputs/<exp>/checkpoints/last.ckpt \
    --scorer outputs/<exp>/checkpoints/last.ckpt
```

### 一括実行

```bash
bash scripts/run_pipeline.sh t   # スモークテスト（Generator → 生成 → Scorer を少ないステップで実行）
bash scripts/run_pipeline.sh c   # 本番（charset=uppercase）
```

途中から再開するには `--resume <ckpt>` を使います。重みだけを読み込んで `max_steps` を延長する場合は `--weights-only <ckpt>` を使います。

## 解析ツール

Scorer が何を根拠に判断しているか（RQ1）を調べるためのツールです。

| スクリプト | 内容 |
|---|---|
| `extract_scorer_features.py` | Scorer の各層（layer1–4）の特徴・スコア・モデル入力サムネイルを parquet に書き出す。以下の解析の入力になる |
| `analyze_scorer_features.py` | 層ごとの線形プローブ、埋め込みの可視化（PCA / t-SNE / UMAP）、文字別・フォント別の AUROC、スコア分布 |
| `analyze_frequency.py` | FFT による周波数解析。インク被覆率を交絡因子として統制する |
| `visualize_occlusion.py` / `analyze_occlusion.py` | Occlusion sensitivity（パッチの必要性と十分性、deletion / insertion 曲線） |
| `visualize_gradcam.py` | Grad-CAM（ResNet のみ） |
| `analyze_attention.py` | ViT の attention 系マップ（rollout、Chefer ら、token Grad-CAM など）が occlusion とどれだけ一致し、どれだけ忠実かを評価 |
| `verify_roundtrip.py` | 実画像をノイズレベル `N` まで拡散し DDIM で戻すと、Scorer の判定がどう変わるかを調べる |
| `find_memorized_pairs.py` | 生成画像ごとに最も近い実フォント画像を探し、学習データの記憶（memorization）を検証 |
| `compute_fid.py` / `score_images.py` | FID の計算、画像群のスコアリング |

複数のチェックポイントに対して上記の解析をまとめて実行するには、評価スイートを使います。

```bash
bash scripts/run_eval_suite.sh                                  # config/eval/eval_suite.yaml
bash scripts/run_eval_suite.sh --checkpoints ranking_128_genkai --steps extract_features,analyze_features
```

## ディレクトリ構成

```
config/          学習・生成・評価の YAML（設定の唯一の情報源）
scripts/         学習・生成・解析のエントリーポイントと run_*.sh ランチャ
src/real_fonts_vs_gen_fonts/
├── datamodule.py        FontDataModule（diffusion / ranking / scorer の 3 モード）
├── sdf.py               SDF の計算とデコード
├── models/              拡散モデル、Scorer、誘導サンプラ、Grad-CAM、occlusion、attention
├── analysis/            特徴解析と周波数解析（CPU だけで動く純関数）
└── utils/               設定の読み込み、実験ディレクトリ、実行情報の記録
notebooks/       検証・可視化用ノートブック
tests/           pytest
```

各スクリプトは出力ディレクトリ（`outputs/<日時>_<experiment_name>/`）に、マージ後の
`config.yaml` と、実行コマンド・git commit を記録した `run_info.yaml` を保存します。

## 開発者向けドキュメント

設計判断や実装上の注意点（例: BatchNorm を通じたラベルリークへの対策、SDF の解像度スケーリング、
`nn.MultiheadAttention` で attention weight に勾配が流れない問題）は [CLAUDE.md](CLAUDE.md) にまとめています。

## 既知の問題と今後の課題

公開時点（2026-10）で分かっている問題と、未着手の作業です。

### コードの問題

- **テストの一部が現在のコードに追従していません**（`uv run pytest` で一部が失敗します）。
  - `tests/test_configs.py`: `load_config()` の引数の形が変わったため、7 件が `TypeError` で失敗
  - `tests/test_datamodule.py::test_resolve_codepoints_unknown`: 未知の charset で `ValueError` が出なくなっている
  - `tests/test_diffusion_module.py::test_loss_with_details_shapes`: 損失関数の戻り値の数が変わったため、アンパックに失敗
  - `tests/test_pipeline.py`: ロガーなしの Trainer で `log_hyperparams` が呼ばれ、4 件が `AttributeError` で失敗
- **一部の設定ファイルが、存在しない古いチェックポイントを指しています。**
  `config/eval/score_landscape.yaml` と `config/eval/verify_roundtrip.yaml` は、初期の `charset=H` の実験を参照しています。
  使う前に、自分の環境のチェックポイントへ書き換えてください。
- **GPU での実行のみを確認しています。** CUDA 12.4 / PyTorch 2.5.1 以外の環境では動作を確認していません。

### 研究上の未解決事項

- **Scorer が見ているのは「拡散モデルの痕跡」かもしれません。**
  `verify_roundtrip.py` で実画像を少しだけ拡散して戻しただけでも、Scorer の判定は本物の生成画像に対する判定と同程度に崩れます。
  つまり Scorer が測っているのは、グリフの形としての不自然さではなく、拡散モデルの出力経路が残す痕跡である可能性があります。
  この解釈と、それを踏まえて Scorer をどう設計し直すかは未決定です。
- **ViT Scorer は動作確認だけで、本格的な学習をしていません。** `config/scorer/scorer_vit.yaml` の学習と、
  `analyze_attention.py` による attention と occlusion の一致度の評価はこれからです。
- **誘導サンプリング（RQ3）と決定境界の解析（RQ4）は初期実装の段階です。** 上の Scorer の問題が解決するまでは、
  誘導の結果を「本物らしさの向上」と解釈するべきではありません。
- **GPU クラスタで学習したチェックポイントの `config.yaml` には、クラスタ上のパスが残っています。** ローカルで使うときは、
  データのパスを上書きした config（評価スイートでは `config_local.yaml`）を別に用意しています。

## ライセンス

[MIT License](LICENSE)
