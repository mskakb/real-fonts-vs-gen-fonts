# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

PyTorch + Lightning を用いて実フォント画像と AI 生成フォント画像を分類する研究プロジェクト。

## セットアップ

```bash
uv sync                  # 依存パッケージのインストール + src/ の editable install
```

## 主要コマンド

```bash
# Generator (拡散モデル) の学習
uv run python scripts/train_diffusion.py

# Scorer (RealnessScorer) の学習（ranking / RankNet、デフォルト）
uv run python scripts/train_scorer.py

# Scorer (RealnessScorer) の学習（classifier / BCE）
uv run python scripts/train_scorer_classifier.py

# フォント画像の生成（--checkpoint は必須）
uv run python scripts/generate_fonts.py --checkpoint outputs/<exp>/checkpoints/last.ckpt

# 設定を上書きして学習・生成
uv run python scripts/train_diffusion.py model.dim=128 trainer.max_steps=200000
uv run python scripts/generate_fonts.py --checkpoint last.ckpt n_samples_per_class=50

# チェックポイントから再開
uv run python scripts/train_diffusion.py --resume outputs/<exp>/checkpoints/last.ckpt
uv run python scripts/train_scorer.py --resume outputs/<exp>/checkpoints/last.ckpt

# パイプライン一括実行
bash scripts/run_pipeline.sh c              # 本番 (charset="uppercase")
bash scripts/run_pipeline.sh t              # スモークテスト (200/100 steps)
bash scripts/run_pipeline.sh a              # デバッグ (charset="A")

# スコアランドスケープ（E2: s(x̂_0) vs. t の可視化）
uv run python scripts/score_landscape.py \
    --diffusion outputs/<exp>/checkpoints/last.ckpt \
    --scorer outputs/<exp>/checkpoints/last.ckpt

# 誘導サンプリング（E3: realness_scale スイープ）
uv run python scripts/guided_sample.py \
    --diffusion outputs/<exp>/checkpoints/last.ckpt \
    --scorer outputs/<exp>/checkpoints/last.ckpt

# 記憶（memorization）検証: 生成画像ごとの最近傍実フォント画像を探す
uv run python scripts/find_memorized_pairs.py --config config/eval/find_memorized_pairs.yaml
uv run python scripts/find_memorized_pairs.py --config config/eval/find_memorized_pairs.yaml charset=H top_k=32
```

### ランチャスクリプト（`scripts/run_*.sh`）と設定の真実

**設定の真実は `config/*/*.yaml`。** 各 `run_*.sh` は `uv run python scripts/X.py --config
config/<subdir>/Y.yaml "$@"` の薄いラッパーで、シェル変数による設定制御は持たない（YAML と二重管理に
なり混乱の元になるため廃止した）。設定を恒久的に変えたいときは **YAML を直接編集**する。
単発の上書きはコマンドラインで `key=value`（および `--resume`/`--weights-only`/`--checkpoint`
等）を渡せば `"$@"` 経由で反映される。

| ランチャ | config | 備考 |
|---|---|---|
| `run_train_diffusion.sh` | `config/diffusion/diffusion_x0_sdf.yaml` | |
| `run_train_scorer.sh` | `config/scorer/scorer_basic.yaml` | |
| `run_train_scorer_classifier.sh` | `config/scorer/scorer_classifier_basic.yaml` | |
| `run_generate_fonts.sh` | `config/generate/generate_fonts.yaml` | `checkpoint`/`scorer`/`use_ema`/`scorer_use_ema` は config キーか CLI フラグ（`--checkpoint` 等）で指定。CLI が config を上書き。checkpoint はどちらかで必須 |
| `run_guided_sample.sh` | `config/generate/guided_sample.yaml` | `diffusion`/`scorer`/`scorer_use_ema` は config キーか CLI フラグ（`--diffusion`/`--scorer`/`--no-ema`）で指定。CLI が config を上書き。diffusion/scorer はどちらかで必須 |
| `run_score_landscape.sh` | `config/eval/score_landscape.yaml` | |
| `run_score_images.sh` | `config/eval/score_images.yaml` | |
| `run_verify_roundtrip.sh` | `config/eval/verify_roundtrip.yaml` | 実画像を `q_sample` で `N` まで forward 拡散 → `ddim_sample_from` で独立に指定した `M` ステップの DDIM で `x_0` まで reverse し、アーティファクト・Scorer 判定への影響を検証する。詳細は「拡散ラウンドトリップ検証」節を参照 |
| `run_verify_roundtrip_{96,128,256}_{ranking,classifier}.sh` | `config/eval/verify_roundtrip_{96,128,256}_{ranking,classifier}.yaml` | 上記の解像度別・Scorer 種別固定版（計 6 本）。96/128/256px の拡散モデルと**画像サイズが一致する** ranking/classifier Scorer を組み合わせ済み。詳細は「拡散ラウンドトリップ検証」節を参照 |
| `run_compute_fid.sh` | `config/eval/compute_fid.yaml` | |
| `run_find_memorized_pairs.sh` | `config/eval/find_memorized_pairs.yaml` | |
| `run_make_split.sh` | `config/splits/make_split.yaml` | |
| `run_eval_suite.sh` | `config/eval/eval_suite.yaml` | 複数 Scorer チェックポイント横断で feature 抽出/特徴解析/周波数解析/occlusion解析/per-char可視化/attention忠実性検証を一括実行するオーケストレータ。`--checkpoints`/`--steps` で絞り込み、`--stop-on-error` で中断可。詳細は「Scorer 評価スイートの一括実行」節を参照 |

例外:
- `run_extract_scorer_features.sh`: `--config` は**学習済み Scorer の `config.yaml`**（モデル
  再構築用）を指すため独立ランチャ config を持たず、コマンドに直書きする。
- `run_visualize_occlusion.sh`: 同様に `--config` は**学習済み Scorer の `config.yaml`**（データ/SDF
  kwarg 再現用）を指すため独立ランチャ config を持たない。occlusion パラメータ（`--patch`/`--stride`/
  `--baseline` 等）は argparse フラグで、`"$@"` から単発上書きできる。`--all-chars` でモデル・データを
  1 回だけロードしたまま全文字（または `--char` で指定したリスト）を内部ループし、文字ごとの
  サブディレクトリに出力する（毎回の再ロードを避けられる）。単一チェックポイントのアドホック実行用。
  複数チェックポイント横断のバッチ実行は `run_eval_suite.sh` を使う。
- `run_visualize_gradcam.sh`: 同様に `--config` は**学習済み Scorer の `config.yaml`**（データ/SDF
  kwarg 再現用）を指すため独立ランチャ config を持たない。対象レイヤー（`--layer`）等は argparse
  フラグで、`"$@"` から単発上書きできる。`--all-chars` は occlusion 版と同様、モデル・データを
  使い回して文字ごとにサブディレクトリへ出力する。単一チェックポイントのアドホック実行用
  （複数チェックポイント横断は `run_eval_suite.sh`）。
- `run_analyze_occlusion.sh` (`analyze_occlusion.py`): 同様に `--config` は**学習済み Scorer の
  `config.yaml`**（データ/SDF kwarg 再現用）を指すため独立ランチャ config を持たない。追加で
  `--features`（`extract_scorer_features.py` 出力の parquet・同一 checkpoint 由来・必須）を取る。
  モデル/checkpoint/features パスはスクリプトに直書きし、`--char`/`--label`/`--n-images`/`--patch`/
  `--embedding-layer`/`--reducer` 等は `"$@"` から単発上書きする。詳細は「Scorer の判断根拠可視化」節を参照。
  単一チェックポイントのアドホック実行用（patch サイズ掃引を含む複数チェックポイント横断バッチは
  `run_eval_suite.sh` の `analyze_occlusion` ステップを使う）。
- `run_pipeline.sh`: 多段オーケストレータ（薄ラッパー不可）。モデル・学習ハイパラは各
  config に従い、スクリプトは phase 由来の `charset`/`max_steps` とステップ間のパス受け渡し
  のみを担う。`experiment_name` に日時を付けない（Python が自動付与するため）。出力先は
  日時接頭辞付きディレクトリを最新一致で解決する。

`score_landscape.py` / `score_images.py` / `compute_fid.py` /
`make_split.py` / `find_memorized_pairs.py` は `--config <yaml>` + `key=value` 上書きで動く
（個別の `--diffusion` 等のフラグは廃止）。

## アーキテクチャ概要

### 3 サブシステム（研究草案: `.memo/research_draft.md`）

```
Generator       → Scorer          → Guided Sampler
GaussianDiff-     RealnessNet       Classifier Guidance
usion (DDIM)      ResNet-18         μ̃ = μ_CFG++ + λ·σ²·∇s(x̂₀)
s(x) ∈ [0,1]
```

### データフロー

```
font2dataset CLI
  → data/raw/real/images/ + metadata.parquet
FontDataModule(mode="diffusion")  →  DiffusionModule  →  学習済み Generator
FontDataModule(mode="ranking")    →  ScorerModule     →  学習済み Scorer（RankNet、デフォルト）
FontDataModule(mode="scorer")     →  ScorerModule     →  学習済み Scorer（BCE 分類）
```

### FontDataModule の 3 モード

| mode | label の意味 | 用途 | 学習スクリプト / config |
|---|---|---|---|
| `"diffusion"` | 文字インデックス A=0…Z=25 | Generator 学習 | `train_diffusion.py` / `config/diffusion/diffusion.yaml` |
| `"ranking"` | ペアワイズ (real, generated) | Scorer 学習・RankNet（`gen_data_dir` 必須、デフォルト） | `train_scorer.py` / `config/scorer/scorer.yaml` |
| `"scorer"` | real=1 / generated=0 | Scorer 学習・BCE 分類（`gen_data_dir` 必須） | `train_scorer_classifier.py` / `config/scorer/scorer_classifier.yaml` |

`mode` と `model.ranking` はセットで変える（`config/scorer/scorer.yaml` / `config/scorer/scorer_classifier.yaml` 参照）。

`"ranking"` モードの `RankingDataset` は `real_image`/`gen_image` に加えて、real-real / gen-gen の
"tie" ペア用に `real_image2`/`gen_image2`（クラス全体からランダムに選んだ別サンプル）も返す。
`model.tie_weight_real` / `model.tie_weight_gen`（それぞれ `> 0` で有効、デフォルト `0.0`）に応じて、
`ScorerModule` はこれらに RankNet 損失（`p_ij=0.5`）を追加する。学習ログには ranking 損失を
`train/rank_loss`、tie 損失を `train/tie_loss_real` / `train/tie_loss_gen` として個別に出力し、
`train/loss` はそれらの重み付き合計（最終的な学習対象の損失）。

**BN リーク対策（重要）**: ranking の `ScorerModule` は real/gen（+ tie ペア）を
`torch.cat([...])` で **1 つの混在バッチに連結して forward → `chunk` で分割**する
（`_ranking_train_step` / `_ranking_val_step` / `test_step`）。real と gen を
**クラス純粋バッチとして別々に forward すると、backbone の BatchNorm がバッチ統計経由で
real/gen ラベルをリーク**し、学習時（train モード=バッチ統計）だけ rank_acc≈1.0 まで簡単に解けるのに
評価時（eval モード=固定 running stats）で AUROC≈0.5 に崩壊する。連結 forward で train/eval の
正規化を一致させ、このショートカットを断つ。`mode="scorer"`（BCE 分類）は元々 real/gen 混在バッチ
なのでこの問題は起きない（ranking 固有）。eval モードでは連結しても数値は不変だが規約統一のため揃える。

scorer/ranking モードも `data.use_sdf=true` で SDF 表現を入力にできる（real/gen 両方）。raw SDF か
グレースケール decode（`data.sdf_decode`）かを選べる。詳細は「事前計算済み SDF」節を参照。

スプリット管理:
- 実フォント: `split_file`（`make_split.py` 生成）— `font_family` 単位で分割。文字レベル分割は同一フォントのリークが起きるため非推奨
- 生成フォント: `gen_split_file`（`make_gen_split.py` 生成）— ファイル単位で分割。`null` 指定時は `seed` ベースのランダム分割にフォールバック

```bash
# 実フォントのスプリット生成
uv run python scripts/make_split.py --real-dir /path/to/real --charset uppercase --output config/splits/split_uppercase_seed42.yaml

# 生成フォントのスプリット生成
uv run python scripts/make_gen_split.py --gen-dir /path/to/generated --charset H --output config/splits/gen_split_H_seed42.yaml
```

### 事前計算済み SDF（precomputed SDF）

新しい font2dataset 出力は `images/` に加えて `sdf/`（事前計算済み SDF の 8-bit grayscale PNG、
`images/` と同名・同数）を含み、`metadata.parquet` に `sdf_png_file` 列を持つ。
`use_sdf=true` のとき、`FontDataModule`（**diffusion / scorer / ranking モード**）は `data.sdf_source`
に従って SDF の入手方法を切り替える:

| `sdf_source` | 挙動 |
|---|---|
| `auto`（デフォルト） | `data_dir/sdf/` と `sdf_png_file` 列があれば事前計算済みをロード、無ければ `binary_image_to_sdf` で on-the-fly 計算（旧データセット互換） |
| `precomputed` | 常に `sdf/` からロード（無ければ `FileNotFoundError`） |
| `compute` | 常に on-the-fly 計算（`distance_transform_edt`、CPU ボトルネック） |

- 事前計算済み SDF は**生成時の `sdf_max_dist` で固定**されている。`data_dir/config.yaml` の
  `sdf_max_dist` と `data.sdf_max_dist` が食い違う場合は警告ログを出す（値はデータセット側に従う）
- **scorer/ranking モードの SDF 対応**: `use_sdf=true` のとき real・gen **両方**を SDF 表現で扱う。
  `auto` では real/gen 双方に `sdf/`（+ `sdf_png_file` 列）が揃う場合のみ precomputed を使い、
  片方でも欠ければ両方 on-the-fly に揃える（real/gen で経路が食い違うと不公平なため）。
  gen 側 `sdf/` は `generate_fonts.py` の `save_sdf=true` で生成する。on-the-fly 計算は
  `binary_image_to_sdf` が 1ch に集約するため `image_channels`（既定 3）へ複製して ResNet に渡す。
- **`sdf_decode`（scorer/ranking のみ）**: Scorer に raw SDF をそのまま渡すか、グレースケール decode して
  渡すかを選ぶ。`null`=raw SDF、`"hard"`=0.5 二値化、`"sigmoid"`=`sdf_sharpness` でソフト二値化、
  `"smoothstep"`=アンチエイリアス連続グレースケール（`sdf_filter_width_px`/`sdf_max_dist` 使用）。
  diffusion の FID decode（`_sdf_decode`）と同じ `sdf.py` 関数を再利用する。**留意**: raw SDF は
  real（font2dataset の EDT）と gen（拡散モデルの直接出力）で統計が異なりショートカット学習の温床に
  なりうる。同一 decode でグレースケール化すると描画パイプラインを揃えられる（`.memo/research_draft.md`）。
- `stroke_density` / `stroke_map` は precomputed 使用時も引き続き `images/` の生画像から算出する
  （diffusion モードのみ。scorer/ranking では未使用）
- `generate_fonts.py` の `save_sdf=true`（`use_sdf=true` のモデルのみ有効）で、生成データセットも
  `sdf/` 併存形式で出力できる（`images/` にデコード済みグリフ、`sdf/` に生 SDF を同名 8-bit PNG、
  `metadata.parquet` に `sdf_png_file` 列）。正規化に使った `sdf_max_dist`・`use_sdf` は出力先の
  `config.yaml` に記録される。diffusion の precomputed SDF 入力、scorer/ranking の SDF 学習、可視化に
  再利用可能

**`sdf_max_dist` の解像度スケーリング**: `max_dist` は px 単位の長さ（輪郭の遷移帯の半幅）なので、
解像度を変えても「同じ連続 SDF 場」を表すには解像度に**線形**でスケールする（基準 64px=10.0 → `max_dist = 10.0 × H/64`、128px なら 20.0）。
面積基準の二乗スケール（×4）は次元が合わず、parity MSE でも 64px の挙動から大きく外れる
（`notebooks/sdf_visualization.ipynb` のセクション8で検証済み）。なお実フォントの線幅は ~4px と細く、
どの `max_dist` でも内部は飽和しない（常にランプ）ため、`max_dist` の実効的な役割は内部の solid 化ではなく
輪郭の鋭さ（傾き `1/(2·max_dist)`）／0.5 レベルセットの位置精度。

### GaussianDiffusion の拡張 API

```python
# return_per_sample=True でノイズレベル別ログに使う per-sample 損失を取得
loss, per_sample, t = diffusion(img, classes=classes, return_per_sample=True)

# return_x0=True でデノイズ後の x̂_0 を取得（パーセプチュアルロス用）
# return_per_sample と併用可能
loss, per_sample, t, x0_pred = diffusion(img, classes=classes, return_per_sample=True, return_x0=True)

# ddim_sample_from: 既知の x_t（q_sample 等で用意）から DDIM で任意ステップ数だけ x_0 まで逆拡散する。
# t_start・sampling_timesteps を独立に指定できる（ddim_sample は img=randn・t_start=T-1 の特殊ケースとして
# これに委譲している）。forward で使ったノイズレベルと reverse のネットワーク評価回数を分離したい
# ラウンドトリップ検証（後述）で使う。
x_recon = diffusion.ddim_sample_from(x_t, classes, t_start=N, sampling_timesteps=M)
```

### 拡散ラウンドトリップ検証（`verify_roundtrip.py`）

実画像 `x_0` を `q_sample` で `t=N` まで forward 拡散（解析的に 1 発）し、`ddim_sample_from` で
`N` と独立に指定した `M` ステップの DDIM で `x_0` まで reverse するだけで、アーティファクトが乗るか・
Scorer の real/fake 判定に影響するかを検証する。`--config config/eval/verify_roundtrip.yaml` +
`key=value` 上書きの規約は他の eval スクリプトと同じ。`noise_steps`（N のリスト）×
`ddim_reverse_steps`（M のリスト）の全組み合わせについて、per 画像の realness スコア
（`s0` vs `s_rt`）・再構成誤差（decode 後 MSE・二値化画素不一致率）・元画像と roundtrip 画像を
並べた可視化グリッド（`grids/N{N}_M{M}_grid.png`、`original | noisy x_N | roundtrip` の3列。
中央列は q_sample 直後・SDF decode 前の生テンソルを `unnormalize_to_zero_to_one` で戻しただけの表示
（ノイズが乗った x_N は有効な SDF 場ではないため decode しない）を出力する（`roundtrip_scores.csv` に集約）。
`N` がどれだけのノイズ強度に対応するかは q_sample の閉形式（`x_N = sqrt(ᾱ_N)·x_0 + sqrt(1-ᾱ_N)·ε`）から
解析的に決まるため、`diffusion.alphas_cumprod` から信号係数 `sqrt(ᾱ_t)`・ノイズ係数 `sqrt(1-ᾱ_t)` を
全 t について1回だけプロットし `noise_steps` の各 N に縦線を引いた `noise_schedule.png` も出力する。
`(N, M)` ごとに `s0`（元画像, label=1）vs `s_rt`（roundtrip 画像, label=0）の AUROC も算出する
（`roundtrip_auroc.csv`・`auroc_vs_noise.png`）。1.0 に近いほど「roundtrip だけで Scorer が
real/fake を明確に切り分けられる」ことを意味し、s0/s_rt の平均差だけでは見えない分布の分離度を表す。
ROC 曲線自体（TPR vs FPR）も `M` ごとに 1 枚、`N` を重ね書きして `roc_curves_M{M}.png` に出力する。
Scorer は 3ch 入力前提のため、diffusion モデルが 1ch（`model.channels=1`）で学習されている場合は
スコアリング直前にのみチャンネル複製する（保存・誤差計算は diffusion 側のネイティブチャンネル数のまま）。
use_sdf のモデルは real/roundtrip 双方を `sdf_decode` で同一 decode してから比較する（SDF 節参照）。
`n_samples` は `chunk_size`（既定 32）単位でチャンク分割して逐次処理する（実画像の読み込み・
decode・スコアリング・DDIM reverse すべて）ため、VRAM 使用量は `n_samples` に依存せず
`chunk_size` だけで決まる（`DiffusionModule._sample_in_chunks` と同じ考え方）。`n_samples` を
数百〜数千に増やしても `chunk_size` を GPU メモリに収まる値に保てば OOM しない。

**解像度別の固定エントリースクリプト**: `uppercase` charset の SDF モデルについて、画像サイズが
一致する拡散モデル×Scorer の組み合わせを固定した config + 起動スクリプトを 96/128/256px 各 2 種
（ranking / classifier）、計 6 組用意している（`run_verify_roundtrip_{96,128,256}_{ranking,classifier}.sh`）。

| 解像度 | 拡散モデル | ranking Scorer | classifier Scorer |
|---|---|---|---|
| 96px  | `2026-06-17-16-48_diffusion_uppercase_sdf` | `2026-06-20-23-41_scorer_uppercase` | `2026-08-10-12-57_scorer_classifier_uppercase` |
| 128px | `2026-07-16-10-35_diffusion_uppercase_sdf_128_genkai` | `2026-08-17-12-20_scorer_uppercase_sdf_128_genkai` | `2026-08-18-05-37_scorer_classifier_uppercase_sdf_128_genkai` |
| 256px | `2026-07-16-06-12_diffusion_uppercase_sdf_256_genkai` | `2026-08-17-12-20_scorer_uppercase_sdf_256_genkai` | `2026-08-19-13-05_scorer_classifier_uppercase_sdf_256_genkai` |

128px/256px は genkai（外部の計算機環境）上で学習されたため、チェックポイント自身の `config.yaml` に
記録された `real_data_dir`/`split_file` は genkai 上のパスのままになっている
（`load_from_checkpoint` が復元する `.ckpt` 内の `hparams` は `image_size`/`use_sdf`/`sdf_max_dist`
等のモデルハイパラのみで、これらは正しく追従するが、real データパスは checkpoint に含まれない）。
そのため対応する各 config はローカルの同一データセット（`font2dataset_google_uppercase_{96,128,256}_sdf`）
と `config/splits/split_ascii_seed42.yaml` のローカルパスに明示的に差し替えてある。

### パーセプチュアルロス（`DiffusionModule`）

x̂_0 に対してパーセプチュアルロスを追加できる（`perceptual_weight > 0` で有効化）。

```yaml
# config/diffusion/diffusion.yaml の model セクション
perceptual_weight: 0.01       # 0 で無効。0.01〜0.1 程度から試す
perceptual_network: "lpips"   # "lpips"（AlexNet）| "vgg16"（relu2_2 特徴）
perceptual_t_max_frac: 0.5    # T の前半のみ適用（高ノイズ域は x̂_0 推定が荒いため）
```

ネットワークの追加: `src/real_fonts_vs_gen_fonts/models/perceptual.py` の `_REGISTRY` に登録する。

### LR スケジューラ

`lr_total_steps` は YAML に書かず、スクリプト側で `trainer.max_steps` から自動注入する
（2 箇所に同じ値を書くと更新漏れが起きるため）。

### Scorer の LR スケジューラ選択

`ScorerModule` は `model.scheduler` で 2 方式を選べる（既定 `cosine` で後方互換）:

- `cosine`: `warmup (LinearLR) → CosineAnnealingLR`。`lr_warmup_steps`・`lr_total_steps`（= `max_steps`）を使う時間ベース減衰。
- `plateau`: `ReduceLROnPlateau`。`plateau_monitor`（既定 `val/loss`、`plateau_mode=min`）が `plateau_patience` 回の検証で改善しなければ lr を `×plateau_factor`（下限 `plateau_min_lr`）。**warmup は無効**（`SequentialLR` が `ReduceLROnPlateau` を子に取れないため）。

plateau は検証カデンツに合わせて step する。`configure_optimizers` が `self.trainer.val_check_interval` を読み、
**int（ステップ単位）→ `interval="step"`/`frequency=val_check_interval`**、**float（エポック単位）→ `interval="epoch"`** に切り替える
（`max_steps` が 1 エポック未満でも発火させるため）。`plateau_patience` の単位は「検証回数」。
`early_stopping` も同じ `val/loss` を監視する場合は `plateau_patience < early_stopping.patience` にしないと
lr が下がる前に学習が止まる。検証は EMA パラメータで評価されるため、plateau は EMA ベースの `val/loss` を見て
生の optimizer lr を調整する。

### Scorer の EMA

`ScorerModule` は DiffusionModule と同様に `torch_ema.ExponentialMovingAverage` で重みの EMA を保持する（学習ロスの振動を緩和するため）。

```yaml
# config/scorer/scorer.yaml の model セクション
ema_decay: 0.999          # EMA 平滑化係数。Scorer は学習が速いため diffusion (0.9999) より短め
ema_start_step: 0         # EMA 更新を開始するグローバルステップ
```

検証・テスト（`validation_step` / `test_step`）は EMA 平均パラメータで評価する。チェックポイントには `ema_state` が保存され、`--resume` で復元される。

### Scorer 検証用ノートブック

`notebooks/scorer_validation.ipynb`: Scorer の学習出力ディレクトリ（`config.yaml` + `checkpoints/*.ckpt`）から `ScorerModule` と `FontDataModule` を直接ロードし、ranking モードでの rank_acc・AUROC・スコア分布・サンプル画像（高/低スコア）を確認する。`OUTPUT_DIR` / `CKPT_NAME` / `USE_EMA` / `SPLIT` をセル先頭で指定する。

### Scorer 特徴解析（parquet 契約による一元化）

Scorer の特徴解析は **「抽出 1 回（GPU）→ 解析多数回（CPU）」を parquet でつなぐ 3 層構成**。
解析ロジックは `src/real_fonts_vs_gen_fonts/analysis/feature_analysis.py` の純関数に集約し、
CLI（`analyze_scorer_features.py`）と notebook（`scorer_feature_analysis.ipynb`）が同じ実装を共用する
（以前は extract / analyze / notebook で同じ「モデルロード→datamodule→split巡回→特徴抽出」を 3 重実装していた）。

```
extract_scorer_features.py (GPU) → scorer_features.parquet (+ inputs/ サムネ)
  │  layer1-4 の GAP 特徴(列 l1_*/l2_*/l3_*/l4_*) + score + logit + image_path
  │  + メタ(split,label,char,font_family)
  ▼
analysis/feature_analysis.py (CPU 純関数: load_features / fit_combined_embedding / 
  fit_linear_probe / layer_probe_auroc / per_group_stats / per_char_auroc / per_font_auroc /
  plot_score_distributions / plot_score_kde / score_separation_stats /
  plot_score_ranked_images / plot_embedding_thumbnails /
  plot_group_representative_images / plot_misclassified_images /
  plot_kde_region_images / plot_worst_auroc_char_examples / has_images / plot_*)
  ├─► analyze_scorer_features.py（CLI ドライバ）
  └─► scorer_feature_analysis.ipynb（薄い呼び出し）
```

- **`extract_scorer_features.py`**: `RealnessScorer.extract_intermediate` で **layer1-4 すべての GAP 特徴**を
  parquet 化（列プレフィックス `l1_`〜`l4_`、`LAYER_PREFIXES` と同期）。これが解析の唯一の真実で、
  analyze はモデルを一切ロードせず parquet だけで全解析を回せる。次元は backbone 依存
  （resnet18: 64/128/256/512、resnet50: 256/512/1024/2048）。backbone 名等は `run_info.yaml` に記録。
  - `score`（sigmoid）に加えて `logit`（sigmoid 前）も列として保存する。
  - 既定で**モデル入力のサムネ**を `inputs/<split>/<idx>.png`（8-bit grayscale）に保存し、
    parquet 相対パスを `image_path` 列に記録する（画像系プロット用）。サムネは
    `batch["image"]`（resize+SDF decode+binarize 済みの**モデルが実際に見た入力**）を使うため、
    use_sdf=true なら SDF/デコード後表現になる。train は augment+shuffle 後の入力＝忠実だが、
    画像系プロットの既定 split は test。`--no-save-inputs` で抑止（`image_path` 列無し＝画像系スキップ）。
- **`analyze_scorer_features.py`**: `--features <parquet>` を食って以下を出力。
  `--features` 省略時に `--checkpoint` があれば、内部で `extract_scorer_features.py` を subprocess 実行して
  parquet を作ってから解析する（後方互換の利便）。CLI 上書き（`key=value`）は extraction に渡る。
  - **per-layer 線形プローブ** + ROC（train で fit → 各 split に適用、split 別 `probe_results.csv`）
  - **同一空間プロット**（`--reducer pca|tsne|umap`、`--embedding-layer` 既定 layer4、
    **全 split を結合して一度だけ fit**）。**real/gen=色相・split=明度**（淡=train→濃=test、マーカー形状も split で冗長付与）/ realness score 連続色 / char カテゴリ色 の 3 枚。
    t-SNE は `transform` を持たないが結合一括 fit なので問題なし。t-SNE/UMAP は `--embedding-samples` で fit 上限。
  - **文字別・フォント別統計**（`--stats-split` 既定 test、`all` で各 split を個別に出力）。
    `per_char_stats.csv`/`per_font_stats.csv`（score の
    mean/std/median/min/max/count を split×label 別に集計）と対応 PNG。per-font の棒グラフ
    （`per_font_real_score_*`）と代表画像（`repr_font_family_*`）は **lowest/middle/highest の 3 帯**を
    出す（フォント数が `3×top_n`/`3×n_groups` 以下なら全件）。char の代表画像も同様に中間群を含む。
    `repr_font_family_*` のセルキャプション（フォント名+スコアが行内で反復し重なりやすい）は
    `--cell-labels` と独立に `--font-cell-labels`/`--no-font-cell-labels`（既定 on）で個別に
    オンオフできる（他グリッドのキャプションは維持したまま font_family だけ純画像グリッドにできる）。
  - **スコア / ロジット分布**（`distributions/`）: real vs gen の重ねヒストグラム。`score` と、
    parquet に `logit` 列があれば `logit` も。全 split 結合（`*_dist.png`）と各 stats-split（`*_dist_<split>.png`）。
  - **スコア KDE**（`distributions/`）: `scipy.stats.gaussian_kde` で real/gen の score（あれば logit）密度を
    推定し 1 枚に重ねる（`score_kde[_<split>].png`）。各クラス平均・外側裾（real p5 / gen p95）・
    **混同領域** `min(p_real, p_gen)` 曲線とそのピーク `s*`（=`argmax_s min`）・半値帯を注釈。凡例に
    **重なり係数 OVL=∫min**（図タイトルは廃止したのでファイル名に無い OVL のみ凡例で表示）。
    あるクラスが <5 サンプルか分散 0 のときはスキップ（クラッシュさせない）。
    `--kde-bw` で帯域幅を上書き（既定 Scott）。
  - **`score_separation.csv`**（output 直下）: split×［real_mean, gen_mean, s_star, ovl, n_real, n_gen］。
    OVL は real/gen の分離不能度を 1 スカラで表し、モデル間比較の素材になる。`score_separation_stats` で算出。
  - **画像グリッド**（`images/`・埋め込みサムネは `embedding/`）: `image_path` 列があり実サムネが
    存在する場合のみ出力（無ければ自動スキップ。`--no-images` で一括スキップ）。表示は**モデル入力**
    （SDF モデルは SDF/デコード後）。各プロットは**単一カテゴリの N×N グリッド**を帯ごとに 1 枚ずつ出力し、
    どのカテゴリ・帯・split かは**ファイル名に記入**する（`--grid-n` 既定 8＝64 枚／不足分は空白セル）。
    各セルの score/char ラベルは `--no-cell-labels` で消して純画像グリッドにできる（識別はファイル名のみ）。
    種類: スコア順（`ranked_{real,gen}_{top,middle,bottom}_*`）、高確信の誤判定
    （`misclassified_real_*`: real→低スコア／`misclassified_gen_*`: gen→高スコア）、文字別/フォント別代表
    （`repr_{char,font_family}_{lowest,middle,highest}_*`、各行＝1 グループ・mean score 近傍）、
    per-char AUROC 最悪文字の例（`worst_auroc_{real,gen}_*`、各行＝1 文字。2 枚を並べて real vs gen 比較）、
    **KDE 領域サンプル**（`kde_{real,gen}_{mean,tail,overlap}_*`: 平均・外側裾・混同点 s* 近傍。
    `kde_real_overlap_*` と `kde_gen_overlap_*` は同一スコア帯なので並べて直接比較）、埋め込み上にグリフ重畳
    （`embed_<reducer>_thumbnails.png`、`--n-thumbs`・グリッドビンで間引き・枠色で real/gen。これは散布図で N×N グリッド化対象外）。
- **per-char AUROC と per-font AUROC は定義が異なる**: char は real/gen 両クラスが揃うので
  同一文字内の real-vs-gen AUROC（`per_char_auroc.csv`）をそのまま出せる。実フォントは label=1
  単一クラスなので同じ定義はできないが、`per_font_auroc.csv`（`per_font_auroc()`）は各実フォントの
  real スコアを**その split の gen スコア全体（フォント別ではなく生成側全体）とプールして** AUROC を
  計算する（サンプル数 `min_real`＝既定 5 未満のフォントはスキップ）。「そのフォントが生成分布全体から
  どれだけ分離できるか」という意味で、フォント同士のペアワイズ判別ではない点に注意。
  `plot_group_stats(..., "font_family", auroc=font_auroc)` は `per_font_real_score_*` と同じ
  lowest/middle/highest 3 帯レイアウトで `per_font_auroc_*.png` も出す（最低 AUROC＝モデルが
  gen と最も混同しやすい実フォント）。生成側 font_family は `"generated"`/`"guided"` の単一バケットで
  フォント別ランキングからは除外する。
  - **`--font-scale`（既定 1.5）**: 全図のフォントサイズを一括倍率調整する単一ノブ。キャプション
    （画像グリッドのセルラベル）だけでなく軸タイトル・軸ラベル・目盛り・凡例・注釈も
    まとめて拡縮する（`feature_analysis.set_font_scale` が明示 `fontsize`＝`_fs()` と matplotlib
    rcParams の両方を更新）。notebook は config セルの `FONT_SCALE` で同じ制御。
  - **`--legend-loc`（既定 `best`）**: 凡例位置を選択（`set_legend_loc` / notebook の `LEGEND_LOC`）。
    `outside`＝軸の右外（データに被らないが図が横に広がりプロットが小さく見える、`savefig(bbox_inches="tight")`
    でクリップ防止）、それ以外は matplotlib の loc（`best`/`upper right` 等）で**軸内**に置く（プロットの
    大きさを保つがフォント拡大時は被りうる）。`_legend` ヘルパが両者を切替。
  - **図タイトルは全廃**（`set_title`/グリッドの `suptitle`、何のプロットかはファイル名で分かるため）。
    タイトルにしか無かった情報は凡例に移した（KDE の OVL、埋め込みサムネの real/gen 色キー）。

### Scorer の判断根拠可視化（Grad-CAM / Occlusion sensitivity）

Scorer が real/gen をどこの入力領域で判断しているかを可視化する 2 手法。

- **Grad-CAM**（`models/grad_cam.py` / `scripts/visualize_gradcam.py` / `run_visualize_gradcam.sh`）:
  指定レイヤーの勾配ベース。**対象レイヤーを `--layer`（layer1〜4、既定 layer4）で選択可能**。
  浅い層ほど空間解像度は上がるが（例: 96px×resnet50 で layer1≈24×24、layer4≈3×3）、conv1/maxpool の
  stride 累積があるため occlusion ほど自由な解像度にはならず、layer1 でも occlusion の
  `stride=4` 相当（~24×24）が上限。かつ Grad-CAM 本来の重み付け（GAP した勾配をチャネル重みにする）は
  最終 conv 層想定の手法のため、layer1〜3 での解釈は layer4 ほど確立していない（動くが定性的な解釈は
  要注意）。**入力は SDF 対応**: occlusion と同じく `FontDataModule(mode="scorer")` から
  `batch["image"]`（モデルが実際に見た入力）を取るため、SDF 学習済み Scorer でも正しい。
  → `--config` は**学習済み Scorer の `config.yaml`**（データ/SDF kwarg 再現用、occlusion と同じ規約）。
  `GradCAM(scorer, layer=...)` は ReLU 後で非負の値を返すため、`overlay_heatmap` は単極カラーマップ
  （`cm.hot`）で `[0, 1]` 共有スケールにする（occlusion の符号付き発散マップ RdBu_r とは意図的に別物）。
  出力は `gradcam_real.png` / `gradcam_gen.png`（共有カラーバー付き）。
- **Occlusion sensitivity**（`models/occlusion.py` / `scripts/visualize_occlusion.py` /
  `run_visualize_occlusion.sh`）: 入力の一部を patch で遮蔽しスコア変化 `Δ = s0 - s_occ` を実測する。
  勾配不要（`forward_logits` のみ・**backbone 非依存**、resnet50 も無改造）、**解像度は stride で自由**
  （96px/stride4 で ~24×24）。`occlusion_sensitivity(scorer, x, patch_size, stride, baseline, target,
  chunk_size)` は符号付きマップ（正=そこを隠すとスコア低下＝realness を支持）を返し、`overlay_occlusion`
  が **発散カラーマップ（RdBu_r）** で 0 中心・画像ごと max|Δ| 対称正規化してオーバレイする（赤=支持/青=抑制）。
  - **入力は SDF 対応**: `extract_scorer_features.py` と同じ `FontDataModule(mode="scorer")` から
    `batch["image"]`（モデルが実際に見た入力＝SDF decode 済み）を取るため、SDF 学習済み Scorer でも正しい。
    → `--config` は**学習済み Scorer の `config.yaml`**（データ/SDF kwarg 再現用、extract と同じ規約）。
  - **baseline（遮蔽の塗り値）が最重要ノブ**: `background`（外周中央値、白黒規約・raw SDF 遠方値に依らず
    背景を頑健に推定、既定）/ `mean` / `min` / `max` / float 定数。`--target` は `logit`（既定・未飽和で
    高感度）/ `score`（sigmoid 後）。`--patch`/`--stride` で解像度と計算コスト（1 画像 (H/stride)² 回の
    前向き、`no_grad`）を調整。出力は `occlusion_real.png` / `occlusion_gen.png`（共有カラーバー付き）。

どちらのスクリプトも `--all-chars` でモデル・データのロードを 1 回に抑えたまま全文字（または
`--char` 指定リスト）を内部ループし、`<output_dir>/<char>/` に文字ごとの `*_real.png`/`*_gen.png` を
出力できる（省略時の既定挙動＝複数文字を 1 枚のグリッドに混在させる、は不変）。複数チェックポイント×
文字別を一括で回すには `run_eval_suite.sh` の `visualize_occlusion`/`visualize_gradcam` ステップを使う。

- **`analyze_occlusion.py`**（`models/occlusion.py` の `build_occluded_batch` を使用）: 1 枚（または
  `--n-images` 枚）の画像に対する occlusion sensitivity をさらに掘り下げ、`occlusion_sensitivity` が
  内部で捨てている **per-patch の絶対スコア `s(x_occ)` と中間特徴ベクトル**を保持したまま可視化する。
  - **`--all-chars` / `--label both`**: `visualize_occlusion.py` と同じ規約で、`--char` を複数指定
    （`nargs="+"`）でき、`--all-chars` 指定時は `--char` 未指定なら charset 全文字を、指定ありなら
    そのリストを内部ループする。`--label both` で real/gen 両方を回す。model/data/features のロードは
    実行全体で 1 回のみ（文字・ラベルごとの再ロードを避ける）。出力は既存どおり
    `<output_dir>/<char>_<real|gen>_<i>/`（char・label を名前に含むため衝突しない）。
    複数チェックポイント×パッチサイズを一括で回すには `run_eval_suite.sh` の `analyze_occlusion`
    ステップ（`patch_sizes` に複数値を指定）を使う。
  - **解析対象画像のメタデータ保存**: `FontDataModule(mode="scorer")` の `_MixedFontDataset` は
    `image_path`（元画像ファイルの絶対パス）も返すようになっており、どのサンプルを解析したか
    事後追跡できる。各 `<char>_<real|gen>_<i>/` に `target_info.yaml`
    （char/label/font_family/image_path/split/pick_index/s0_score/s0_logit）を保存し、
    `occlusion_scores.csv` のヘッダコメントにも font_family/image_path を追記する。実行全体では
    `<output_dir>/targets_manifest.csv`（全 pick の一覧、列は target_info.yaml と同じ + output_dir
    + `s0_score`/`s0_logit`/`s_blank_score`/`s_blank_logit`/`s_fill_score`/`s_fill_logit`/
    `keep_score_max`/`keep_score_min`/`deletion_auc`/`insertion_auc`、後者 6 つは
    `--keep`/`--curves` 有効時のみ）を出力し、
    `--all-chars` の大量実行でも grep/pandas で特定サンプルを引ける他、モデル間・文字間で
    deletion/insertion AUC を直接比較できる。
  - **per-patch 絶対スコア掃引**: `build_occluded_batch` で全パッチ位置の遮蔽画像を自前構築し
    （`occlusion_sensitivity` と同一のパッチ幾何で座標が一致）、`forward_logits` で各遮蔽画像の
    スコア・logit を個別に取得する。`score_sweep.png`（入力／絶対スコアヒートマップ／既存 Δ 発散
    マップ）と `occlusion_scores.csv`（top, left, score, logit, delta, 任意で probe の p_real）を出力。
  - **背景 no-op パッチの除外（`--min-change`、既定 0.02）**: 背景だけを覆い遮蔽しても入力が実質
    変わらないパッチ（＝スコアも動かない）を解析対象から外す。ストロークが消えたときの挙動だけを
    見たいときに有用。しきい値は入力のダイナミックレンジ (max-min) に対する割合で、パッチ領域内の
    `|img - fill|` の最大値がこれ未満の位置を落とす（`build_occluded_batch(min_change=...)`）。raw SDF
    でも binary でも表現に依らず効く。`--min-change 0` で全パッチを解析（従来挙動）。全パッチが除外
    されると（空バッチ回避のため）フィルタは自動で無効化される。除外後の残パッチ数はログに出す。
  - **特徴空間写像**: `extract_intermediate` で各遮蔽バリアントの `--embedding-layer`（既定 layer4）
    GAP 特徴を取得し、`--features`（`extract_scorer_features.py` 出力 parquet・同一 checkpoint 由来・
    必須）で読む real/gen 母集団特徴と**連結して `fit_combined_embedding` を一度だけ fit**する
    （同関数は fit_transform 一発で reducer を破棄し `.transform()` で新規点を射影できないため、
    母集団+原画像+遮蔽点を毎回まとめて re-fit するのが tsne を含む全 reducer で動く唯一の方法。
    母集団は `--pop-samples`（既定 5000）で事前にサブサンプルしてから `max_samples=-1` で渡し、
    遮蔽点が間引きで落ちないようにする。tsne/umap が重い場合は `--pop-samples` を下げる）。
    **母集団は既定でターゲット画像と同じ文字に char-matched**（'H' を解析するなら背景も 'H' の
    real/gen のみ）。全文字混在の背景より「遮蔽点がどちらのクラスへ動くか」を読みやすい。
    `--pop-char all` で全文字に、`--pop-char A` で特定文字に上書きできる。probe（p_real）も
    char-matched な母集団 train split で fit する。出力する特徴空間プロットは 3 枚:
    `feature_embedding.png`（母集団を real/gen 淡色ドット、遮蔽点をスコアで連続配色）、
    `feature_embedding_byscore.png`（**母集団も遮蔽点も同一の RdYlGn スコア連続色**）、
    `feature_embedding_bypos.png`（遮蔽点をパッチ位置 (top,left) で配色、空間位置↔特徴空間の
    移動先の対応を見る）。既定 reducer は **pca**（微小摂動→微小移動を決定論的・線形に
    見るため、`analyze_scorer_features.py` の既定 tsne とは異なる）。
  - **遮蔽画像のスコア順グリッド**: 遮蔽バリアントを occ score 降順に並べ、`top`（高スコア=隠しても
    realness が落ちない領域）/ `middle`（中央値付近）/ `bottom`（低スコア=realness を最も支える領域）の
    3 帯それぞれ `--grid-n`²（既定 6²=36）枚を `occluded_ranked_{top,middle,bottom}.png` に N×N
    グリッド保存（各セルに `s={score} (top,left)`）。**セル 0 は常に遮蔽なしの原画像**（青枠、
    ラベル `original s0=...`、パッチ枠なし）で固定するため、帯の実バリアントは `grid_n²-1` 枚。
    同一図内で元画像と直接比較できる。埋め込み図では遮蔽画像を間引いてしか見られないため、
    実際の遮蔽画像そのものをスコア帯ごとに一覧できる。
    **描画はモデル入力そのものの解像度**（96px 等）で行い、`imshow(interpolation="nearest")` で
    画素境界を保つ。以前は埋め込みプロット用の 28px サムネイルを流用していたが、`--patch 8` は
    28px 上でわずか 2.3px・実フォントのストローク幅 ~4px は 1.2px に潰れ（`mode="area"` の面積平均で
    周囲のインクが混ざり戻る）、遮蔽の有無が視覚的に判別できなかった。グリッドのセルは
    ~210 device px あるので、フル解像度を渡すのが正しい（`_to_gray` と `_make_thumbnails` を分離）。
    各セルには遮蔽パッチ位置を赤枠（`Rectangle`）で明示する（フル解像度座標なので**画素境界と厳密一致**）。
    さらに既定の `--grid-style diff` では、**原画像との差分画素（＝そのパッチが実際に消したインク）を
    赤で塗る**。baseline=`background` の遮蔽は背景に当たると入力が実質変わらないため、赤枠だけでは
    「そのパッチが何を壊したのか」が読めず、スコアが大きく違う 2 セルが同じ絵に見えてしまう
    （`--min-change` の no-op パッチ判定と同じ問題意識の可視化版）。判定しきい値は入力の
    ダイナミックレンジ比 `_TINT_REL_TOL`（2%）で、raw SDF でも binary でも表現に依らず効く。
    色の意味は図タイトル禁止の規約に従い凡例で示す。`--grid-style plain` で従来どおりの
    純グレースケール（赤枠のみ）に戻せる。
  - **preservation（パッチだけ残す、`--keep`、既定 on）**: `build_occluded_batch(mode="keep")`
    で、パッチ領域だけ原画像を残し他を全面 baseline で塗った画像を作る。deletion が
    「そこを壊すと realness が落ちるか＝必要性」を測るのに対し、preservation は「そこ**だけ**で
    realness が出るか＝十分性」を測る対になる指標。`min_change` フィルタも deletion と同一条件で
    適用されるため `positions` は deletion と完全一致し、per-patch で突き合わせられる
    （`build_occluded_batch` の docstring 参照）。
    **分布外性への対処**: `--patch 8` 等で「パッチだけ残す」と 96px のほぼ空白キャンバスに
    小さな断片が乗った学習分布外の入力になり、素の `s(keep)` は断片の realness ではなく
    「ほぼ空白であること」に支配されやすい。そこで**白紙**（全面 1.0）のスコア `s_blank` と
    **黒紙**（全面 0.0）のスコア `s_fill`（`make_blank_like`、各 forward 1 回、
    `_WHITE_VALUE`/`_BLACK_VALUE`）という両極端の基準点を取り、`keep_delta = s(keep) - s_blank`
    で読む（`occlusion_scores.csv` の `keep_score`/`keep_logit`/`keep_delta` 列、ヘッダコメントの
    `s_blank_score`/`s_blank_logit`/`s_fill_score`/`s_fill_logit`）。**この 1.0/0.0 は `--baseline`
    （パッチの塗り値、既定 background 自動推定）とは独立**で、`sdf_decode=hard`/`smoothstep` や
    素の二値画像の規約（`sdf.py` の `sdf_to_smoothstep`: `stroke=0, background=1`）に合わせて
    固定している。raw SDF（`sdf_decode=None`）はこの極性が逆（stroke=1.0/background=0.0）になる
    ため、その場合 `s_blank`/`s_fill` の意味が反転する点に注意。出力: `keep_sweep.png`
    （`_plot_score_sweep` を `score_label`/`delta_label`/`ref_label` で preservation 用に流用、
    基準は s0 ではなく s_blank）、`kept_ranked_{top,bottom}.png`（`_plot_occluded_ranked` を
    `file_prefix="kept_ranked"`・`tint=False`・`box_label="kept"` で流用。**diff 着色はしない**:
    遮蔽領域外が全部「変化した画素」になり画面が赤で埋まるため。赤枠の意味も「消した」→
    「残した」に変わる）、`necessity_vs_sufficiency.png`（x=deletion 後のスコア・y=preservation
    スコアの散布図、点色は空間位置、`s0`/`s_blank(white)`/`s_fill(black)` の基準線付き。
    preservation スコアが `s_blank`/`s_fill` と大差なければ、そのパッチ単独の寄与は白紙・黒紙という
    無内容な入力と区別できていない＝十分性が低いことを意味する。左上=必要かつ十分／左下=必要だが
    単独では不十分／右上=冗長だが十分／右下=無関係、を軸ラベルで示す）。`--no-keep` で全て無効化。
    - **`kept_ranked_{top,bottom}.png` の白紙・黒紙参照セル（`--grid-refs`、既定 on）**:
      セル 0（原画像、s0 基準）の直後に、白紙（`s_blank`、灰枠）と黒紙（`s_fill`、橙枠）を
      同じグリッドに並べる。preservation の正しい基準は s0 ではなく s_blank であり、
      「s=0.93 の断片が白紙 s_blank=0.94 と区別できていない」という誤読しやすい事実を
      グリッド図の中で直接確認できるようにする（`_plot_occluded_ranked` の `ref_cells`
      引数、`_occluded_grid` の `frame_legend` で複数の枠色ハンドルを凡例に出す）。
      帯の実バリアントは `grid_n**2 - 3` 枚になる。`occluded_ranked_*.png`（deletion 側）
      には入れない: セルがほぼ全画像で白紙・黒紙とは regime が違いすぎ、63 枠中 2 枠を
      消費する価値が薄いため。
    - **`necessity_sufficiency.png`（`--keep` に連動、既定 on）**: necessity（`delta`）と
      sufficiency（`keep_delta`）を 1 枚の 2x3 ダッシュボードに統合する。
      row0=[input, necessity マップ（`delta` の発散マップ）, sufficiency マップ
      （`keep_delta` の発散マップ）]、row1=[**4 色カテゴリカルマップ**, 4 色凡例
      （2x2 quadrant）, 散布図（joint と同配色の散布図）]。
      配色（`_necessity_sufficiency_colors`）は **delta/keep_delta の符号だけ**で
      4 値に分類したフラット色（強弱の階調なし）: 紫=必要かつ十分（`delta>0` かつ
      `keep_delta>0`）／赤=必要のみ／青=十分のみ／灰=どちらでもない。0 は
      「occluding/keeping が絶対的に良い方向に効くか」という符号そのものなので
      **画像間でも比較可能な絶対基準**（当初の連続 bivariate 配色は強弱が読みにくく
      画像間比較もできなかったため、ユーザ指摘を受けてこの 4 色分類に単純化した）。
      joint パネルはグリフに重畳しない（色が黒インクと衝突して読めなくなるため）。
      代わりに 0.5 レベルセットの輪郭線だけ黒で重ねる（`ax.contour`）。
      散布図は既存 `necessity_vs_sufficiency.png`（点色=空間位置）と軸は同じだが、
      **点色を joint と同じ 4 色分類**にすることで空間マップの色分けと散布図の
      色分けを相互参照できる（既存の空間位置配色図はそのまま残す）。
  - **deletion / insertion 曲線（`--curves`、既定 on）**: パッチを `delta`
    **降順**（`|delta|` ではない。正=occluding でスコアが下がる=realness を支える方から）に
    並べ、その順で領域を累積和集合しながら（1) deletion: 原画像から順に**白紙**（`_WHITE_VALUE`）で
    消す、(2) insertion: 全面白紙から順に原画像へ戻す、の 2 系列の絶対スコア推移を測る
    （`_cumulative_curve_batches`、累積画像を一括バッチ forward）。「消す」を `--baseline`
    自動推定ではなく固定の白紙にしているのは、`s_blank` を基準とした境界条件の一貫性のため
    （後述）。画像 1 枚が `deletion_auc`/`insertion_auc` という 1 スカラに落ちるため、
    RankNet と BCE 分類器の判断根拠の鋭さを定量比較できる（`targets_manifest.csv` に集約、後述）。
    x 軸は「塗り替えた面積の割合」（パッチのタイル幾何は全面を覆うため終端で 1.0 に達する。
    ただし `--min-change` で除外された背景パッチは元々 fill とほぼ同一なので、この近似は
    `--min-change 0` でのみ厳密に成り立つ）。境界条件: deletion は `s0` から始まり `s_blank`
    近傍で終わる、insertion は `s_blank` から始まり `s0` 近傍で終わる。出力:
    `deletion_insertion.png`（2 曲線 + AUC を凡例に）、`curve_metrics.csv`
    （1 行: `deletion_auc, insertion_auc, s0_score, s_blank_score, s_fill_score, n_patches`）。
    `--no-curves` で無効化。
  - **散布図上に実画像サムネイルを重畳**: 原画像（黒枠・大きめ、`--thumb-zoom` の 1.6 倍）と
    遮蔽バリアント（`feature_embedding.png` はスコア色・`feature_embedding_bypos.png` は位置色の
    枠）の両方に対応するグリフ画像を表示する。全遮蔽点（stride 次第で最大 ~500 点）に貼ると
    密な格子状で重なるため、埋め込み座標を空間グリッドでビニングし 1 セル 1 点まで間引いて
    `--max-occ-thumbs`（既定 24）件に抑える（間引かれた点も色付きドットとしては残る）。
    `--thumb-size`（サムネイル元解像度、既定 `0`＝リサイズせずモデル入力の解像度を使う）/
    `--thumb-zoom`（表示倍率、既定 0.2）で調整可。`OffsetImage` の表示サイズは
    `thumb_size × zoom × dpi/72` px なので、この 2 つは**セットで変える**（既定は旧設定
    `28 × 0.7` と同じ見た目の大きさを保ったまま元解像度だけを上げてある。28px を 40.8 device px へ
    拡大していた分が純損失だったため）。**この縮小はスコア順グリッドとは無関係**（グリッドは
    常にフル解像度、上記参照）。
    **既知の制約**: 遮蔽点同士が特徴空間で近接している場合（layer4 の粗さ、微小パッチ遮蔽など）、
    ビニングで間引いても固定ピクセルサイズのサムネイル同士が画面上で重なることがある
    （データ空間の距離が小さいことと表示サイズが固定であることの必然的なトレードオフ）。
    `--thumb-zoom` を下げる、`--patch`/`--stride` を粗くする、より細かくない layer を選ぶ、などで緩和する。
  - **layer の粗さに注意**: layer4 は空間的に粗い（96px×resnet50 で GAP 前 ~3×3）ため、小パッチ
    遮蔽では layer4 の GAP 特徴がほぼ動かず遮蔽点が原画像点に重なりうる。`--embedding-layer` で
    layer2/3 も試すこと。

### Scorer の周波数解析（FFT）

occlusion / Grad-CAM が空間領域の判断根拠可視化なのに対し、`analyze_frequency.py` は**周波数領域**
から「real/gen に周波数的な違いがあるか」「その違いは realness score と相関するか」を調べる。
`extract_scorer_features.py` の parquet（メタデータ + `inputs/<split>/*.png` モデル入力サムネイル）
だけを読む純 CPU 解析で、checkpoint・config・モデルロードは一切不要（`--features` のみ必須）。
実装は `analysis/spectral_analysis.py`（純関数）+ `scripts/analyze_frequency.py`（CLI ドライバ）の
2 層構成（`feature_analysis.py` と同じ「parquet 契約」に従い、`REAL_COLOR`/`GEN_COLOR`/`_fs`/
`_legend`/`_load_thumb`/`_simple_grid`/`fit_linear_probe`/`layer_probe_auroc` を再利用する）。

**入力表現に関する重要な注意**: 解析対象はモデルの**デコード後の入力**（`sdf_decode: hard` の
学習済み Scorer なら real/gen とも 96px の二値画像）であり、`images/`（real=アンチエイリアス／
gen=完全二値）や `sdf/`（real=EDT 量子化／gen=拡散モデルの連続場）のようなレンダリングパイプライン
由来の非対称性は入力段で意図的に潰されている（「事前計算済み SDF」節参照）。したがってこの解析が
検出しうるのは**二値化後に残る幾何的な差**（ストローク太さ・曲率・端の形状・しきい値でのギザつき）の
周波数的な現れであり、アンチエイリアスのアーティファクトそのものではない。生 `images/`/`sdf/` の
解析は parquet に元画像パスが無い（`image_path` はサムネの相対パス）ためスコープ外
（`FontDataModule` 経由で別途読む必要がある）。

**最大の交絡はインク被覆率（`ink_fraction`）**: ストローク太さが違うだけでスペクトル形状は動く。
そのため常に `ink_only`（`ink_fraction` 単独）を probe のベースラインとして併記し、`radial`
（半径プロファイル全体）が `ink_only` と同程度の AUROC なら「周波数の違い」ではなくインク量の
違いだと判断する。判定は p 値ではなく **Cohen's d と AUROC** で行う（test split は数万枚あり
p 値は自明に有意になるため、`stats/metric_separation.csv` の効果量を見る）。

- **特徴量計算**（`spectral_features()`）: 画像平均を引いてから Hann 窓を掛けて 2D FFT
  （`image_power_spectrum`、WHY: DC 項の支配とゼロパディング境界のリーク十字を抑えるため）、
  半径方向プロファイル `r000..r047`（96px の Nyquist 半径 48）と角度方向プロファイル `a00..a17`
  （10 度刻み、ストローク方向の異方性）を算出。要約スカラは `ink_fraction`（インク被覆率、
  交絡の統制変数）、`band_low/mid/high`（k<8/8≤k<20/k≥20 の正規化エネルギー）、`hf_ratio`
  （high/(low+mid)、見出し指標）、`centroid`（パワー重心周波数）、`slope`（log-log 傾き、
  k∈[2,40] 線形回帰）、`aniso_index`・`hv_ratio`（角度異方性）。既定 `--normalize per-image`
  はプロファイルを画像ごとに総和 1 へ正規化してから比較する（インク量の多い画像がプロファイル平均を
  支配しないようにするため。`--normalize none` で生の平均も出せる）。
- **プロファイル比較**（`profiles/`）: `radial_profile[_<split>].png` は上段に real/gen 平均±95%CI
  （log-y）、**下段に gen/real 比（log-y）**を出す。平均の重ね描きはこのサンプル数だと目視で
  ほぼ一致して見えるため、比プロットが微小な系統差を可視化する要。`radial_profile_by_char_<split>.png`
  は文字別の小倍数図。`angular_profile[_<split>].png` は極座標。`profile_by_score_bin_{real,gen}_
  <split>.png` は**クラス内**スコア四分位ごとの平均プロファイル（クラス間で切ると real/gen ラベル差を
  スコア差と取り違えるため、必ずクラス内で切る）。
- **分布・分離統計**（`distributions/` / `stats/`）: `{metric}_dist_<split>.png` は各スカラの
  real/gen 重ねヒストグラム。`stats/metric_separation.csv` は各スカラの
  `real_mean, gen_mean, cohens_d, auroc, n_real, n_gen`。`stats/per_char_spectral_auroc.csv` は
  半径プロファイル probe の文字別 AUROC。`stats/score_metric_correlations.csv` は score とスカラの
  Spearman ρ（全体/real内/gen内）に加え、`ink_fraction` を線形回帰で除去した残差同士の相関
  `partial_rho_ink_controlled`（インク統制後の偏相関）。
- **線形プローブ + ROC**（`probe/`）: `FEATURE_SETS`（`ink_only`/`radial`/`angular`/`scalars`/`all`）
  ごとに train split で `fit_linear_probe` → 指定 split で AUROC・ROC。`roc_spectral_<split>.png`
  に **Scorer 本体の ROC（parquet の `score` 列から算出）を黒線で重畳**し、「周波数だけでどこまで
  分離できるか」と「Scorer 本体」を 1 枚で比較する見出し図。`probe_results.csv` に
  `feature_set, split, auroc`。
- **スコアと周波数の関係**（`scatter/`）: `score_vs_{metric}_<split>.png`（散布図 + Spearman ρ を
  凡例に表示）、`score_vs_spectral_probe_<split>.png`（`radial` probe の p(real) vs Scorer の
  score、両者が独立な判断根拠なら相関は弱いはず）。
- **画像グリッド**（`images/`）: `--metric`（既定 `hf_ratio`）でソートした real/gen 別
  top/bottom N×N グリッド `ranked_{real,gen}_{top,bottom}_<metric>_<split>.png`（`--grid-n` 既定 8、
  `--no-cell-labels` でラベル無し純画像グリッドに）。
- **出力・再現性**: `spectral_features.parquet`（1 画像 1 行、計算 1 回→再解析多数回。既存があれば
  再利用、`--recompute` で強制再計算）、`config.yaml`/`run_info.yaml` は他の解析スクリプトと同じ
  規約。`--split`（特徴量計算対象、既定 `all`）と `--stats-split`（プロット対象、既定 `test`）は別軸。
  `--char`/`--n-per-char` で対象を絞れる（省略時は charset 全文字・全枚数）。

複数チェックポイント一括実行は `run_eval_suite.sh` の `analyze_frequency` ステップを使う
（「Scorer 評価スイートの一括実行」節を参照）。単発実行は `analyze_frequency.py` を直接呼ぶ。

### Scorer の backbone 選択

`RealnessScorer` / `ScorerModule` の `model.backbone` で torchvision の ResNet 系 backbone
（`resnet18`（既定）/ `resnet50`）、または ViT backbone（`vit_backbone.VIT_SPECS`、下記）を選べる。
head の入力次元は ResNet では `backbone.fc.in_features`、ViT では `net.hidden_dim` から自動取得する
ため、次元の手書きは不要。既定が `resnet18` なので既存 config・チェックポイントは後方互換。

**対応 ResNet backbone の追加**: `src/real_fonts_vs_gen_fonts/models/scorer.py` の `_BACKBONES` に
`"name": (constructor, default_weights_enum)` を 1 行足すだけ。`extract_intermediate`
（`conv1/layer1..4` を直参照）と `GradCAM`（`backbone.layer1`〜`layer4` を `getattr` で選択）は torchvision の **ResNet 属性
レイアウト**前提なので、ResNet-18/34/50/101/152・ResNeXt・WideResNet は無改造で動く。ConvNeXt 等
それ以外の非 ResNet 系はこれらが動かないため対象外。

### Scorer の ViT backbone

`src/real_fonts_vs_gen_fonts/models/vit_backbone.py` の `VIT_SPECS`（`vit_tiny_p8` / `vit_small_p8`
（既定候補）/ `vit_small_p16`）で `torchvision.models.VisionTransformer` をスクラッチ構築する
（全 scorer config が `pretrained: false` のため、224px 固定・ImageNet 事前学習という通常の
制約は問題にならず、`timm` 等の追加依存も不要）。`RealnessScorer(backbone="vit_small_p8",
image_size=96, ...)` のように `image_size` を渡す（`ScorerModule` 経由なら
`train_scorer.py`/`train_scorer_classifier.py` が `data.image_size` から自動注入、
`lr_total_steps` と同じ「YAML に二重管理しない」パターン）。`image_size` が `patch_size` で
割り切れないと `ValueError`（暗黙の切り捨てでトークングリッドが崩れるのを防ぐため）。

**採用理由**: 「Attention で判断根拠が分かる」「大域依存を捉えられる」は単体では弱い論拠
（attention weight が feature importance の忠実な説明にならないことは Jain & Wallace 2019 /
Serrano & Smith 2019 / Wiegreffe & Pinter 2019 で繰り返し示されており、96px 中背景が大半の
入力では ViT の *register* 現象（Darcet et al. 2024、背景トークンが高ノルム外れ値になり attention
を吸う）も懸念される。また 96px 入力への ResNet50 の受容野は既に画像全体を超えており CNN で
大域依存が取れていない証拠もない）。本当の狙いは **`patch_size=8`・`image_size=96` で
12×12=144 トークンにし、`scripts/analyze_occlusion.py --patch 8 --stride 8` の occlusion パッチ
幾何と厳密に一致させること**。これにより attention 系 attribution を occlusion sensitivity
（因果的な基準）とパッチ単位で直接突き合わせ、**忠実性を定量検証**できる
（`scripts/analyze_attention.py`、次節）。

**parquet 契約との互換**: `extract_intermediate` は ViT でも `layer1`〜`layer4` の 4 キーを返す
（`vit_backbone.stage_tokens` が `num_layers` を 4 等分したブロック出力を返し、CLS を除いた
patch トークンを平均する。ResNet のような段ごとの次元拡大はなく全層 `feature_dim` で共通）。
そのため `extract_scorer_features.py`・`analyze_scorer_features.py`・
`analyze_occlusion.py --embedding-layer`・`analyze_frequency.py` は **ViT でも無改造で動く**
（実機で `l1_`〜`l4_` 各 384 列 = `vit_small_p8` の `hidden_dim` を確認済み）。

**GradCAM は ViT 非対応**: `GradCAM.__init__` は ViT backbone を渡すと明示的に `ValueError`
（`backbone.layer1..4` の ResNet 属性を前提とするため）。ViT の判断根拠可視化は
`models/attention_maps.py` / `scripts/analyze_attention.py` を使う。

### Scorer の attention 忠実性検証（ViT のみ）

`models/attention_maps.py` は ViT Scorer の判断根拠マップを 5 手法で計算する。すべて CLS を
除いた `(B, N_patches)` を返す:

| 関数 | 手法 | 備考 |
|---|---|---|
| `cls_attention` | 最終ブロックの CLS→patch attention（ヘッド平均） | 素朴版 |
| `attention_rollout` | Abnar & Zuidema 2020（`0.5A+0.5I` を行正規化して層方向に積） | |
| `chefer_relevance` | Chefer et al. 2021（`E_h[(∇A⊙A)^+]` の rollout） | 勾配併用の最有力候補 |
| `token_grad_cam` | トークン版 Grad-CAM（`GradCAM` の CNN 版と同じ重み付け） | attention 非依存の対照。CNN 版同様、未学習/弱学習モデルでは重み付き和が全トークンで非正になり ReLU で全 0 に潰れうる（バグではなく手法の性質） |
| `token_norm_map` | パッチトークンの L2 ノルム | **register 診断用**。背景に高ノルム外れ値が集中していれば attention 系マップの解釈は疑ってかかる |

**`vit_backbone.capture_attention` の実装上の注意（重要）**: 当初 `nn.MultiheadAttention.forward`
を `need_weights=True` で呼び、返り値の attention weight にそのまま `retain_grad()` する実装だった
が、**このアプローチは勾配が常に 0 になる不具合があった**。torchvision の `MultiheadAttention` は
実際の出力を融合 SDPA カーネルで計算し、`need_weights=True` 時に返す `attn_output_weights` は
**報告用に別経路で計算された、実際の出力とは計算グラフ上切り離れたテンソル**（`grad_fn` は持つが
`.grad` が常に `None`）だったため（`chefer_relevance`/`token_grad_cam` が全 0 マップを返す形で発覚、
`capture_attention` 内で直接 `retain_grad()`→`backward()` して検証済み）。対策として
`capture_attention` は `in_proj_weight`/`in_proj_bias`/`out_proj` を使い scaled dot-product
attention を**手動で再実装**し、返す attention weight を実際の出力計算経路上に置くことで勾配を
通す（`attention_dropout` は全 config で 0.0 のため省略している）。

`scripts/analyze_attention.py`（`--config` は学習済み ViT Scorer の `config.yaml`。単発実行は
直接呼ぶか `run_eval_suite.sh` の `analyze_attention` ステップ経由。後者は非ViT backbone の
チェックポイントに対しては自動スキップされる）は上記 5 手法を
`models/occlusion.py` の occlusion sensitivity（実際にパッチを遮蔽してスコア変化を測る、
**因果的**な基準）と同一の 12×12 グリッド上で突き合わせる。`--patch`/`--stride` は
指定しない: `model.backbone` から解決した ViT の `patch_size` に固定し、`min_change=0`
（トークングリッドとの行数一致が前提のため、背景 no-op パッチも除外しない）。

- **一致度**: 各手法 vs `occlusion_delta`（`s0 - s_occ`）の Spearman ρ・Kendall τ・
  top-k overlap（k=10, 20）。
- **忠実性**: 各手法の attribution 降順で `models/occlusion.py` の
  `cumulative_curve_batches`（`analyze_occlusion.py` の deletion/insertion 曲線と共通処理に
  リファクタ済み）を回し、deletion/insertion AUC を算出。**`occlusion_delta` はこの目的関数
  （累積遮蔽時のスコア変化）に対する貪欲順序そのものなので上限リファレンスであり競合手法では
  ない**。`random`（一様乱数順）を下限として併記する。
- 出力: `attribution_maps.png`（入力 + 各手法のヒートマップ + occlusion delta）、
  `faithfulness_curves.png`（手法別 deletion/insertion 曲線）、
  `attribution_agreement.csv` / `faithfulness.csv` / `attribution_maps.csv`（1 画像単位）、
  実行全体で `faithfulness_summary.csv`（手法別 deletion/insertion AUC の平均±std）。

### Scorer 評価スイートの一括実行

`scripts/run_eval_suite.py`（`run_eval_suite.sh`、`config/eval/eval_suite.yaml`）は、複数の
学習済み Scorer チェックポイント横断で評価・解析一式（`extract_scorer_features.py` →
`analyze_scorer_features.py` / `analyze_frequency.py` / `analyze_occlusion.py` →
`visualize_occlusion.py` / `visualize_gradcam.py` / `analyze_attention.py`）を1コマンドで
実行するオーケストレータ。各スクリプトの内部ロジックには一切触れず、subprocess として呼び出す
実行フロー層に徹する（設定の真実は各スクリプト自身の argparse のまま）。

- **YAML スキーマ**: `output_root`（結果の出力先）、`checkpoints`（`name`/`checkpoint`/`config`/
  `use_ema` の任意個数リスト）、`steps`（ステップ名 → `enabled` + そのステップの CLI フラグに
  対応するパラメータ）。`analyze_occlusion.patch_sizes` はリストで、要素ごとに 1 回ずつ
  `analyze_occlusion.py` を呼ぶ（`patch=stride` で非重複、出力は `occlusion_batch/patch{N}/`）。
- **依存順序**: `extract_features` が生成した parquet を `analyze_features`/`analyze_frequency`/
  `analyze_occlusion` が消費する。`extract_features` が無効化/失敗している場合、これらは
  `skipped`（`extract_features did not produce a parquet`）としてマニフェストに記録され、
  処理は継続する。
- **backbone による自動スキップ**: チェックポイントごとに自身の `config.yaml` の
  `model.backbone` を読み、`models/vit_backbone.VIT_SPECS` に含まれるかで ViT/非ViT を判定する。
  `visualize_gradcam` は ViT なら skip（`GradCAM` が ValueError を出すため）、`analyze_attention`
  は非ViT なら skip（`analyze_attention.py` 自身が ValueError を出すため）。
- **実行結果**: `<output_root>/manifest.csv`（列: `checkpoint_name, step, status(ok/error/skipped),
  skip_reason, duration_s, output_dir, returncode`）に全ステップの結果を記録。デフォルトは
  continue-on-error（1 ステップ失敗しても後続は実行）、`--stop-on-error` で即中断できる。
  `<output_root>/config.yaml`/`run_info.yaml` はオーケストレータ自身の再現性情報。
- **CLI**: `--checkpoints name1,name2` で対象チェックポイントを、`--steps step1,step2` で
  対象ステップを絞り込める。位置引数の `key=value` オーバーライドは `eval_suite.yaml` の
  **トップレベルキーのみ**に適用され（例: `output_root=...`）、各スクリプトへは転送しない
  （ステップ別パラメータは YAML 直接編集が真実、という規約を維持するため）。
- 単一チェックポイントに対するアドホックな単発実行には、引き続き `run_analyze_occlusion.sh` /
  `run_visualize_occlusion.sh` / `run_visualize_gradcam.sh` / `run_extract_scorer_features.sh` を
  直接使う（`run_eval_suite.py` はこれらの `.sh` ではなく対応する `scripts/*.py` を直接
  subprocess 実行する）。

### Scorer の head 構成

`RealnessScorer` / `ScorerModule` の `hidden_dim` は head の中間層次元数。`None` を指定すると
`Linear(feature_dim, hidden_dim) → ReLU → Dropout → Linear(hidden_dim, 1)` の 2 層 MLP ではなく、
`Linear(feature_dim, 1)` の 1 層線形ヘッドになる（`dropout` は未使用）。`feature_dim` は backbone 依存。

### Scorer のベーシック構成（head-overfitting 仮説の検証用ベースライン）

`config/scorer/scorer_basic.yaml`（RankNet）/ `config/scorer/scorer_classifier_basic.yaml`（BCE）は、
`analyze_scorer_features.py` の layer4 線形プローブ（sklearn LogisticRegression、test AUROC≈0.9）
と比較するための最小構成。`charset="H"`・augmentationなし・`hidden_dim: null`（線形ヘッド）・
（RankNet版は）`tie_weight_real`/`tie_weight_gen: 0.0` で揃え、「ヘッド構成」1変数のみを
切り出して評価する。

## コーディング規則

### 全般

- 型ヒントを必ず付ける (`x: torch.Tensor`, `-> dict[str, float]`)
- `nn.Module` のサブクラスは `__init__` と `forward` のみで完結させる
- マジックナンバーは定数か設定ファイルに切り出す
- シードは `lightning.seed_everything(seed, workers=True)` で固定する
- **コード内の文字列リテラル（エラーメッセージ・dict キー・定数値など）は英語で書く**

### データ

- `Dataset.__getitem__` は `dict[str, Any]` を返す (キー: `"image"`, `"label"` 等)
- 前処理と augmentation は `torchvision.transforms.v2` で定義し、`Dataset` に渡す
- DataLoader の `num_workers` は CPU コア数に合わせ、`pin_memory=True` を GPU 時に使う
- 実フォント画像は [font2dataset](https://github.com/mskakb/font2dataset) で生成する
- `FontDataModule(charset="uppercase")` で文字種を切り替える。利用可能: `uppercase` / `lowercase` / `digits` / `ascii`

### モデル

- `nn.Module` は `src/real_fonts_vs_gen_fonts/models/` に置く
- `LightningModule` ラッパーは `*_module.py` として `nn.Module` と分離して定義する
- `LightningModule.__init__` では必ず `self.save_hyperparameters()` を呼ぶ
- 推論時は必ず `model.eval()` と `torch.no_grad()` を使う

### 学習スクリプト

- 設定は YAML ファイルで管理し、CLI で `section.key=value` 形式で上書きできる
- CLI 引数は argparse で統一: `--config <yaml>`, `--resume <ckpt>` (train_*.py) / `--checkpoint <ckpt>` (generate_fonts.py)
- 共通の設定読み込み・上書き処理は `src/real_fonts_vs_gen_fonts/utils/config.py` の `load_config()` を使う
- チェックポイントからの再開は `--resume <path>` で指定する（重み・オプティマイザ・スケジューラ・global_step をすべて復元）
- `max_steps` を増やして再学習する場合は `--weights-only <path>` を使う（重みのみ復元、スケジューラは新 `max_steps` でリセット）
- 学習中のサンプル可視化は `sample_every_n_val_checks`、FID 計算は `fid_every_n_val_checks` で頻度を指定し TensorBoard にログする（`val_check_interval` 単位でカウント）
- サンプリング（`_log_samples` / `_log_fid`）と InceptionV3 特徴抽出の推論ミニバッチは、学習ミニバッチサイズ（`data.batch_size`）を上限にチャンク分割して逐次実行する。`fid_n_samples`（デフォルト 256）を一括生成すると学習時よりはるかに大きい VRAM を消費し検証時 OOM の原因になるため。総生成枚数・生成内容は不変（各サンプル独立で数値的に等価）
- FID は **grayscale 画像空間**で計算する。`use_sdf=true` のとき real/gen とも `_sdf_decode` で grayscale（binary/smoothstep）に decode してから InceptionV3 特徴を抽出する（gen は `_log_fid`、real は `_get_real_features` で同一の decode 経路を通す）。real を SDF 生のまま比較すると表現空間が gen とずれて FID が無意味になる。`use_sdf=false` のとき `_sdf_decode` は恒等写像
- `val_check_interval`: **int はステップ数単位、float はエポック単位**（両スクリプトともデフォルト `1.0`。データセットが大きく 1 エポックが長い場合は int 値に変更して頻度を上げる）
- `limit_val_batches`: 両スクリプトとも `trainer_cfg.get("limit_val_batches", 1.0)` で対応。デバッグ時は `trainer.limit_val_batches=10` で上書き
- 学習スクリプトは冒頭で `setup_logging()`（`utils/config.py`）を呼び、INFO ログを stdout に出す。本プロジェクトはロギング未設定でモジュールロガーの `log.info` がデフォルトで握り潰されるため、これを呼ばないと split サマリ等が表示されない。`FontDataModule` は `setup()` 開始時に使用設定（mode・charset・batch_size・SDF 設定・augmentation・データパス等）を `FontDataModule config:` としてログ出力する

### 実験管理

- 実験名（`experiment_name`）にはハイパーパラメータの要約を含める（`diffusion_lr1e-4` など）。
  **日付は `setup_experiment_dir()` が `YYYY-MM-DD-HH-MM_` 接頭辞として自動付与**するため
  YAML 側には書かない（出力ディレクトリは `<日時>_<experiment_name>[_<comment>]`）
- `outputs/<experiment_name>/` に全成果物をまとめる（`output_root` で変更可）
- **config・スクリプト・notebook に個人環境の絶対パス（`/home/...`）を書かない**（公開リポジトリのため）。
  リポジトリルート相対の `data/datasets/`（実データ）・`models/`（チェックポイント）・`results/`
  （生成・解析結果）を使う。これらは gitignore 済みで、ローカルでは外部ストレージへのシンボリックリンク。
  スクリプトはリポジトリルートから実行する前提
- `.memo/`（研究メモ・`research_draft.md`）はローカル専用で git 管理外
- 結果の再現に必要な設定は全て `config/` に YAML で保存する
- **全スクリプトは実行直後に出力ディレクトリへ以下を保存すること（再現性の担保）**
  - `config.yaml`: マージ済み設定（CLI overrides 反映済み）＋ `--checkpoint` 等の argparse 引数
  - `run_info.yaml`: `sys.argv`・タイムスタンプ・git commit hash（`save_run_info()` で自動生成）
- `save_run_info()` は `src/real_fonts_vs_gen_fonts/utils/config.py` に実装。新規スクリプト追加時も必ず呼ぶ

### ドキュメント更新ルール

コードや設計を変更した際は必ず以下を更新する：

- コーディング規則・環境・使い方・アーキテクチャの変更 → **CLAUDE.md**
- 研究設計・データ戦略・実験計画・システム設計の変更 → **`.memo/research_draft.md`**

### コメント

- WHY が非自明な箇所のみコメントを書く（WHAT は書かない）
- 実験ノートはコードではなく `outputs/<experiment_name>/README.md` に記述する
