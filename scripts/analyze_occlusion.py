"""Occlusion sensitivity のさらなる解析: per-patch 絶対スコア掃引 + 特徴空間写像。

`occlusion_sensitivity`（models/occlusion.py）は 1 画像あたり全パッチ位置を遮蔽した際の
スコア変化を平均した Δ マップしか返さず、個々の遮蔽画像の絶対スコア s(x_occ) や中間特徴
ベクトルは内部で捨てている。本スクリプトは 1 枚（または --n-images 枚）の画像について
`build_occluded_batch` で遮蔽バッチを自前構築し、各遮蔽バリアントの絶対スコア・logit・
中間特徴を保持したまま:
  1. per-patch 絶対スコアの空間ヒートマップ + 数値ダンプ（score_sweep.png / occlusion_scores.csv）
  2. 遮蔽バリアントの特徴空間写像（母集団 parquet + 原画像 + 遮蔽点を連結して一括 fit した
     embedding 上に散布。real/gen 母集団に対して遮蔽で点がどちらへ動くかを見る）
を出力する。

`fit_combined_embedding`（analysis/feature_analysis.py）は fit_transform 一発で reducer を
破棄するため、母集団特徴 + 遮蔽バリアント特徴を連結して一度だけ fit してからスライスする
（t-SNE を含む全 reducer で動く唯一の方法）。既定 reducer は pca（微小摂動 → 微小移動を
決定論的・線形に見るため。tsne/umap は --reducer で選択可）。

layer4 は空間的に粗い（96px×resnet50 で GAP 前 ~3×3）ため、小パッチ遮蔽では layer4 の
GAP 特徴がほとんど動かず遮蔽点が原画像点に重なりうる。--embedding-layer で layer2/3 も
試すこと。

入力は visualize_occlusion.py と同じく `FontDataModule(mode="scorer")` から
`batch["image"]`（モデルが実際に見た入力、use_sdf なら SDF decode 済み）を取るため、
--config は「学習済み Scorer の config.yaml」を指す（SDF kwarg 再現用）。--features は
同一 checkpoint 由来の extract_scorer_features.py 出力 parquet を指定する（必須）。

Usage:
    uv run python scripts/analyze_occlusion.py \\
        --checkpoint outputs/<exp>/checkpoints/last.ckpt \\
        --config     outputs/<exp>/config.yaml \\
        --features   outputs/<exp>/features/scorer_features.parquet \\
        --char H --label real --output-dir /path/to/output

    # 複数画像・粗い遮蔽・別レイヤーで特徴空間を見る
    uv run python scripts/analyze_occlusion.py \\
        --checkpoint last.ckpt --config config.yaml --features features.parquet \\
        --char H --label real --n-images 5 --patch 12 --stride 6 --embedding-layer layer2

    # 全文字 x real/gen 両方を一括処理（model/data/features のロードは 1 回のみ）
    uv run python scripts/analyze_occlusion.py \\
        --checkpoint last.ckpt --config config.yaml --features features.parquet \\
        --all-chars --label both --n-images 5
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Rectangle

from real_fonts_vs_gen_fonts.analysis.feature_analysis import (
    fit_combined_embedding,
    fit_linear_probe,
    load_features,
)
from real_fonts_vs_gen_fonts.datamodule import FontDataModule, _resolve_codepoints
from real_fonts_vs_gen_fonts.models.occlusion import (
    build_occluded_batch,
    cumulative_curve_batches as _cumulative_curve_batches,
    make_blank_like,
    overlay_occlusion,
    patch_value_map as _patch_value_map,
)
from real_fonts_vs_gen_fonts.models.scorer_module import ScorerModule
from real_fonts_vs_gen_fonts.utils.config import load_config, save_run_info, setup_logging
from real_fonts_vs_gen_fonts.utils.torch_utils import get_device

REAL_COLOR = "#2196F3"
GEN_COLOR = "#FF5722"

_WHITE_VALUE = 1.0
"""s_blank（白紙）の塗り値。sdf_decode=hard/smoothstep や素の二値画像では background=1.0。"""
_BLACK_VALUE = 0.0
"""s_fill（黒紙）の塗り値。同上の規約で stroke(ink)=0.0。raw SDF は極性が逆になるため注意。"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Per-patch absolute-score occlusion sweep + feature-space mapping"
    )
    parser.add_argument("--checkpoint", required=True, metavar="CKPT",
                        help="ScorerModule checkpoint (.ckpt)")
    parser.add_argument("--config", required=True, metavar="FILE",
                        help="Trained Scorer config.yaml (for data / SDF kwargs)")
    parser.add_argument("--features", required=True, metavar="PARQUET",
                        help="Population features parquet from extract_scorer_features.py "
                             "(must be from the same checkpoint)")
    parser.add_argument("--output-dir", default=None, metavar="DIR",
                        help="Output directory (default: <ckpt_dir>/../occlusion_analysis/)")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"],
                        help="Which split to sample images from (default: test)")
    parser.add_argument("--char", nargs="+", default=None, metavar="CHAR",
                        help="Restrict image selection to specific character(s), e.g. --char A B "
                             "(default: any, or all charset chars with --all-chars)")
    parser.add_argument("--all-chars", action="store_true",
                        help="Loop internally over all charset characters (or --char list), "
                             "writing per-char/per-label output dirs. Loads model/data/features "
                             "once instead of once per character (default: off)")
    parser.add_argument("--label", default="real", choices=["real", "gen", "both"],
                        help="Which class to sample images from; 'both' analyzes real and gen "
                             "(default: real)")
    parser.add_argument("--n-images", type=int, default=1,
                        help="Number of images to analyze (default: 1)")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed (default: 0)")
    parser.add_argument("--patch", type=int, default=8,
                        help="Occlusion patch size in px (default: 8)")
    parser.add_argument("--stride", type=int, default=4,
                        help="Occlusion stride in px; smaller = finer/slower (default: 4)")
    parser.add_argument("--baseline", default="background",
                        help="Patch fill: background/mean/min/max or a float (default: background)")
    parser.add_argument("--target", default="logit", choices=["logit", "score"],
                        help="Sweep heatmap target: logit (pre-sigmoid) or score (default: logit)")
    parser.add_argument("--min-change", type=float, default=0.02,
                        help="Skip background-only patches whose occlusion barely changes the "
                             "input, as a fraction of the input dynamic range (0 = analyze every "
                             "patch). Keeps only stroke-touching patches (default: 0.02)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Occlusion forward chunk size (default: from config data.batch_size)")
    parser.add_argument("--embedding-layer", default="layer4",
                        choices=["layer1", "layer2", "layer3", "layer4"],
                        help="Which extract_intermediate layer to embed (default: layer4)")
    parser.add_argument("--reducer", default="pca", choices=["pca", "tsne", "umap"],
                        help="Dimensionality reducer for the feature-space plot (default: pca)")
    parser.add_argument("--pop-samples", type=int, default=5000,
                        help="Population subsample cap before the combined fit (default: 5000)")
    parser.add_argument("--pop-char", default=None, metavar="CHAR",
                        help="Restrict the feature-space population to a character. Unset = "
                             "match each analyzed image's own char (char-matched background); "
                             "'all' = keep every char; a specific char, e.g. H (default: unset)")
    parser.add_argument("--grid-n", type=int, default=6,
                        help="N for the occluded-image N x N ranked grids (default: 6 = 36/band)")
    parser.add_argument("--grid-style", default="diff", choices=["diff", "plain"],
                        help="Ranked-grid cell rendering: 'diff' tints pixels the patch actually "
                             "changed in red, 'plain' is grayscale only (default: diff)")
    parser.add_argument("--thumb-size", type=int, default=0,
                        help="Thumbnail resolution (px) for embedding-plot overlays; "
                             "0 = keep the model-input resolution (default: 0)")
    parser.add_argument("--thumb-zoom", type=float, default=0.2,
                        help="Matplotlib zoom for occluded-variant thumbnails; the displayed "
                             "size is thumb_size x zoom x dpi/72 px, and the original image is "
                             "drawn at 1.6x this zoom (default: 0.2)")
    parser.add_argument("--max-occ-thumbs", type=int, default=24,
                        help="Max occluded-variant thumbnails per embedding plot; points are "
                             "spatially binned so nearby ones don't overlap (default: 24)")
    parser.add_argument("--probe", action="store_true", default=True,
                        help="Fit a linear probe on the population train split and append "
                             "p(real) per occluded variant to the CSV (default)")
    parser.add_argument("--no-probe", dest="probe", action="store_false",
                        help="Skip the linear-probe p(real) column")
    parser.add_argument("--keep", action="store_true", default=True,
                        help="Also compute preservation (patch-kept-only) scores, normalized "
                             "against a fully-baseline-filled image s_blank: keep_sweep.png, "
                             "kept_ranked_{top,bottom}.png, necessity_vs_sufficiency.png, "
                             "necessity_sufficiency.png, and keep_* columns in "
                             "occlusion_scores.csv (default)")
    parser.add_argument("--no-keep", dest="keep", action="store_false",
                        help="Skip preservation outputs")
    parser.add_argument("--grid-refs", action="store_true", default=True,
                        help="Add blank(white)/fill(black) reference cells to "
                             "kept_ranked_{top,bottom}.png, framed and labeled with their "
                             "scores, so fragment scores can be visually compared against the "
                             "s_blank/s_fill extremes (requires --keep, default)")
    parser.add_argument("--no-grid-refs", dest="grid_refs", action="store_false",
                        help="Skip the blank/fill reference cells in kept_ranked grids")
    parser.add_argument("--curves", action="store_true", default=True,
                        help="Compute cumulative deletion/insertion curves (delta-descending "
                             "patch order) and their AUC: deletion_insertion.png, "
                             "curve_metrics.csv (default)")
    parser.add_argument("--no-curves", dest="curves", action="store_false",
                        help="Skip deletion/insertion curve outputs")
    parser.add_argument("--use-ema", action="store_true", default=True,
                        help="Use EMA-averaged Scorer weights (default)")
    parser.add_argument("--no-ema", dest="use_ema", action="store_false",
                        help="Use raw (non-EMA) Scorer weights")
    parser.add_argument("overrides", nargs="*", metavar="key=value",
                        help="Config overrides, e.g. data.real_data_dir=/path")
    return parser.parse_args()


def _resolve_baseline_arg(baseline: str) -> str | float:
    try:
        return float(baseline)
    except ValueError:
        return baseline


def _collect_split(loader: torch.utils.data.DataLoader) -> dict[int, list[dict]]:
    """split の全サンプルを label(1=real/0=gen) 別に集める。"""
    by_label: dict[int, list[dict]] = {0: [], 1: []}
    for batch in loader:
        for i in range(len(batch["label"])):
            lab = int(batch["label"][i])
            by_label[lab].append({
                "image": batch["image"][i],
                "char": str(batch["char"][i]),
                "font_family": str(batch["font_family"][i]),
                "image_path": str(batch["image_path"][i]),
            })
    return by_label


def _forward_chunked(
    scorer, x: torch.Tensor, embedding_layer: str, chunk_size: int, want_feats: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """x を chunk_size ごとに forward し、(logit, score, feature) を連結して返す。

    `want_feats=False` は preservation / deletion-insertion 曲線用の経路で、特徴空間写像に
    使わないため `extract_intermediate` を省いて forward コストを下げる。
    """
    logits_list, feats_list = [], []
    with torch.no_grad():
        for start in range(0, len(x), chunk_size):
            batch = x[start : start + chunk_size]
            logits_list.append(scorer.forward_logits(batch).cpu())
            if want_feats:
                feats_list.append(scorer.extract_intermediate(batch)[embedding_layer].cpu())
    logits = torch.cat(logits_list)
    scores = torch.sigmoid(logits)
    feats = torch.cat(feats_list) if want_feats else None
    return logits, scores, feats


def _plot_score_sweep(
    image: np.ndarray,
    positions: list[tuple[int, int]],
    scores: np.ndarray,
    deltas: np.ndarray,
    patch_size: int,
    ref_score: float,
    output_path: Path,
    score_label: str = "occluded score s(x_occ)",
    delta_label: str = "delta = s0 - s_occ",
    ref_label: str = "s0",
) -> None:
    """左=入力グリフ、中=絶対スコアの空間ヒートマップ、右=発散マップ を1枚に保存する。

    既定は deletion 用（`score_sweep.png`）。preservation 用（`keep_sweep.png`）は基準値が
    s0 ではなく s_blank になるため `score_label`/`delta_label`/`ref_label` で差し替えて流用する。
    """
    H, W = image.shape[:2]
    score_map = _patch_value_map(positions, scores, patch_size, (H, W))
    delta_map = _patch_value_map(positions, deltas, patch_size, (H, W))

    fig, axes = plt.subplots(1, 3, figsize=(11, 4))

    axes[0].imshow(image)
    axes[0].set_title("input", fontsize=9)
    axes[0].axis("off")

    im1 = axes[1].imshow(score_map, cmap="viridis")
    axes[1].set_title(f"{score_label}  ({ref_label}={ref_score:.3f})", fontsize=9)
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    overlay = overlay_occlusion(image, delta_map, target_size=(H, W))
    axes[2].imshow(overlay)
    axes[2].set_title(delta_label, fontsize=9)
    axes[2].axis("off")
    sm = cm.ScalarMappable(norm=Normalize(vmin=-1.0, vmax=1.0), cmap="RdBu_r")
    fig.colorbar(sm, ax=axes[2], fraction=0.046, pad=0.04, ticks=[-1, 0, 1])

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def _to_gray(x: torch.Tensor) -> np.ndarray:
    """モデル入力 (N, C, H, W) をフル解像度のままグレースケール (N, H, W) にする。"""
    return x.mean(dim=1).clamp(0.0, 1.0).cpu().numpy()


def _make_thumbnails(x: torch.Tensor, size: int | None) -> np.ndarray:
    """モデル入力 (N, C, H, W) から埋め込みプロット用のグレースケールサムネイルを作る。

    `size` が None か 0 以下ならリサイズせずモデル入力の解像度をそのまま使う。縮小は
    埋め込みプロット（AnnotationBbox の表示サイズ）だけの都合であり、スコア順グリッドは
    フル解像度を必要とするため経路を分けている。
    """
    if size is None or size <= 0:
        return _to_gray(x)
    gray = x.mean(dim=1, keepdim=True)
    resized = F.interpolate(gray, size=(size, size), mode="area")
    return resized.squeeze(1).clamp(0.0, 1.0).cpu().numpy()


def _bin_and_cap(points: np.ndarray, max_count: int, grid: int = 20) -> np.ndarray:
    """空間グリッドで間引きつつ max_count 件までのインデックスを返す（サムネイル重なり防止）。

    1 セルにつき最初に出現した 1 点だけ残す。密な格子状の遮蔽バリアントでも代表点だけが
    残るため、サムネイルが積み重なって潰れるのを防ぐ。
    """
    if len(points) == 0 or max_count <= 0:
        return np.array([], dtype=int)
    mins = points.min(axis=0)
    span = np.where(points.max(axis=0) - mins == 0, 1.0, points.max(axis=0) - mins)
    cells = np.floor((points - mins) / span * (grid - 1e-9)).astype(int)
    seen: set[tuple[int, int]] = set()
    keep: list[int] = []
    for i in range(len(points)):
        key = (int(cells[i, 0]), int(cells[i, 1]))
        if key in seen:
            continue
        seen.add(key)
        keep.append(i)
        if len(keep) >= max_count:
            break
    return np.array(keep, dtype=int)


def _add_thumbnail(
    ax: "plt.Axes", xy: tuple[float, float], image: np.ndarray, zoom: float,
    border_color, lw: float = 1.2,
) -> None:
    """埋め込み座標 xy に画像サムネイルを縁取り付きで重畳する。"""
    ab = AnnotationBbox(
        OffsetImage(image, cmap="gray", zoom=zoom), xy,
        frameon=True, pad=0.05, bboxprops=dict(edgecolor=border_color, linewidth=lw),
    )
    ax.add_artist(ab)


def _plot_feature_embedding(
    emb: np.ndarray,
    pop_labels: np.ndarray,
    n_pop: int,
    occ_scores: np.ndarray,
    occ_thumbs: np.ndarray,
    orig_thumb: np.ndarray,
    max_thumbs: int,
    thumb_zoom: float,
    output_dir: Path,
) -> None:
    """母集団（real/gen 淡色ドット）+ 遮蔽点（score 連続色、間引いてサムネイル重畳）+ 原画像
    サムネイル（黒枠・大きめ）を1枚に描く。全遮蔽点は常に色付きドットとして残り、
    サムネイルは空間ビニングで間引いた代表点にのみ重畳する。
    """
    pop_emb = emb[:n_pop]
    orig_emb = emb[n_pop : n_pop + 1]
    occ_emb = emb[n_pop + 1 :]

    fig, ax = plt.subplots(figsize=(9, 8))
    for lab, name, color in [(1, "real (pop)", REAL_COLOR), (0, "gen (pop)", GEN_COLOR)]:
        mask = pop_labels == lab
        if not mask.any():
            continue
        ax.scatter(pop_emb[mask, 0], pop_emb[mask, 1], c=color, alpha=0.12, s=5,
                   linewidths=0, label=name)
    sc = ax.scatter(occ_emb[:, 0], occ_emb[:, 1], c=occ_scores, cmap="RdYlGn",
                    vmin=0.0, vmax=1.0, s=14, alpha=0.9, linewidths=0.3,
                    edgecolors="black", label="occluded", zorder=3)
    fig.colorbar(sc, ax=ax, label="occluded score s(x_occ)")

    cmap = plt.get_cmap("RdYlGn")
    norm = Normalize(vmin=0.0, vmax=1.0)
    for i in _bin_and_cap(occ_emb, max_thumbs):
        _add_thumbnail(ax, (occ_emb[i, 0], occ_emb[i, 1]), occ_thumbs[i], thumb_zoom,
                      border_color=cmap(norm(occ_scores[i])))
    _add_thumbnail(ax, (orig_emb[0, 0], orig_emb[0, 1]), orig_thumb, thumb_zoom * 1.6,
                  border_color="black", lw=2.0)
    ax.scatter(orig_emb[:, 0], orig_emb[:, 1], c="black", marker="*", s=60,
              label="original", zorder=6)

    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.legend(loc="best", fontsize=8, markerscale=1.5)
    fig.tight_layout()
    fig.savefig(output_dir / "feature_embedding.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_dir / 'feature_embedding.png'}")


def _position_colors(
    positions: list[tuple[int, int]], image_size: tuple[int, int]
) -> np.ndarray:
    """(top, left) を [0,1]^2 に正規化し、赤=top軸・緑=left軸の2チャンネルRGBに写像する。

    特徴空間の位置別散布図と necessity-vs-sufficiency 散布図で共有する、空間位置の共通配色。
    """
    H, W = image_size
    tops = np.array([p[0] for p in positions], dtype=np.float64)
    lefts = np.array([p[1] for p in positions], dtype=np.float64)
    r = tops / max(H - 1, 1)
    g = lefts / max(W - 1, 1)
    return np.stack([r, g, np.full_like(r, 0.5)], axis=1)


def _plot_feature_embedding_by_position(
    emb: np.ndarray,
    n_pop: int,
    positions: list[tuple[int, int]],
    image_size: tuple[int, int],
    occ_thumbs: np.ndarray,
    orig_thumb: np.ndarray,
    max_thumbs: int,
    thumb_zoom: float,
    output_dir: Path,
) -> None:
    """遮蔽点を (top, left) の 2D 位置カラーで配色し、間引いたサムネイルを重畳する。

    空間上の遮蔽位置↔特徴空間上の移動先の対応を、実画像で直接確認できるようにする。
    """
    orig_emb = emb[n_pop : n_pop + 1]
    occ_emb = emb[n_pop + 1 :]

    colors = _position_colors(positions, image_size)

    fig, ax = plt.subplots(figsize=(9, 8))
    pop_emb = emb[:n_pop]
    ax.scatter(pop_emb[:, 0], pop_emb[:, 1], c="lightgray", alpha=0.12, s=5, linewidths=0,
              label="population")
    ax.scatter(occ_emb[:, 0], occ_emb[:, 1], c=colors, s=14, alpha=0.9, linewidths=0.3,
              edgecolors="black", label="occluded (color = patch position)", zorder=3)

    for i in _bin_and_cap(occ_emb, max_thumbs):
        _add_thumbnail(ax, (occ_emb[i, 0], occ_emb[i, 1]), occ_thumbs[i], thumb_zoom,
                      border_color=colors[i])
    _add_thumbnail(ax, (orig_emb[0, 0], orig_emb[0, 1]), orig_thumb, thumb_zoom * 1.6,
                  border_color="black", lw=2.0)
    ax.scatter(orig_emb[:, 0], orig_emb[:, 1], c="black", marker="*", s=60,
              label="original", zorder=6)

    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.legend(loc="best", fontsize=8, markerscale=1.5)
    fig.tight_layout()
    fig.savefig(output_dir / "feature_embedding_bypos.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_dir / 'feature_embedding_bypos.png'}")


def _filter_population(
    df_meta: pd.DataFrame, pop_feats: np.ndarray, char: str, pop_char: str | None
) -> tuple[pd.DataFrame, np.ndarray]:
    """母集団を char で絞る。pop_char=None ならターゲットと同じ char、"all" なら絞らない。

    ターゲット画像と同じ文字だけを特徴空間の背景にすることで、遮蔽点がどちらのクラスへ
    動くかを char-matched に比較できる（全文字混在の背景より解釈しやすい）。
    """
    if pop_char is not None and pop_char.lower() == "all":
        return df_meta.reset_index(drop=True), pop_feats
    target = char if pop_char is None else pop_char
    mask = (df_meta["char"] == target).to_numpy()
    if not mask.any():
        available = sorted(df_meta["char"].unique())
        raise ValueError(
            f"No population samples for char={target!r}. Available: {available}"
        )
    return df_meta[mask].reset_index(drop=True), pop_feats[mask]


def _plot_feature_embedding_byscore(
    emb: np.ndarray,
    pop_scores: np.ndarray,
    n_pop: int,
    occ_scores: np.ndarray,
    occ_thumbs: np.ndarray,
    orig_thumb: np.ndarray,
    max_thumbs: int,
    thumb_zoom: float,
    output_dir: Path,
) -> None:
    """母集団も realness score で連続配色した特徴空間プロット。

    `feature_embedding.png`（母集団を real/gen の 2 色）と違い、母集団・遮蔽点の両方を
    同一の RdYlGn スコアスケールに乗せる。母集団のスコア勾配と遮蔽点の位置を一枚で見る。
    """
    pop_emb = emb[:n_pop]
    orig_emb = emb[n_pop : n_pop + 1]
    occ_emb = emb[n_pop + 1 :]

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.scatter(pop_emb[:, 0], pop_emb[:, 1], c=pop_scores, cmap="RdYlGn",
               vmin=0.0, vmax=1.0, alpha=0.25, s=6, linewidths=0, label="population")
    sc = ax.scatter(occ_emb[:, 0], occ_emb[:, 1], c=occ_scores, cmap="RdYlGn",
                    vmin=0.0, vmax=1.0, s=14, alpha=0.95, linewidths=0.3,
                    edgecolors="black", label="occluded", zorder=3)
    fig.colorbar(sc, ax=ax, label="realness score")

    cmap = plt.get_cmap("RdYlGn")
    norm = Normalize(vmin=0.0, vmax=1.0)
    for i in _bin_and_cap(occ_emb, max_thumbs):
        _add_thumbnail(ax, (occ_emb[i, 0], occ_emb[i, 1]), occ_thumbs[i], thumb_zoom,
                      border_color=cmap(norm(occ_scores[i])))
    _add_thumbnail(ax, (orig_emb[0, 0], orig_emb[0, 1]), orig_thumb, thumb_zoom * 1.6,
                  border_color="black", lw=2.0)
    ax.scatter(orig_emb[:, 0], orig_emb[:, 1], c="black", marker="*", s=60,
              label="original", zorder=6)

    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.legend(loc="best", fontsize=8, markerscale=1.5)
    fig.tight_layout()
    fig.savefig(output_dir / "feature_embedding_byscore.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_dir / 'feature_embedding_byscore.png'}")


_TINT_RGB = (0.90, 0.18, 0.18)
_TINT_REL_TOL = 0.02
"""遮蔽で「実質変化した」と見なす画素のしきい値（入力のダイナミックレンジに対する相対値）。

`--min-change` と同じ「実質変化なし」の基準だが、あちらはパッチ単位の最大値でパッチ自体を
解析対象から落とすためのもの、こちらは画素単位の着色判定なので別定数にしている。
"""


def _composite_removed_ink(orig: np.ndarray, occ: np.ndarray) -> np.ndarray:
    """遮蔽で変化した画素を赤で着色した RGB (H, W, 3) を返す。

    黒 = 残ったインク / 白 = 背景 / 赤 = このパッチが消した（変化させた）画素。
    baseline="background" の遮蔽は背景に当たると入力が実質変わらないため、赤枠だけでは
    「そのパッチが何を壊したのか」が読めない。差分を明示することで no-op パッチと
    ストロークを消したパッチを一目で区別できる。
    """
    rgb = np.repeat(occ[:, :, None], 3, axis=2)
    rng = float(orig.max() - orig.min())
    if rng <= 0.0:
        return rgb
    mask = np.abs(orig - occ) > _TINT_REL_TOL * rng
    rgb[mask] = _TINT_RGB
    return rgb


def _occluded_grid(
    images: list[np.ndarray], labels: list[str], output_path: Path, ncols: int,
    patch_boxes: list[tuple[float, float, float, float] | None] | None = None,
    legend: bool = False,
    frame_colors: list[str | None] | None = None,
    tint_legend: bool = False,
    box_label: str = "patch",
    frame_legend: list[tuple[str, str]] | None = None,
) -> None:
    """遮蔽画像を N 列グリッドで保存する（空セルは空白）。グレースケールと RGB の両方を受ける。

    `patch_boxes`（各画像に対応する (x, y, w, h)、画像と同じピクセル座標系、`None` 要素は
    枠なし）を渡すと、パッチ位置を赤枠で囲む。背景色で塗りつぶした遮蔽は淡い/アンチエイリアシング
    の影響でどこが隠れているか目視しにくいサンプルがあるため、位置を明示する。
    `frame_colors`（`None` 以外の要素だけ軸枠を着色）は原画像セルなど特定セルを強調するのに使う。
    `frame_legend`（`(色, ラベル)` のリスト）を渡すと、`frame_colors` の凡例ハンドルを複数
    出せる（既定は従来どおり blue/"original" 1 個）。
    """
    n = len(images)
    if n == 0:
        return
    nrows = (n + ncols - 1) // ncols
    # WHY a fixed inch margin instead of a fraction of figure height: the bottom legend's
    # bbox_to_anchor y-offset used to be a small negative *fraction*, so its absolute gap
    # from the axes shrank along with figsize height (nrows * 1.55in). For nrows=1 that left
    # only ~0.02in of clearance and the legend overlapped the row of images; for large grids
    # the same fraction gave plenty of room, which is why the bug only showed up at nrows=1.
    legend_margin_in = 0.45 if legend else 0.0
    grid_height_in = nrows * 1.55
    fig_height_in = grid_height_in + legend_margin_in
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.4, fig_height_in),
                             squeeze=False)
    for k in range(nrows * ncols):
        ax = axes[k // ncols][k % ncols]
        ax.set_xticks([])
        ax.set_yticks([])
        if k < n:
            # WHY interpolation="nearest": 96px の入力を ~210 device px のセルへ拡大する図なので、
            # 既定の補間だと画素境界がぼけて「どの画素が塗られたか」が読めなくなる。
            if images[k].ndim == 3:
                ax.imshow(images[k], interpolation="nearest")
            else:
                ax.imshow(images[k], cmap="gray", vmin=0.0, vmax=1.0,
                          interpolation="nearest")
            ax.set_title(labels[k], fontsize=7)
            if patch_boxes is not None and patch_boxes[k] is not None:
                x, y, w, h = patch_boxes[k]
                ax.add_patch(Rectangle(
                    (x - 0.5, y - 0.5), w, h,
                    edgecolor="red", facecolor="none", linewidth=1.3,
                ))
            if frame_colors is not None and frame_colors[k] is not None:
                for spine in ax.spines.values():
                    spine.set_edgecolor(frame_colors[k])
                    spine.set_linewidth(2.0)
        else:
            ax.axis("off")
    if legend:
        fig.tight_layout(rect=[0, legend_margin_in / fig_height_in, 1, 1])
    else:
        fig.tight_layout()
    if legend:
        # 図タイトル禁止の規約に従い、色の意味は凡例で示す。
        handles = []
        if tint_legend:
            handles.append(Rectangle((0, 0), 1, 1, facecolor=_TINT_RGB, edgecolor="none",
                                      label="removed ink (occluded)"))
        if patch_boxes is not None:
            handles.append(Rectangle((0, 0), 1, 1, facecolor="none", edgecolor="red",
                                      linewidth=1.3, label=box_label))
        if frame_colors is not None:
            if frame_legend is not None:
                for color, label in frame_legend:
                    handles.append(Rectangle((0, 0), 1, 1, facecolor="none", edgecolor=color,
                                              linewidth=2.0, label=label))
            else:
                handles.append(Rectangle((0, 0), 1, 1, facecolor="none", edgecolor="blue",
                                          linewidth=2.0, label="original"))
        fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=8,
                   frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def _plot_occluded_ranked(
    occ: torch.Tensor,
    image: torch.Tensor,
    occ_scores: np.ndarray,
    positions: list[tuple[int, int]],
    grid_n: int,
    output_dir: Path,
    patch_size: int,
    grid_style: str,
    s0_score: float,
    file_prefix: str = "occluded_ranked",
    band_names: tuple[str, ...] = ("top", "middle", "bottom"),
    tint: bool = True,
    box_label: str = "patch",
    ref_cells: list[tuple[np.ndarray, str, str]] | None = None,
) -> None:
    """遮蔽/保持バリアントを score 降順に並べ、指定した帯を N x N グリッド保存する。

    top = 高スコア帯、bottom = 低スコア帯、middle = 中央値付近
    （feature_analysis.plot_score_ranked_images と同じ流儀）。deletion（`tint=True`）では
    top = 隠しても realness が落ちない領域・bottom = realness を最も支える領域。
    preservation（`tint=False`）では同じ score 順ロジックが keep_score に適用され、
    top = そこだけで realness が高い（十分性が高い）領域・bottom = 単独では realness が
    出ない領域、という逆の意味になる（呼び出し元が `occ_scores` に渡す配列で切り替わる）。

    各グリッドの**セル 0 は常に遮蔽なしの原画像**（青枠、パッチ枠なし）で、同一図内で
    直接比較できるようにする。`ref_cells`（`(画像, ラベル, 枠色)` のリスト）を渡すと
    原画像の直後に追加の参照セルを挿入する（preservation の `s_blank`/`s_fill` 基準画像など。
    パッチ枠・diff 着色の対象外）。帯の実バリアントは `grid_n**2 - 1 - len(ref_cells)` 枚。

    描画は**モデル入力そのものの解像度**で行い（`occ` は (N, C, H, W) のフル解像度バッチ）、
    パッチ位置を実画素境界に一致する赤枠で重畳する。`tint=True` かつ `grid_style="diff"` の
    ときだけ原画像との差分画素を赤で塗る（`tint=False` の preservation は遮蔽領域外が
    全部「変化した画素」になり画面が赤で埋まるため、着色しない）。
    """
    orig_gray = _to_gray(image.unsqueeze(0))[0]
    do_tint = tint and grid_style == "diff"
    orig_cell = np.repeat(orig_gray[:, :, None], 3, axis=2) if do_tint else orig_gray

    ref_cells = ref_cells or []
    ref_images = [np.repeat(img[:, :, None], 3, axis=2) if do_tint else img
                  for img, _, _ in ref_cells]
    ref_labels = [label for _, label, _ in ref_cells]
    ref_frame_colors = [color for _, _, color in ref_cells]
    ref_boxes: list[tuple[float, float, float, float] | None] = [None] * len(ref_cells)

    # WHY 反転ビュー (`[::-1]`) を作らない: 負ストライドの配列は torch のインデクシングが
    # 受け付けず、しかも 1 要素配列は numpy が C 連続と判定するため `np.ascontiguousarray`
    # でも救えない（min_change で残るパッチが 1 個だけのとき実際に落ちた）。降順は
    # `argsort(-x)`、昇順は `argsort(x)` で正ストライドのまま作る。
    desc = np.argsort(-occ_scores)
    asc = np.argsort(occ_scores)
    m = len(desc)
    # セル0=原画像 + ref_cells で固定するので帯はそれを引いた枚数
    cap = max(grid_n * grid_n - 1 - len(ref_cells), 0)
    mid_start = max(0, m // 2 - cap // 2)
    all_bands = {
        "top": desc[:cap],
        "middle": desc[mid_start:mid_start + cap],
        "bottom": asc[:cap],  # 最低スコアを先頭に
    }
    frame_legend = [("blue", "original")] + list(zip(ref_frame_colors, ref_labels))
    for name in band_names:
        band = all_bands[name]
        band_gray = _to_gray(occ[band])  # 帯の分だけフル解像度化する
        if do_tint:
            variants = [_composite_removed_ink(orig_gray, g) for g in band_gray]
        else:
            variants = list(band_gray)
        images = [orig_cell] + ref_images + variants
        labels = [f"original s0={s0_score:.2f}"] + ref_labels + [
            f"s={occ_scores[j]:.2f} ({positions[j][0]},{positions[j][1]})" for j in band
        ]
        boxes = [None] + ref_boxes + [
            (float(positions[j][1]), float(positions[j][0]),
             float(patch_size), float(patch_size))
            for j in band
        ]
        frame_colors = ["blue"] + ref_frame_colors + [None] * len(band)
        _occluded_grid(images, labels, output_dir / f"{file_prefix}_{name}.png", grid_n,
                       patch_boxes=boxes, legend=True, frame_colors=frame_colors,
                       tint_legend=do_tint, box_label=box_label, frame_legend=frame_legend)


def _plot_necessity_vs_sufficiency(
    del_scores: np.ndarray,
    keep_scores: np.ndarray,
    positions: list[tuple[int, int]],
    image_size: tuple[int, int],
    s0_score: float,
    s_blank_score: float,
    s_fill_score: float,
    output_path: Path,
) -> None:
    """x=deletion 後のスコア（低いほど必要性が高い）、y=preservation スコア（高いほど十分性が高い）
    の散布図。点色は空間位置（`_position_colors` と共通）。deletion 側の `s0` に垂直線、
    preservation 側の `s_blank`（白紙）/`s_fill`（黒紙）に水平線を引く。象限の意味は
    図タイトル禁止の規約に従い軸ラベルで示す: 左上=必要かつ十分／左下=必要だが単独では
    不十分／右上=冗長だが十分／右下=どちらでもない。preservation のスコアが `s_blank`/`s_fill`
    と大差なければ、そのパッチ単独の寄与は白紙・黒紙という無内容な入力と区別できていない
    （十分性が低い）ことを意味する。
    """
    colors = _position_colors(positions, image_size)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(del_scores, keep_scores, c=colors, s=20, alpha=0.85,
              edgecolors="black", linewidths=0.3, zorder=3)
    ax.axvline(s0_score, color="black", linestyle="--", linewidth=1.0,
              label=f"s0={s0_score:.2f}")
    ax.axhline(s_blank_score, color="gray", linestyle="--", linewidth=1.0,
              label=f"s_blank(white)={s_blank_score:.2f}")
    ax.axhline(s_fill_score, color="dimgray", linestyle=":", linewidth=1.0,
              label=f"s_fill(black)={s_fill_score:.2f}")
    ax.set_xlabel("score with this patch occluded (deletion) - low = necessary")
    ax.set_ylabel("score with only this patch kept (preservation) - high = sufficient")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


_NS_PALETTE = np.array([
    [0.482, 0.176, 0.557],  # 0: necessary & sufficient (purple)
    [0.827, 0.184, 0.184],  # 1: necessary only (red, matches deletion curve color)
    [0.098, 0.463, 0.824],  # 2: sufficient only (blue, matches insertion curve color)
    [0.780, 0.780, 0.780],  # 3: neither (gray)
])
_NS_LABELS = ("necessary & sufficient", "necessary only", "sufficient only", "neither")


def _necessity_sufficiency_colors(delta: np.ndarray, keep_delta: np.ndarray) -> np.ndarray:
    """delta/keep_delta の**符号**だけで 4 値カテゴリに分け、フラットな色を返す（強弱なし）。

    necessary = delta > 0（隠すとスコアが下がる）、sufficient = keep_delta > 0
    （そのパッチだけで s_blank を上回る）。0 は「occluding/keeping が絶対的に良い方向に
    効くか」という符号そのものであり、画像ごとの min-max ではなく画像間で比較可能な
    絶対基準（以前の連続 bivariate 配色は強弱が読みにくく画像間比較もできなかったため、
    2 値 x 2 値 の 4 色に単純化した）。
    """
    necessary = delta > 0
    sufficient = keep_delta > 0
    cat = np.full(delta.shape, 3, dtype=np.int64)
    cat[necessary & sufficient] = 0
    cat[necessary & ~sufficient] = 1
    cat[~necessary & sufficient] = 2
    return _NS_PALETTE[cat]


def _plot_necessity_sufficiency_map(
    image: np.ndarray,
    positions: list[tuple[int, int]],
    deltas: np.ndarray,
    keep_deltas: np.ndarray,
    del_scores: np.ndarray,
    keep_scores: np.ndarray,
    patch_size: int,
    s0_score: float,
    s_blank_score: float,
    s_fill_score: float,
    output_path: Path,
) -> None:
    """necessity（deletion Δ）と sufficiency（preservation keep_delta）を 1 枚の
    4 色カテゴリカルマップに合成し、凡例・散布図と並べた 2x3 ダッシュボードを保存する。

    joint パネルはグリフに重畳しない（濃色が黒インクと衝突して読めなくなるため）。
    代わりに 0.5 レベルセットの輪郭線だけ黒で重ねる。
    """
    H, W = image.shape[:2]
    delta_map = _patch_value_map(positions, deltas, patch_size, (H, W))
    keep_delta_map = _patch_value_map(positions, keep_deltas, patch_size, (H, W))

    joint_map = _necessity_sufficiency_colors(delta_map, keep_delta_map)
    point_colors = _necessity_sufficiency_colors(deltas, keep_deltas)

    fig, axes = plt.subplots(2, 3, figsize=(11, 7.5))

    axes[0, 0].imshow(image)
    axes[0, 0].set_title(f"input (s0={s0_score:.2f})", fontsize=9)
    axes[0, 0].axis("off")

    nec_overlay = overlay_occlusion(image, delta_map, target_size=(H, W))
    axes[0, 1].imshow(nec_overlay)
    axes[0, 1].set_title("necessity: delta = s0 - s_occ", fontsize=9)
    axes[0, 1].axis("off")
    sm_nec = cm.ScalarMappable(norm=Normalize(vmin=-1.0, vmax=1.0), cmap="RdBu_r")
    fig.colorbar(sm_nec, ax=axes[0, 1], fraction=0.046, pad=0.04, ticks=[-1, 0, 1])

    suf_overlay = overlay_occlusion(image, keep_delta_map, target_size=(H, W))
    axes[0, 2].imshow(suf_overlay)
    axes[0, 2].set_title("sufficiency: keep_delta = s_keep - s_blank", fontsize=9)
    axes[0, 2].axis("off")
    sm_suf = cm.ScalarMappable(norm=Normalize(vmin=-1.0, vmax=1.0), cmap="RdBu_r")
    fig.colorbar(sm_suf, ax=axes[0, 2], fraction=0.046, pad=0.04, ticks=[-1, 0, 1])

    axes[1, 0].imshow(joint_map, interpolation="nearest")
    gray = image.mean(axis=-1) if image.ndim == 3 else image
    axes[1, 0].contour(gray, levels=[0.5], colors="black", linewidths=0.8)
    axes[1, 0].set_title("joint (necessary/sufficient by sign)", fontsize=9)
    axes[1, 0].axis("off")

    # 4 色凡例: 2x2 の色見本を quadrant として並べる（x=necessity, y=sufficiency の符号）。
    legend_grid = np.stack([
        [_NS_PALETTE[3], _NS_PALETTE[1]],  # row0 (sufficiency<=0): neither, necessary only
        [_NS_PALETTE[2], _NS_PALETTE[0]],  # row1 (sufficiency>0): sufficient only, both
    ])
    axes[1, 1].imshow(legend_grid, origin="lower", extent=(-1, 1, -1, 1), aspect="auto")
    axes[1, 1].axhline(0, color="black", linewidth=0.8)
    axes[1, 1].axvline(0, color="black", linewidth=0.8)
    for (row, col), label in zip([(0, 0), (0, 1), (1, 0), (1, 1)],
                                  [_NS_LABELS[3], _NS_LABELS[1], _NS_LABELS[2], _NS_LABELS[0]]):
        axes[1, 1].text(-0.5 + col, -0.5 + row, label, ha="center", va="center",
                        fontsize=7, wrap=True)
    axes[1, 1].set_xticks([])
    axes[1, 1].set_yticks([])
    axes[1, 1].set_xlabel("necessity (delta sign)", fontsize=8)
    axes[1, 1].set_ylabel("sufficiency (keep_delta sign)", fontsize=8)

    axes[1, 2].scatter(del_scores, keep_scores, c=point_colors, s=20, alpha=0.85,
                       edgecolors="black", linewidths=0.3, zorder=3)
    axes[1, 2].axvline(s0_score, color="black", linestyle="--", linewidth=1.0,
                       label=f"s0={s0_score:.2f}")
    axes[1, 2].axhline(s_blank_score, color="gray", linestyle="--", linewidth=1.0,
                       label=f"s_blank(white)={s_blank_score:.2f}")
    axes[1, 2].axhline(s_fill_score, color="dimgray", linestyle=":", linewidth=1.0,
                       label=f"s_fill(black)={s_fill_score:.2f}")
    axes[1, 2].set_xlabel("deletion score", fontsize=8)
    axes[1, 2].set_ylabel("preservation score", fontsize=8)
    axes[1, 2].legend(loc="best", fontsize=7)
    axes[1, 2].tick_params(labelsize=7)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def _plot_deletion_insertion(
    area_frac: np.ndarray,
    del_curve: np.ndarray,
    ins_curve: np.ndarray,
    deletion_auc: float,
    insertion_auc: float,
    output_path: Path,
) -> None:
    """delta（絶対値ではなく符号付き）降順に累積遮蔽/累積保持したときのスコア推移。

    deletion: 原画像から realness を最も支えるパッチ順に消していく（急落するほど鋭い attribution）。
    insertion: 空白から同じ順にパッチを戻していく（急騰するほど鋭い attribution）。
    x 軸は塗り替えた面積の割合（`_cumulative_curve_batches` よりタイル幾何上、終端で 1.0 に達する）。
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(area_frac, del_curve, color="#D32F2F", marker=".", markersize=3,
           label=f"deletion (AUC={deletion_auc:.3f})")
    ax.plot(area_frac, ins_curve, color="#1976D2", marker=".", markersize=3,
           label=f"insertion (AUC={insertion_auc:.3f})")
    ax.set_xlabel("cumulative area fraction (delta-descending patch order)")
    ax.set_ylabel("score")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def _subsample_population(
    df_meta: pd.DataFrame, pop_feats: np.ndarray, cap: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """母集団特徴を cap 件まで単純ランダムサブサンプルする（real/gen 両クラスが残るよう seed 固定）。

    Returns (feats, labels, scores)（行整列）。
    """
    n = len(pop_feats)
    if cap <= 0 or cap >= n:
        return (pop_feats, df_meta["label"].to_numpy(), df_meta["score"].to_numpy())
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(n, cap, replace=False))
    return (pop_feats[idx], df_meta["label"].to_numpy()[idx],
            df_meta["score"].to_numpy()[idx])


def analyze_one_image(
    scorer,
    image: torch.Tensor,
    char: str,
    label_name: str,
    args: argparse.Namespace,
    baseline: str | float,
    chunk: int,
    df_meta: pd.DataFrame,
    pop_feats_full: np.ndarray,
    output_dir: Path,
    target_meta: dict,
) -> dict[str, float]:
    """target_meta: どの画像を解析したかの出所情報（font_family/image_path/split/pick_index）。
    target_info.yaml に保存し、事後にどのサンプルを解析したか追跡できるようにする。

    Returns: manifest 集約用のスカラ指標（s0/s_blank・deletion/insertion AUC 等）。
    """
    device = image.device
    # 母集団をターゲットの char に絞る（既定 char-matched、--pop-char で all/任意文字に上書き）。
    df_pop, pop_feats = _filter_population(df_meta, pop_feats_full, char, args.pop_char)

    occ, positions = build_occluded_batch(
        image, patch_size=args.patch, stride=args.stride, baseline=baseline,
        min_change=args.min_change,
    )
    occ = occ.to(device)

    orig_logit, orig_score, orig_feat = _forward_chunked(
        scorer, image.unsqueeze(0), args.embedding_layer, chunk
    )
    s0_logit = float(orig_logit.item())
    s0_score = float(orig_score.item())

    occ_logits, occ_scores, occ_feats = _forward_chunked(
        scorer, occ, args.embedding_layer, chunk
    )
    target_s0 = s0_logit if args.target == "logit" else s0_score
    target_vals = occ_logits if args.target == "logit" else occ_scores
    deltas = (target_s0 - target_vals).numpy()

    # preservation / deletion-insertion 曲線共通の基準: 白紙(s_blank)・黒紙(s_fill) の
    # 両極端キャンバスをそのまま forward したスコア。"keep" は 96px のほぼ空白キャンバスに
    # 小片が乗った分布外入力になるため、素の値ではなく s_blank からの差分で読む
    # （models/occlusion.py の make_blank_like 参照）。
    # WHY --baseline（背景自動推定）ではなく固定 1.0/0.0: s_blank/s_fill は「白紙に消す/
    # 黒インクで塗り潰す」という表現非依存の物理的な両極端を指す基準点であり、パッチの
    # 塗り値（--baseline、既定 background 自動推定）とは役割が異なる。_WHITE_VALUE=1.0
    # (背景) / _BLACK_VALUE=0.0 (インク) は sdf_decode=hard/smoothstep や素の二値画像の
    # 規約（sdf.py の sdf_to_smoothstep: stroke=0, background=1）に一致するが、
    # raw SDF（sdf_decode=None）は極性が逆（stroke=1.0/background=0.0）になる点に注意。
    need_blank = args.keep or args.curves
    s_blank_logit = s_blank_score = s_fill_logit = s_fill_score = None
    if need_blank:
        blank_white = make_blank_like(image, _WHITE_VALUE).to(device)
        blank_black = make_blank_like(image, _BLACK_VALUE).to(device)
        white_logit, white_score, _ = _forward_chunked(
            scorer, blank_white, args.embedding_layer, chunk, want_feats=False
        )
        black_logit, black_score, _ = _forward_chunked(
            scorer, blank_black, args.embedding_layer, chunk, want_feats=False
        )
        s_blank_logit, s_blank_score = float(white_logit.item()), float(white_score.item())
        s_fill_logit, s_fill_score = float(black_logit.item()), float(black_score.item())

    output_dir.mkdir(parents=True, exist_ok=True)

    # どの画像を解析したかのメタデータ（font_family・元ファイルパス・split・抽出時の pick 番号）
    target_info = {
        "char": char, "label": label_name,
        "font_family": target_meta["font_family"], "image_path": target_meta["image_path"],
        "split": target_meta["split"], "pick_index": target_meta["pick_index"],
        "s0_score": s0_score, "s0_logit": s0_logit,
    }
    if need_blank:
        target_info["s_blank_score"] = s_blank_score
        target_info["s_blank_logit"] = s_blank_logit
        target_info["s_fill_score"] = s_fill_score
        target_info["s_fill_logit"] = s_fill_logit
    with open(output_dir / "target_info.yaml", "w") as f:
        yaml.dump(target_info, f, allow_unicode=True, sort_keys=False)

    # preservation（パッチだけ残す）: deletion と同一の positions になる
    # (build_occluded_batch は img/patch_size/stride/baseline/min_change が同じなら
    # min_change フィルタも同一集合を返すため)。per-patch で deletion と突き合わせられる。
    keep_scores = keep_logits = None
    keep_deltas = None
    if args.keep:
        keep_occ, keep_positions = build_occluded_batch(
            image, patch_size=args.patch, stride=args.stride, baseline=baseline,
            min_change=args.min_change, mode="keep",
        )
        assert keep_positions == positions, "keep/occlude positions diverged unexpectedly"
        keep_occ = keep_occ.to(device)
        keep_logits, keep_scores, _ = _forward_chunked(
            scorer, keep_occ, args.embedding_layer, chunk, want_feats=False
        )
        target_blank = s_blank_logit if args.target == "logit" else s_blank_score
        target_keep = keep_logits if args.target == "logit" else keep_scores
        keep_deltas = (target_keep - target_blank).numpy()  # 正 = blank より realness が高い(十分)

    # per-patch 数値ダンプ
    csv_path = output_dir / "occlusion_scores.csv"
    df = pd.DataFrame({
        "top": [p[0] for p in positions],
        "left": [p[1] for p in positions],
        "patch_size": args.patch,
        "score": occ_scores.numpy(),
        "logit": occ_logits.numpy(),
        "delta": deltas,
    })
    if args.keep:
        df["keep_score"] = keep_scores.numpy()
        df["keep_logit"] = keep_logits.numpy()
        df["keep_delta"] = keep_deltas
    if args.probe:
        train_mask = (df_pop["split"] == "train").to_numpy()
        pop_layer = pop_feats[train_mask]
        pop_label = df_pop.loc[train_mask, "label"].to_numpy()
        scaler, clf = fit_linear_probe(pop_layer, pop_label)
        p_real = clf.predict_proba(scaler.transform(occ_feats.numpy()))[:, 1]
        df["p_real"] = p_real
    df.attrs["s0_score"] = s0_score
    df.attrs["s0_logit"] = s0_logit
    with open(csv_path, "w") as f:
        header = (f"# s0_score={s0_score:.6f} s0_logit={s0_logit:.6f} ")
        if need_blank:
            header += (f"s_blank_score={s_blank_score:.6f} s_blank_logit={s_blank_logit:.6f} "
                      f"s_fill_score={s_fill_score:.6f} s_fill_logit={s_fill_logit:.6f} ")
        header += (f"baseline={args.baseline} char={char} label={label_name} "
                   f"font_family={target_meta['font_family']} image_path={target_meta['image_path']}\n")
        f.write(header)
        df.to_csv(f, index=False)
    print(f"Saved: {csv_path}")

    # score_sweep.png（deletion）
    img_np = image.permute(1, 2, 0).cpu().numpy()
    _plot_score_sweep(
        img_np, positions, occ_scores.numpy(), deltas, args.patch, target_s0,
        output_dir / "score_sweep.png",
    )

    orig_thumb = _make_thumbnails(image.unsqueeze(0), args.thumb_size)[0]
    occ_thumbs = _make_thumbnails(occ, args.thumb_size)

    # 遮蔽画像のスコア順グリッド（top / middle / bottom）。埋め込み用サムネではなく
    # フル解像度の `occ` を直接渡す（縮小すると遮蔽状況が読めなくなるため）。セル0は
    # 遮蔽なしの原画像で同一図内比較できる。
    _plot_occluded_ranked(
        occ, image, occ_scores.numpy(), positions, args.grid_n, output_dir,
        patch_size=args.patch, grid_style=args.grid_style, s0_score=s0_score,
    )

    if args.keep:
        # keep_sweep.png（preservation、基準は s0 ではなく s_blank）
        _plot_score_sweep(
            img_np, positions, keep_scores.numpy(), keep_deltas, args.patch, s_blank_score,
            output_dir / "keep_sweep.png",
            score_label="preserved-patch score s(x_keep)",
            delta_label="keep_delta = s_keep - s_blank",
            ref_label="s_blank",
        )
        # kept_ranked_{top,bottom}.png（top/bottom のみ。diff 着色なし: パッチ外が全部
        # 「変化した画素」になり画面が赤で埋まるため。赤枠の意味は「消した」→「残した」）
        # ref_cells: kept セルの正しい基準は s0 ではなく s_blank（preservation は分布外の
        # ほぼ空白キャンバス+断片のため）。白紙/黒紙を並置し、断片スコアが両極端とどれだけ
        # 区別できているか同一図内で確認できるようにする（--grid-refs で無効化可）。
        H, W = image.shape[-2:]
        ref_cells = None
        if args.grid_refs:
            ref_cells = [
                (np.full((H, W), _WHITE_VALUE), f"blank(white) s={s_blank_score:.2f}", "gray"),
                (np.full((H, W), _BLACK_VALUE), f"fill(black) s={s_fill_score:.2f}", "darkorange"),
            ]
        _plot_occluded_ranked(
            keep_occ, image, keep_scores.numpy(), positions, args.grid_n, output_dir,
            patch_size=args.patch, grid_style=args.grid_style, s0_score=s0_score,
            file_prefix="kept_ranked", band_names=("top", "bottom"), tint=False,
            box_label="kept", ref_cells=ref_cells,
        )
        _plot_necessity_vs_sufficiency(
            occ_scores.numpy(), keep_scores.numpy(), positions, image.shape[-2:],
            s0_score, s_blank_score, s_fill_score, output_dir / "necessity_vs_sufficiency.png",
        )
        # necessity_sufficiency.png: necessity(delta)/sufficiency(keep_delta) を
        # 1 枚の 2 変量空間マップ + 2D 凡例 + 対応配色の散布図に統合したダッシュボード。
        _plot_necessity_sufficiency_map(
            img_np, positions, deltas, keep_deltas, occ_scores.numpy(), keep_scores.numpy(),
            args.patch, s0_score, s_blank_score, s_fill_score,
            output_dir / "necessity_sufficiency.png",
        )

    deletion_auc = insertion_auc = None
    if args.curves:
        # 「消す」= 白紙に塗る、と一貫させるため s_blank と同じ _WHITE_VALUE を fill に使う
        # （--baseline はパッチの occlude/keep 塗りにのみ使う別の役割）。
        order = np.argsort(-deltas)  # delta 降順（|delta| ではない。正=realness を支える方から）
        del_batch, ins_batch, area_frac = _cumulative_curve_batches(
            image, positions, order, args.patch, _WHITE_VALUE
        )
        _, del_curve_t, _ = _forward_chunked(
            scorer, del_batch, args.embedding_layer, chunk, want_feats=False
        )
        _, ins_curve_t, _ = _forward_chunked(
            scorer, ins_batch, args.embedding_layer, chunk, want_feats=False
        )
        del_curve = del_curve_t.numpy()
        ins_curve = ins_curve_t.numpy()
        deletion_auc = float(np.trapezoid(del_curve, area_frac))
        insertion_auc = float(np.trapezoid(ins_curve, area_frac))
        _plot_deletion_insertion(
            area_frac, del_curve, ins_curve, deletion_auc, insertion_auc,
            output_dir / "deletion_insertion.png",
        )
        curve_metrics = pd.DataFrame([{
            "deletion_auc": deletion_auc, "insertion_auc": insertion_auc,
            "s0_score": s0_score, "s_blank_score": s_blank_score, "s_fill_score": s_fill_score,
            "n_patches": len(positions),
        }])
        curve_metrics.to_csv(output_dir / "curve_metrics.csv", index=False)
        print(f"Saved: {output_dir / 'curve_metrics.csv'}")

    # 特徴空間写像: 母集団（char-matched）+ 原画像 + 遮蔽点 を連結して一括 fit
    pop_sub, pop_labels, pop_scores = _subsample_population(
        df_pop, pop_feats, args.pop_samples, args.seed
    )
    X = np.vstack([pop_sub, orig_feat.numpy(), occ_feats.numpy()])
    emb, idx = fit_combined_embedding(X, reducer=args.reducer, max_samples=-1, seed=args.seed)
    # max_samples=-1 なので idx は 0..len(X)-1 の全件（順序保持）。
    n_pop = len(pop_sub)

    _plot_feature_embedding(
        emb, pop_labels, n_pop, occ_scores.numpy(), occ_thumbs, orig_thumb,
        args.max_occ_thumbs, args.thumb_zoom, output_dir,
    )
    _plot_feature_embedding_byscore(
        emb, pop_scores, n_pop, occ_scores.numpy(), occ_thumbs, orig_thumb,
        args.max_occ_thumbs, args.thumb_zoom, output_dir,
    )
    _plot_feature_embedding_by_position(
        emb, n_pop, positions, image.shape[-2:], occ_thumbs, orig_thumb,
        args.max_occ_thumbs, args.thumb_zoom, output_dir,
    )

    print(f"{char} {label_name}: s0={s0_score:.4f}  "
          f"occluded score range=[{occ_scores.min():.4f}, {occ_scores.max():.4f}]  "
          f"patches={len(positions)} (min_change={args.min_change})  "
          f"pop(char={char if args.pop_char is None else args.pop_char})={len(df_pop)}")

    metrics: dict[str, float] = {"s0_score": s0_score, "s0_logit": s0_logit}
    if need_blank:
        metrics["s_blank_score"] = s_blank_score
        metrics["s_blank_logit"] = s_blank_logit
        metrics["s_fill_score"] = s_fill_score
        metrics["s_fill_logit"] = s_fill_logit
    if args.keep:
        metrics["keep_score_max"] = float(keep_scores.max())
        metrics["keep_score_min"] = float(keep_scores.min())
    if args.curves:
        metrics["deletion_auc"] = deletion_auc
        metrics["insertion_auc"] = insertion_auc
    return metrics


def main() -> None:
    setup_logging()
    args = parse_args()
    cfg = load_config(args.config, args.overrides)

    chunk = args.batch_size if args.batch_size is not None else cfg["data"]["batch_size"]
    baseline = _resolve_baseline_arg(args.baseline)

    ckpt_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir) if args.output_dir else (
        ckpt_path.parent.parent / "occlusion_analysis"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump({**cfg, "analyze_occlusion": vars(args)}, f, allow_unicode=True, sort_keys=False)
    save_run_info(output_dir, extra={
        "checkpoint": args.checkpoint, "features": args.features, "split": args.split,
        "use_ema": args.use_ema, "patch": args.patch, "stride": args.stride,
        "baseline": args.baseline, "target": args.target, "min_change": args.min_change,
        "char": args.char, "all_chars": args.all_chars,
        "label": args.label, "n_images": args.n_images,
        "embedding_layer": args.embedding_layer, "reducer": args.reducer,
        "pop_samples": args.pop_samples, "pop_char": args.pop_char,
        "grid_n": args.grid_n, "grid_style": args.grid_style,
        "max_occ_thumbs": args.max_occ_thumbs,
        "keep": args.keep, "curves": args.curves, "grid_refs": args.grid_refs,
    })

    device = get_device()
    print(f"Device: {device}")

    module: ScorerModule = ScorerModule.load_from_checkpoint(
        args.checkpoint, map_location="cpu"
    )
    module.eval().to(device)
    if args.use_ema:
        module._ema.shadow_params = [p.to(device) for p in module._ema.shadow_params]
        module._ema.copy_to(module.scorer.parameters())
    scorer = module.scorer

    # SDF kwargs must mirror train_scorer.py / extract_scorer_features.py: an SDF-trained
    # Scorer expects SDF input, so omitting these silently feeds plain glyphs and makes
    # every occluded score (and thus the whole analysis) meaningless.
    dm = FontDataModule(
        real_data_dir=cfg["data"]["real_data_dir"],
        gen_data_dir=cfg["data"]["gen_data_dir"],
        mode="scorer",
        pair_sampling=cfg["data"].get("pair_sampling", "same_char"),
        charset=cfg["data"]["charset"],
        image_size=cfg["data"]["image_size"],
        batch_size=cfg["data"]["batch_size"],
        val_split=cfg["data"]["val_split"],
        test_split=cfg["data"]["test_split"],
        num_workers=cfg["data"].get("num_workers", 4),
        seed=cfg["data"]["seed"],
        binarize_threshold=cfg["data"].get("binarize_threshold"),
        split_file=cfg["data"].get("split_file"),
        gen_split_file=cfg["data"].get("gen_split_file"),
        use_sdf=cfg["data"].get("use_sdf", False),
        sdf_max_dist=cfg["data"].get("sdf_max_dist", 10.0),
        sdf_source=cfg["data"].get("sdf_source", "auto"),
        sdf_decode=cfg["data"].get("sdf_decode"),
        sdf_sharpness=cfg["data"].get("sdf_sharpness"),
        sdf_filter_width_px=cfg["data"].get("sdf_filter_width_px", 1.5),
    )
    dm.setup(None)
    loader = {"train": dm.train_dataloader, "val": dm.val_dataloader,
              "test": dm.test_dataloader}[args.split]()

    by_label = _collect_split(loader)
    rng = random.Random(args.seed)

    print(f"Loading population features from {args.features}...")
    df_meta, feats = load_features(args.features)
    if args.embedding_layer not in feats:
        raise ValueError(
            f"Layer '{args.embedding_layer}' not found in {args.features}. "
            f"Available: {sorted(feats)}"
        )
    pop_feats_full = feats[args.embedding_layer]

    labels = ["real", "gen"] if args.label == "both" else [args.label]
    manifest_rows: list[dict] = []

    def run(label_name: str, char_filter: list[str] | None) -> None:
        label_int = 1 if label_name == "real" else 0
        pool = by_label[label_int]
        if char_filter is not None:
            pool = [p for p in pool if p["char"] in char_filter]
            if not pool:
                available = sorted({p["char"] for p in by_label[label_int]})
                raise ValueError(
                    f"No samples for char={char_filter} label={label_name}. "
                    f"Available: {available}"
                )
        picks = rng.sample(pool, min(args.n_images, len(pool)))
        for i, pick in enumerate(picks):
            image = pick["image"].to(device)
            char = pick["char"]
            image_dir = output_dir / f"{char}_{label_name}_{i}"
            print(f"Analyzing {char} ({label_name}) [{i + 1}/{len(picks)}] -> {image_dir}")
            target_meta = {
                "font_family": pick["font_family"], "image_path": pick["image_path"],
                "split": args.split, "pick_index": i,
            }
            metrics = analyze_one_image(
                scorer, image, char, label_name, args, baseline, chunk,
                df_meta, pop_feats_full, image_dir, target_meta,
            )
            manifest_rows.append({
                "output_dir": str(image_dir.relative_to(output_dir)),
                "char": char, "label": label_name, **target_meta, **metrics,
            })

    if args.all_chars:
        # Each char gets its own n_images picks (rather than one pool spanning all chars).
        chars = args.char if args.char is not None else [
            chr(cp) for cp in _resolve_codepoints(cfg["data"]["charset"])
        ]
        for char in chars:
            for label_name in labels:
                run(label_name, [char])
    else:
        for label_name in labels:
            run(label_name, args.char)

    manifest_path = output_dir / "targets_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    print(f"Saved: {manifest_path} ({len(manifest_rows)} images)")


if __name__ == "__main__":
    main()
