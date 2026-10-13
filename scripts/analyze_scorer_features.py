"""Analyze a trained ScorerModule's features from a scorer-features parquet.

CPU-side driver: reads the parquet written by extract_scorer_features.py (per-layer
GAP features + metadata) and produces, under <output-dir>/:
  - per-layer linear probe AUROC + ROC curves (probe fit on train, applied to each split)
  - a single shared 2D embedding (PCA / t-SNE / UMAP, fit once over all splits),
    coloured by real/gen (hue) × split (shade, marker too), by realness score, and by char
  - per-char / per-font realness-score stats (CSV) + plots, for --stats-split
    (a single split, or "all" to emit a plot per available split). per-font bars and
    representative images now cover lowest/middle/highest bands, not just the extremes.
    (per-char AUROC is well-defined; per-font is single-class for real fonts → stats only)
  - score / logit distributions (real vs gen histograms) and KDE densities annotated
    with each class mean, outer tails and the confusion zone s*=argmax min(p_real,p_gen)
  - score_separation.csv: per-split real/gen means, confusion s* and overlap
    coefficient OVL=∫min (a 0..1 separability summary)
  - image grids of model-input thumbnails (only when the parquet has an image_path
    column with existing thumbnails; disable with --no-images): each is a single-category
    N x N grid (--grid-n, default 8) with the selection encoded in the filename and
    per-cell labels toggled by --no-cell-labels. Kinds: score-ranked
    (ranked_{real,gen}_{top,middle,bottom}), high-confidence misclassifications
    (misclassified_{real,gen}), per-char/per-font representatives
    (repr_<key>_{lowest,middle,highest}), worst-AUROC chars (worst_auroc_{real,gen}),
    KDE-region samples (kde_{real,gen}_{mean,tail,overlap}), and an embedding overlaid
    with glyphs

All analysis logic lives in real_fonts_vs_gen_fonts.analysis.feature_analysis so this
script and notebooks/scorer_feature_analysis.ipynb share one implementation.

Usage:
    # from a parquet (preferred: extract once, analyze many times)
    uv run python scripts/analyze_scorer_features.py \\
        --features outputs/<exp>/features/scorer_features.parquet \\
        --reducer umap

    # convenience: extract on the fly from a checkpoint, then analyze
    uv run python scripts/analyze_scorer_features.py \\
        --checkpoint outputs/<exp>/checkpoints/last.ckpt \\
        --config config/scorer/scorer.yaml \\
        data.real_data_dir=data/raw/real data.gen_data_dir=data/raw/generated
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from real_fonts_vs_gen_fonts.analysis import (
    fit_combined_embedding,
    fit_linear_probe,
    has_images,
    layer_probe_auroc,
    load_features,
    per_char_auroc,
    per_font_auroc,
    per_group_stats,
    plot_combined_embedding,
    plot_embedding_thumbnails,
    plot_group_representative_images,
    plot_group_stats,
    plot_kde_region_images,
    plot_misclassified_images,
    plot_roc_curves,
    plot_score_distributions,
    plot_score_kde,
    plot_score_ranked_images,
    plot_worst_auroc_char_examples,
    score_separation_stats,
    set_font_scale,
    set_legend_loc,
)
from real_fonts_vs_gen_fonts.utils.config import save_run_info

LEGEND_LOCS = [
    "outside", "best", "upper right", "upper left", "lower left", "lower right",
    "center left", "center right", "upper center", "lower center", "center",
]

SPLIT_ORDER = ["train", "val", "test"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Per-layer probe + combined-space embedding + per-group stats"
    )
    src = parser.add_argument_group("feature source (one required)")
    src.add_argument("--features", metavar="PARQUET",
                     help="scorer_features.parquet from extract_scorer_features.py")
    src.add_argument("--checkpoint", metavar="CKPT",
                     help="ScorerModule checkpoint; extract features on the fly if "
                          "--features is not given")
    parser.add_argument("--config", default="config/scorer/scorer.yaml", metavar="FILE",
                        help="YAML config (only used with --checkpoint)")
    parser.add_argument("--output-dir", metavar="DIR",
                        help="Output directory (default: <features_dir>/../feature_analysis/)")
    parser.add_argument("--reducer", default="tsne", choices=["pca", "tsne", "umap"],
                        help="Dimensionality reduction for the combined embedding")
    parser.add_argument("--embedding-layer", default="layer4",
                        help="Which layer's features to embed (default: layer4)")
    parser.add_argument("--embedding-samples", type=int, default=10000,
                        help="Subsample cap for t-SNE/UMAP fit (-1 for all)")
    parser.add_argument("--stats-split", default="all",
                        choices=["train", "val", "test", "all"],
                        help="Which split to plot per-group stats for, or 'all' to "
                             "plot every available split (default: test)")
    parser.add_argument("--no-images", dest="images", action="store_false", default=True,
                        help="Skip all image-grid plots (run even if thumbnails exist)")
    parser.add_argument("--grid-n", type=int, default=8,
                        help="N for the single-category N x N image grids (default: 8 = 64)")
    parser.add_argument("--cell-labels", dest="cell_labels", action="store_true", default=True,
                        help="Annotate each grid cell with its score/char (default)")
    parser.add_argument("--no-cell-labels", dest="cell_labels", action="store_false",
                        help="Pure image grids; identify via the filename only")
    parser.add_argument("--font-cell-labels", dest="font_cell_labels", action="store_true",
                        default=True,
                        help="Annotate repr_font_family cells with font name/score "
                             "(default). Independent of --cell-labels: font_family "
                             "names repeated across a row tend to overlap, so this can "
                             "be turned off without affecting other grids")
    parser.add_argument("--no-font-cell-labels", dest="font_cell_labels", action="store_false",
                        help="Pure repr_font_family image grids (font name only in filename)")
    parser.add_argument("--n-thumbs", type=int, default=10,
                        help="Max thumbnails to overlay on the embedding plot")
    parser.add_argument("--kde-bw", type=float, default=None,
                        help="KDE bandwidth for score densities (default: Scott's rule)")
    parser.add_argument("--font-scale", type=float, default=1.8,
                        help="Multiplier for all figure font sizes "
                             "(captions, titles, labels, legends; default: 1.5)")
    parser.add_argument("--legend-loc", default="best", choices=LEGEND_LOCS,
                        metavar="LOC",
                        help="Legend placement: 'outside' (right of the axes, never "
                             "covers data but shrinks the plot) or a matplotlib loc "
                             "kept inside the axes (default: best). Choices: "
                             + ", ".join(LEGEND_LOCS))
    parser.add_argument("--use-ema", action="store_true", default=True,
                        help="Use EMA weights for on-the-fly extraction (default)")
    parser.add_argument("--no-ema", dest="use_ema", action="store_false",
                        help="Use raw (non-EMA) weights for on-the-fly extraction")
    parser.add_argument("overrides", nargs="*", metavar="key=value",
                        help="Config overrides for on-the-fly extraction")
    return parser.parse_args()


def _extract_on_the_fly(args: argparse.Namespace, feat_dir: Path) -> Path:
    """Run extract_scorer_features.py to produce a parquet, return its path."""
    feat_dir.mkdir(parents=True, exist_ok=True)
    extract_script = Path(__file__).with_name("extract_scorer_features.py")
    cmd = [
        sys.executable, str(extract_script),
        "--checkpoint", args.checkpoint,
        "--config", args.config,
        "--output-dir", str(feat_dir),
        "--split", "all",
    ]
    if not args.use_ema:
        cmd.append("--no-ema")
    cmd += args.overrides
    print(f"Extracting features → {feat_dir}")
    subprocess.run(cmd, check=True)
    return feat_dir / "scorer_features.parquet"


def main() -> None:
    args = parse_args()
    set_font_scale(args.font_scale)
    set_legend_loc(args.legend_loc)

    if args.features:
        parquet_path = Path(args.features)
    elif args.checkpoint:
        ckpt = Path(args.checkpoint)
        feat_dir = Path(args.output_dir) / "features" if args.output_dir \
            else ckpt.parent.parent / "features"
        parquet_path = _extract_on_the_fly(args, feat_dir)
    else:
        sys.exit("Either --features or --checkpoint is required.")

    output_dir = Path(args.output_dir) if args.output_dir \
        else parquet_path.parent.parent / "feature_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_run_info(output_dir, extra={
        "features": str(parquet_path),
        "reducer": args.reducer,
        "embedding_layer": args.embedding_layer,
        "stats_split": args.stats_split,
        "images": args.images,
        "grid_n": args.grid_n,
        "cell_labels": args.cell_labels,
        "font_cell_labels": args.font_cell_labels,
        "font_scale": args.font_scale,
        "legend_loc": args.legend_loc,
    })

    df_meta, feats = load_features(parquet_path)
    labels = df_meta["label"].to_numpy()
    split_col = df_meta["split"].to_numpy()
    splits = [s for s in SPLIT_ORDER if s in set(split_col)]
    print(f"Loaded {len(df_meta):,} samples | splits={splits} | layers={list(feats)}")

    # ---- per-layer linear probe (fit on train, apply to each split) ----
    train_mask = split_col == "train"
    if train_mask.any():
        records: dict[str, list[dict]] = {s: [] for s in splits}
        layer_curves: dict[str, dict] = {s: {} for s in splits}
        for layer, X in feats.items():
            scaler, clf = fit_linear_probe(X[train_mask], labels[train_mask])
            for s in splits:
                m = split_col == s
                fpr, tpr, auroc = layer_probe_auroc(scaler, clf, X[m], labels[m])
                layer_curves[s][layer] = (fpr, tpr, auroc)
                records[s].append({"layer": layer, "dim": X.shape[1], "auroc": auroc})
                print(f"  [{s}] {layer} probe AUROC={auroc:.4f}")

        for s in splits:
            m = split_col == s
            scores = df_meta["score"].to_numpy()[m]
            y = labels[m]
            scorer_fpr, scorer_tpr, _ = roc_curve(y, scores)
            scorer_auroc = float(roc_auc_score(y, scores))
            records[s].append({"layer": "scorer", "dim": None, "auroc": scorer_auroc})
            split_dir = output_dir / s
            split_dir.mkdir(parents=True, exist_ok=True)
            plot_roc_curves(layer_curves[s], (scorer_fpr, scorer_tpr, scorer_auroc),
                            split_dir / "roc_curves.png")
            pd.DataFrame(records[s]).to_csv(split_dir / "probe_results.csv", index=False)
    else:
        print("No train split found → skipping per-layer probe.")

    # ---- combined-space embedding (single layer, fit once over all splits) ----
    if args.embedding_layer not in feats:
        sys.exit(f"--embedding-layer {args.embedding_layer!r} not in parquet "
                 f"(available: {list(feats)})")
    print(f"Fitting {args.reducer.upper()} embedding on {args.embedding_layer}…")
    emb, idx = fit_combined_embedding(
        feats[args.embedding_layer], reducer=args.reducer,
        max_samples=args.embedding_samples,
    )
    embedding_dir = output_dir / "embedding"
    plot_combined_embedding(emb, df_meta.iloc[idx], embedding_dir, args.reducer)

    # ---- per-char / per-font stats ----
    stats_dir = output_dir / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)

    char_stats = per_group_stats(df_meta, "char")
    char_stats.to_csv(stats_dir / "per_char_stats.csv", index=False)
    char_auroc = per_char_auroc(df_meta)
    char_auroc.to_csv(stats_dir / "per_char_auroc.csv", index=False)

    font_stats = per_group_stats(df_meta, "font_family")
    font_stats.to_csv(stats_dir / "per_font_stats.csv", index=False)
    font_auroc = per_font_auroc(df_meta)
    font_auroc.to_csv(stats_dir / "per_font_auroc.csv", index=False)

    stats_splits = splits if args.stats_split == "all" else [args.stats_split]
    for s in stats_splits:
        plot_group_stats(char_stats, "char", stats_dir, split=s, auroc=char_auroc)
        plot_group_stats(font_stats, "font_family", stats_dir, split=s, auroc=font_auroc)

    # ---- score / logit distributions + KDE (real vs gen) ----
    dist_dir = output_dir / "distributions"
    plot_score_distributions(df_meta, dist_dir, split=None)
    plot_score_kde(df_meta, dist_dir, split=None, bw=args.kde_bw)
    for s in stats_splits:
        plot_score_distributions(df_meta, dist_dir, split=s)
        plot_score_kde(df_meta, dist_dir, split=s, bw=args.kde_bw)

    # ---- real/gen score-separation scalars (OVL, confusion s*, means) ----
    sep_rows = [{"split": "all", **score_separation_stats(df_meta, None, bw=args.kde_bw)}]
    sep_rows += [{"split": s, **score_separation_stats(df_meta, s, bw=args.kde_bw)}
                 for s in splits]
    pd.DataFrame(sep_rows).to_csv(output_dir / "score_separation.csv", index=False)

    # ---- image grids (model-input thumbnails) ----
    image_root = parquet_path.parent
    if not args.images:
        print("Skipping image plots (--no-images).")
    elif not has_images(df_meta, image_root):
        print("Skipping image plots (no image_path / thumbnails in parquet; "
              "re-extract without --no-save-inputs).")
    else:
        img_dir = output_dir / "images"
        plot_embedding_thumbnails(emb, df_meta.iloc[idx], image_root, embedding_dir,
                                  args.reducer, max_thumbs=args.n_thumbs)
        gn, cl = args.grid_n, args.cell_labels
        for s in stats_splits:
            plot_score_ranked_images(df_meta, image_root, img_dir, split=s,
                                     grid_n=gn, show_labels=cl)
            plot_misclassified_images(df_meta, image_root, img_dir, split=s,
                                      grid_n=gn, show_labels=cl)
            plot_group_representative_images(df_meta, image_root, img_dir, "char", split=s,
                                             grid_n=gn, show_labels=cl)
            plot_group_representative_images(df_meta, image_root, img_dir, "font_family",
                                             split=s, grid_n=gn, show_labels=args.font_cell_labels)
            plot_worst_auroc_char_examples(df_meta, char_auroc, image_root, img_dir, split=s,
                                           grid_n=gn, show_labels=cl)
            plot_kde_region_images(df_meta, image_root, img_dir, split=s,
                                   grid_n=gn, bw=args.kde_bw, show_labels=cl)

    print(f"\nSaved → {output_dir}")


if __name__ == "__main__":
    main()
