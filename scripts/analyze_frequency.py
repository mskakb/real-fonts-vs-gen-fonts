"""周波数解析: 実フォント vs 生成フォントのスペクトル差分、およびスコアとの関係。

Scorer の判断根拠可視化（occlusion / Grad-CAM）は空間領域のみだったのに対し、本スクリプトは
周波数領域から「real/gen に周波数的な違いがあるか」「その違いは Scorer のスコアと相関するか」を
調べる。`extract_scorer_features.py` が出力した parquet（メタデータ + モデル入力サムネイル
`inputs/<split>/*.png`）だけを消費する純 CPU 解析であり、checkpoint も config も不要
（analyze_occlusion.py と違い、モデルへのフォワードを一切行わない）。

**入力表現に関する重要な注意**: サムネイルはモデルが実際に見た入力（`sdf_decode: hard` なら
real/gen とも 96px の二値画像）であり、`images/` の生ラスタ（real=アンチエイリアス／gen=完全
二値）や `sdf/` の生 SDF（real=量子化／gen=連続）のようなレンダリングパイプライン由来の非対称性は
入力段で意図的に潰されている。したがって本解析が検出しうるのは「二値化後に残る幾何的な差」
（ストローク太さ・曲率・端の形状・しきい値でのギザつき）の周波数的な現れであり、アンチエイリアス
アーティファクトそのものではない。

**最大の交絡はインク被覆率（ink_fraction）**: ストローク太さが違うだけでスペクトル形状は動く。
そのため常に `ink_only`（ink_fraction 単独）を probe のベースラインとして併記し、
`radial`（半径プロファイル全体）がそれと同程度なら「周波数の違い」ではなくインク量の違いだと
判断する（`analysis/spectral_analysis.py` のモジュール docstring も参照）。

Usage:
    uv run python scripts/analyze_frequency.py \\
        --features outputs/<exp>/features/scorer_features.parquet

    # 文字・枚数を絞ったスモークテスト
    uv run python scripts/analyze_frequency.py \\
        --features outputs/<exp>/features/scorer_features.parquet \\
        --char A B --n-per-char 100 --output-dir /path/to/smoke
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml
from sklearn.metrics import roc_curve

from real_fonts_vs_gen_fonts.analysis.feature_analysis import (
    has_images,
    load_feature_meta,
    set_font_scale,
    set_legend_loc,
)
from real_fonts_vs_gen_fonts.analysis.spectral_analysis import (
    FEATURE_SETS,
    compute_spectral_table,
    metric_separation_stats,
    per_char_spectral_auroc,
    plot_angular_profiles,
    plot_metric_distributions,
    plot_metric_ranked_images,
    plot_profile_by_score_bin,
    plot_radial_profiles,
    plot_radial_profiles_by_char,
    plot_score_vs_metric,
    plot_score_vs_probe,
    plot_spectral_roc,
    probe_results_table,
    score_metric_correlations,
    spectral_probe,
)
from real_fonts_vs_gen_fonts.datamodule import _resolve_codepoints
from real_fonts_vs_gen_fonts.utils.config import save_run_info, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Frequency-domain (2D FFT) analysis of real vs generated font glyphs, "
                    "and its relation to the Scorer realness score"
    )
    parser.add_argument("--features", required=True, metavar="PARQUET",
                        help="scorer_features.parquet from extract_scorer_features.py")
    parser.add_argument("--output-dir", default=None, metavar="DIR",
                        help="Output directory (default: <parquet's grandparent>/frequency_analysis)")
    parser.add_argument("--split", default="all", choices=["all", "train", "val", "test"],
                        help="Which split(s) to compute spectral features for (default: all)")
    parser.add_argument("--stats-split", default="test",
                        help="Split used for profile/distribution/correlation plots (default: test)")
    parser.add_argument("--fit-split", default="train",
                        help="Split used to fit the spectral linear probes (default: train)")
    parser.add_argument("--char", nargs="+", default=None, metavar="CHAR",
                        help="Restrict to specific character(s) (default: all charset chars)")
    parser.add_argument("--charset", default="uppercase",
                        help="Charset name for resolving the default --char list "
                            "(default: uppercase; only used when --char is omitted)")
    parser.add_argument("--n-per-char", type=int, default=None,
                        help="Cap images per (char, label, split) via random sampling "
                            "(default: use all available images)")
    parser.add_argument("--window", default="hann", choices=["hann", "none"],
                        help="FFT window (default: hann)")
    parser.add_argument("--normalize", default="per-image", choices=["per-image", "none"],
                        help="Radial/angular profile normalization (default: per-image)")
    parser.add_argument("--metric", default="hf_ratio",
                        help="Scalar metric used for the ranked-image grids (default: hf_ratio)")
    parser.add_argument("--grid-n", type=int, default=8,
                        help="N for N x N ranked-image grids (default: 8)")
    parser.add_argument("--cell-labels", action="store_true", default=True,
                        help="Show per-cell labels on image grids (default)")
    parser.add_argument("--no-cell-labels", dest="cell_labels", action="store_false",
                        help="Hide per-cell labels (pure image grids)")
    parser.add_argument("--font-scale", type=float, default=1.5,
                        help="Scale factor for all figure fonts (default: 1.5)")
    parser.add_argument("--legend-loc", default="best",
                        help="Legend placement: matplotlib loc, or 'outside' (default: best)")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed (default: 0)")
    parser.add_argument("--recompute", action="store_true",
                        help="Recompute spectral_features.parquet even if it already exists "
                            "in --output-dir (default: reuse if present)")
    parser.add_argument("--no-images", action="store_true",
                        help="Skip the ranked-image grids even if thumbnails are available")
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    set_font_scale(args.font_scale)
    set_legend_loc(args.legend_loc)

    features_path = Path(args.features)
    output_dir = Path(args.output_dir) if args.output_dir else (
        features_path.parent.parent / "frequency_analysis"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump({"analyze_frequency": vars(args)}, f, allow_unicode=True, sort_keys=False)
    save_run_info(output_dir, extra={
        "features": args.features, "split": args.split, "stats_split": args.stats_split,
        "fit_split": args.fit_split, "char": args.char, "n_per_char": args.n_per_char,
        "window": args.window, "normalize": args.normalize,
    })

    image_root = features_path.parent
    table_path = output_dir / "spectral_features.parquet"

    if table_path.exists() and not args.recompute:
        print(f"Reusing existing spectral table: {table_path}")
        df = pd.read_parquet(table_path)
    else:
        print(f"Loading feature metadata from {args.features}...")
        df_meta = load_feature_meta(args.features)
        if not has_images(df_meta, image_root):
            raise ValueError(
                f"No input thumbnails found under {image_root}. Was --save-inputs used "
                f"when extract_scorer_features.py generated this parquet?"
            )

        if args.split != "all":
            df_meta = df_meta[df_meta["split"] == args.split]

        chars = args.char if args.char is not None else [
            chr(cp) for cp in _resolve_codepoints(args.charset)
        ]
        df_meta = df_meta[df_meta["char"].isin(chars)]

        if args.n_per_char is not None:
            # shuffle then groupby().head(n): groupby().apply() silently drops the
            # group-key columns in this pandas version when the callback returns a
            # row subset of the same frame, so a plain per-group cap must avoid apply.
            df_meta = df_meta.sample(frac=1.0, random_state=args.seed)
            df_meta = (
                df_meta.groupby(["char", "label", "split"], group_keys=False)
                .head(args.n_per_char)
            )

        print(f"Computing spectral features for {len(df_meta)} images "
              f"(window={args.window}, normalize={args.normalize})...")
        df = compute_spectral_table(df_meta, image_root, window=args.window,
                                     normalize=args.normalize)
        df.to_parquet(table_path, index=False)
        print(f"Saved: {table_path} ({len(df)} rows)")

    stats_split = args.stats_split

    # --- profiles ---
    profiles_dir = output_dir / "profiles"
    plot_radial_profiles(df, profiles_dir, split=None)
    plot_radial_profiles(df, profiles_dir, split=stats_split)
    plot_radial_profiles_by_char(df, profiles_dir, split=stats_split)
    plot_angular_profiles(df, profiles_dir, split=None)
    plot_angular_profiles(df, profiles_dir, split=stats_split)
    plot_profile_by_score_bin(df, profiles_dir, split=stats_split)
    print(f"Saved profiles to {profiles_dir}")

    # --- scalar-metric distributions ---
    dist_dir = output_dir / "distributions"
    plot_metric_distributions(df, dist_dir, split=stats_split)
    print(f"Saved distributions to {dist_dir}")

    # --- stats CSVs ---
    stats_dir = output_dir / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    sep = metric_separation_stats(df, split=stats_split)
    sep.to_csv(stats_dir / "metric_separation.csv", index=False)
    auroc = per_char_spectral_auroc(df, split=stats_split, feature_set="radial")
    auroc.to_csv(stats_dir / "per_char_spectral_auroc.csv", index=False)
    corr = score_metric_correlations(df, split=stats_split)
    corr.to_csv(stats_dir / "score_metric_correlations.csv", index=False)
    print(f"Saved stats to {stats_dir}")
    if not sep.empty:
        ink_row = sep[sep["metric"] == "ink_fraction"]
        if not ink_row.empty:
            print(f"  ink_fraction AUROC (baseline) = {ink_row['auroc'].iloc[0]:.4f}")

    # --- spectral probes + ROC (vs the scorer itself) ---
    probe_dir = output_dir / "probe"
    probe_dir.mkdir(parents=True, exist_ok=True)
    probes = spectral_probe(df, fit_split=args.fit_split, eval_splits=[stats_split])
    probe_results_table(probes).to_csv(probe_dir / "probe_results.csv", index=False)
    scorer_curve = None
    eval_df = df[df["split"] == stats_split]
    if "score" in eval_df.columns and len(eval_df["label"].unique()) > 1:
        fpr, tpr, _ = roc_curve(eval_df["label"].to_numpy(), eval_df["score"].to_numpy())
        from sklearn.metrics import roc_auc_score
        auroc_scorer = float(roc_auc_score(eval_df["label"].to_numpy(), eval_df["score"].to_numpy()))
        scorer_curve = (fpr, tpr, auroc_scorer)
    plot_spectral_roc(probes, probe_dir, split=stats_split, scorer_curve=scorer_curve)
    print(f"Saved probe results to {probe_dir}")
    for name in FEATURE_SETS:
        if name in probes and stats_split in probes[name]:
            print(f"  probe[{name}] {stats_split} AUROC = {probes[name][stats_split][2]:.4f}")

    # --- score <-> frequency scatter plots ---
    scatter_dir = output_dir / "scatter"
    plot_score_vs_metric(df, scatter_dir, split=stats_split)
    if "radial" in probes:
        from real_fonts_vs_gen_fonts.analysis.feature_analysis import fit_linear_probe

        fit_df = df[df["split"] == args.fit_split]
        cols = FEATURE_SETS["radial"]
        scaler, clf = fit_linear_probe(
            fit_df[cols].to_numpy(dtype=float), fit_df["label"].to_numpy()
        )
        proba = clf.predict_proba(scaler.transform(eval_df[cols].to_numpy(dtype=float)))[:, 1]
        plot_score_vs_probe(df, proba, scatter_dir, split=stats_split, feature_set_name="radial")
    print(f"Saved scatter plots to {scatter_dir}")

    # --- ranked image grids ---
    if not args.no_images and has_images(df, image_root):
        images_dir = output_dir / "images"
        plot_metric_ranked_images(
            df, image_root, images_dir, metric=args.metric, split=stats_split,
            grid_n=args.grid_n, show_labels=args.cell_labels,
        )
        print(f"Saved ranked image grids to {images_dir}")

    print("Done.")


if __name__ == "__main__":
    main()
