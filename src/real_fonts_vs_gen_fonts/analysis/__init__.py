"""Scorer feature analysis (CPU-side, parquet-driven).

`extract_scorer_features.py` writes a parquet with per-layer GAP features + metadata;
the pure functions here consume that parquet and produce the derived analysis
(combined-space embeddings, per-layer linear probes, per-char / per-font stats).
Both `analyze_scorer_features.py` and `notebooks/scorer_feature_analysis.ipynb`
import these functions so the logic lives in one place.
"""

from __future__ import annotations

from real_fonts_vs_gen_fonts.analysis.feature_analysis import (
    LAYER_NAMES,
    LAYER_PREFIXES,
    fit_combined_embedding,
    fit_linear_probe,
    has_images,
    layer_probe_auroc,
    load_feature_meta,
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

__all__ = [
    "LAYER_NAMES",
    "LAYER_PREFIXES",
    "fit_combined_embedding",
    "fit_linear_probe",
    "has_images",
    "layer_probe_auroc",
    "load_feature_meta",
    "load_features",
    "per_char_auroc",
    "per_font_auroc",
    "per_group_stats",
    "plot_combined_embedding",
    "plot_embedding_thumbnails",
    "plot_group_representative_images",
    "plot_group_stats",
    "plot_kde_region_images",
    "plot_misclassified_images",
    "plot_roc_curves",
    "plot_score_distributions",
    "plot_score_kde",
    "plot_score_ranked_images",
    "plot_worst_auroc_char_examples",
    "score_separation_stats",
    "set_font_scale",
    "set_legend_loc",
]
