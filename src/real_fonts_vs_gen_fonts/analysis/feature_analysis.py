"""Pure functions for Scorer feature analysis.

Consumes the parquet written by `extract_scorer_features.py` (per-layer GAP
features + metadata) and produces:
  - combined-space embeddings (PCA / t-SNE / UMAP, fit once over all splits)
  - per-layer linear-probe AUROC + ROC curves
  - per-char / per-font realness-score statistics
  - score / logit distributions and image grids (model-input thumbnails) for
    score-ranked, misclassified, per-group-representative and worst-AUROC samples

No model / GPU is required here: everything runs on the parquet (image plots
additionally read the input thumbnails saved next to it via the image_path column).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

# Layer names (as used by RealnessScorer.extract_intermediate) and the parquet
# column prefixes they map to. "layer1" -> columns "l1_0", "l1_1", ...
LAYER_NAMES = ["layer1", "layer2", "layer3", "layer4"]
LAYER_PREFIXES = {name: f"l{i + 1}_" for i, name in enumerate(LAYER_NAMES)}

META_COLS = ["split", "label", "char", "font_family", "score", "logit", "image_path"]

# real = blue, generated = orange (matches utils/visualization.py palette)
REAL_COLOR = "#2196F3"
GEN_COLOR = "#FF5722"
SPLIT_MARKERS = {"train": "o", "val": "^", "test": "s"}

# Combined (label, split) palette for the shared embedding: hue encodes real/gen
# (blue vs orange), lightness encodes split (light=train → dark=test). Marker shape
# (SPLIT_MARKERS) doubles as a redundant split cue.
LABEL_SPLIT_COLORS = {
    (1, "train"): "#90CAF9", (1, "val"): "#1E88E5", (1, "test"): "#0D47A1",
    (0, "train"): "#FFCC80", (0, "val"): "#FB8C00", (0, "test"): "#E65100",
}


# --------------------------------------------------------------------------- #
# Font sizing
# --------------------------------------------------------------------------- #
# A single scale knob enlarges every figure font at once. _fs() scales the explicit
# fontsize= values (image-grid cell labels, suptitles, in-axes annotations); the
# matplotlib rcParams cover the calls that omit fontsize (axes titles, axis labels,
# ticks, legends). Both are driven by set_font_scale so one CLI/notebook value
# (--font-scale) controls captions and labels uniformly across all plots.
_FONT_SCALE = 1.0


def set_font_scale(scale: float) -> None:
    """Scale every analysis-figure font size (captions, titles, labels, legends).

    Multiplies both the explicit fontsizes routed through _fs() and the matplotlib
    rcParams used by plot calls that omit fontsize. Call once before plotting.
    """
    global _FONT_SCALE
    _FONT_SCALE = scale
    plt.rcParams.update({
        "font.size": 10.0 * scale,
        "axes.titlesize": 11.0 * scale,
        "axes.labelsize": 10.0 * scale,
        "xtick.labelsize": 9.0 * scale,
        "ytick.labelsize": 9.0 * scale,
        "legend.fontsize": 9.0 * scale,
        "figure.titlesize": 13.0 * scale,
    })


def _fs(size: float) -> float:
    """Apply the current font scale to an explicit fontsize value."""
    return size * _FONT_SCALE


# Legend placement. "outside" puts the legend to the right of the axes (never covers
# the data, but widens the figure so the plot looks smaller); any matplotlib loc
# string ("best", "upper right", …) keeps it inside, preserving the plot size at the
# risk of overlap when fonts are large. Selectable via set_legend_loc / --legend-loc.
_LEGEND_LOC = "best"


def set_legend_loc(loc: str) -> None:
    """Set legend placement for all analysis figures ("outside" or a matplotlib loc)."""
    global _LEGEND_LOC
    _LEGEND_LOC = loc


def _legend(ax: "plt.Axes", **kwargs) -> None:
    """Add a legend at the configured location (_LEGEND_LOC).

    For "outside" the legend sits to the right of the axes; pair with
    savefig(bbox_inches="tight") so it is not clipped. Otherwise the loc is passed
    straight to ax.legend (kept inside the axes).
    """
    if _LEGEND_LOC == "outside":
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0, **kwargs)
    else:
        ax.legend(loc=_LEGEND_LOC, **kwargs)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_features(
    parquet_path: str | Path,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Load a scorer-features parquet into metadata + per-layer arrays.

    Args:
        parquet_path: path to scorer_features.parquet (from extract_scorer_features.py).
    Returns:
        (df_meta, feats):
            df_meta: DataFrame with the metadata columns present among META_COLS
                     (index reset to 0..N-1, aligned with the feature arrays).
            feats:   dict layer_name -> np.ndarray (N, C_layer), columns ordered by
                     their numeric suffix. Layers absent from the parquet are skipped.
    """
    df = pd.read_parquet(parquet_path).reset_index(drop=True)
    meta_cols = [c for c in META_COLS if c in df.columns]
    df_meta = df[meta_cols].copy()

    feats: dict[str, np.ndarray] = {}
    for name, prefix in LAYER_PREFIXES.items():
        cols = [c for c in df.columns if c.startswith(prefix)]
        if not cols:
            continue
        cols = sorted(cols, key=lambda c: int(c[len(prefix):]))
        feats[name] = df[cols].to_numpy(dtype=np.float32)
    return df_meta, feats


def load_feature_meta(parquet_path: str | Path) -> pd.DataFrame:
    """Load only the metadata columns of a scorer-features parquet.

    Unlike load_features(), this never materializes the l1_*..l4_* feature columns
    (~3GB for the shipped parquets), so callers that only need split/label/char/
    font_family/score/logit/image_path (e.g. a frequency-domain analysis) can read
    the file cheaply.
    """
    schema_cols = set(pq.ParquetFile(parquet_path).schema.names)
    meta_cols = [c for c in META_COLS if c in schema_cols]
    return pd.read_parquet(parquet_path, columns=meta_cols).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Combined-space embedding
# --------------------------------------------------------------------------- #
def fit_combined_embedding(
    X: np.ndarray,
    reducer: str = "umap",
    standardize: bool = True,
    max_samples: int = -1,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a 2D embedding once over all rows of X (all splits combined).

    Args:
        X:           feature matrix (N, D).
        reducer:     "pca" | "tsne" | "umap".
        standardize: z-score features before reduction.
        max_samples: subsample cap before fitting (t-SNE/UMAP scale poorly).
                     <= 0 means use all rows. PCA ignores this (cheap & exact).
        seed:        RNG seed for subsampling and the reducer.
    Returns:
        (emb, idx): emb (M, 2) and idx (M,) the row indices of X used, so callers
                    can align metadata via df_meta.iloc[idx].
    """
    reducer = reducer.lower()
    n = len(X)
    if reducer != "pca" and 0 < max_samples < n:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n, max_samples, replace=False))
    else:
        idx = np.arange(n)
    Xs = X[idx]

    if standardize:
        Xs = StandardScaler().fit_transform(Xs)

    if reducer == "pca":
        emb = PCA(n_components=2, random_state=seed).fit_transform(Xs)
    elif reducer == "tsne":
        perplexity = min(30.0, max(5.0, (len(Xs) - 1) / 3.0))
        emb = TSNE(
            n_components=2, perplexity=perplexity, random_state=seed, n_jobs=-1
        ).fit_transform(Xs)
    elif reducer == "umap":
        from umap import UMAP

        emb = UMAP(n_components=2, random_state=seed, n_jobs=1).fit_transform(Xs)
    else:
        raise ValueError(f"Unknown reducer: {reducer!r} (pca | tsne | umap)")
    return emb, idx


def plot_combined_embedding(
    emb: np.ndarray,
    meta: pd.DataFrame,
    output_dir: Path,
    reducer_name: str,
    point_size: float = 6.0,
) -> None:
    """Save three views of a single shared 2D embedding.

    `meta` must be row-aligned with `emb` (e.g. df_meta.iloc[idx]) and contain
    columns split, label, score, char.

    Produces, in `output_dir`:
        embed_<reducer>_by_label_split.png  (color=real/gen, marker=split)
        embed_<reducer>_by_score.png        (continuous realness score)
        embed_<reducer>_by_char.png         (categorical char)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    meta = meta.reset_index(drop=True)
    split = meta["split"].to_numpy()
    label = meta["label"].to_numpy()

    # (1) real/gen as hue, split as lightness (+ marker shape as redundant split cue)
    fig, ax = plt.subplots(figsize=(7, 6))
    for sp, marker in SPLIT_MARKERS.items():
        for lab, name in [(1, "real"), (0, "gen")]:
            mask = (split == sp) & (label == lab)
            if not mask.any():
                continue
            ax.scatter(
                emb[mask, 0], emb[mask, 1], c=LABEL_SPLIT_COLORS[(lab, sp)],
                marker=marker, label=f"{name}-{sp}", alpha=0.5, s=point_size,
                linewidths=0,
            )
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    _legend(ax, markerscale=2, fontsize=_fs(8))
    fig.tight_layout()
    fig.savefig(output_dir / f"embed_{reducer_name}_by_label_split.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # (2) continuous realness score
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(
        emb[:, 0], emb[:, 1], c=meta["score"].to_numpy(), cmap="RdYlGn",
        vmin=0.0, vmax=1.0, alpha=0.6, s=point_size, linewidths=0,
    )
    fig.colorbar(sc, ax=ax, label="realness score")
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    fig.tight_layout()
    fig.savefig(output_dir / f"embed_{reducer_name}_by_score.png", dpi=150)
    plt.close(fig)

    # (3) categorical char
    chars = sorted(meta["char"].unique())
    cmap = plt.get_cmap("tab20")
    char_arr = meta["char"].to_numpy()
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, ch in enumerate(chars):
        mask = char_arr == ch
        ax.scatter(
            emb[mask, 0], emb[mask, 1], color=cmap(i % 20), label=str(ch),
            alpha=0.5, s=point_size, linewidths=0,
        )
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    if len(chars) <= 30:
        _legend(ax, markerscale=2, fontsize=_fs(7), ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / f"embed_{reducer_name}_by_char.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Per-layer linear probe
# --------------------------------------------------------------------------- #
def fit_linear_probe(
    X: np.ndarray, y: np.ndarray
) -> tuple[StandardScaler, LogisticRegression]:
    """Fit StandardScaler + LogisticRegression (linear probe) on (X, y)."""
    scaler = StandardScaler()
    clf = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs")
    clf.fit(scaler.fit_transform(X), y)
    return scaler, clf


def layer_probe_auroc(
    scaler: StandardScaler,
    clf: LogisticRegression,
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Apply a fitted probe to (X, y); return (fpr, tpr, auroc)."""
    proba = clf.predict_proba(scaler.transform(X))[:, 1]
    fpr, tpr, _ = roc_curve(y, proba)
    return fpr, tpr, float(roc_auc_score(y, proba))


def plot_roc_curves(
    layer_curves: dict[str, tuple[np.ndarray, np.ndarray, float]],
    scorer_curve: tuple[np.ndarray, np.ndarray, float] | None,
    output_path: Path,
) -> None:
    """Plot per-layer probe ROC curves (+ the scorer itself if provided)."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=0.8, label="chance")
    for layer_name, (fpr, tpr, auroc) in layer_curves.items():
        ax.plot(fpr, tpr, label=f"{layer_name} (AUROC={auroc:.3f})")
    if scorer_curve is not None:
        fpr, tpr, auroc = scorer_curve
        ax.plot(fpr, tpr, color="black", linewidth=2, label=f"scorer (AUROC={auroc:.3f})")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    _legend(ax, fontsize=_fs(8))
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Per-group statistics (char / font_family)
# --------------------------------------------------------------------------- #
def per_group_stats(df_meta: pd.DataFrame, group_key: str) -> pd.DataFrame:
    """Realness-score stats grouped by (group_key, split, label).

    Args:
        df_meta:   metadata DataFrame with columns split, label, score, and group_key.
        group_key: "char" or "font_family".
    Returns:
        Long-form DataFrame: [group_key, split, label, count, mean, std, median, min, max].
        label is rendered as "real"/"gen".
    """
    g = df_meta.groupby([group_key, "split", "label"])["score"]
    stats = g.agg(["count", "mean", "std", "median", "min", "max"]).reset_index()
    stats["label"] = stats["label"].map({1: "real", 0: "gen"})
    return stats


def per_char_auroc(df_meta: pd.DataFrame) -> pd.DataFrame:
    """Per-char real-vs-gen AUROC of the realness score, by split.

    AUROC is only defined where both classes are present, so only chars/splits with
    both real and gen samples are returned.
    """
    rows: list[dict] = []
    for (char, split), grp in df_meta.groupby(["char", "split"]):
        y = grp["label"].to_numpy()
        if len(np.unique(y)) < 2:
            continue
        rows.append({
            "char": char,
            "split": split,
            "auroc": float(roc_auc_score(y, grp["score"].to_numpy())),
            "n": int(len(grp)),
        })
    return pd.DataFrame(rows)


def per_font_auroc(df_meta: pd.DataFrame, min_real: int = 5) -> pd.DataFrame:
    """Per real-font_family AUROC of the realness score against the pooled gen scores.

    Real fonts are single-class (every real sample has label=1), so there is no
    per-font gen counterpart to pair against directly (unlike per_char_auroc, where
    both classes share the same char). Instead each font's real scores are pooled
    against *every* gen score in the same split, so the metric answers "how
    separable is this font from the overall generated distribution" rather than a
    matched real-vs-gen-of-that-font AUROC. The generated bucket (font_family in
    {"generated", "guided"}) is excluded from the real side. Fonts with fewer than
    min_real samples are skipped (too few for a stable ROC).
    """
    rows: list[dict] = []
    for split, grp in df_meta.groupby("split"):
        gen = grp[grp["label"] == 0]
        if gen.empty:
            continue
        gen_scores = gen["score"].to_numpy()
        real = grp[(grp["label"] == 1) & (~grp["font_family"].isin(["generated", "guided"]))]
        for font, fg in real.groupby("font_family"):
            if len(fg) < min_real:
                continue
            y = np.concatenate([np.ones(len(fg)), np.zeros(len(gen_scores))])
            s = np.concatenate([fg["score"].to_numpy(), gen_scores])
            rows.append({
                "font_family": font,
                "split": split,
                "auroc": float(roc_auc_score(y, s)),
                "n_real": int(len(fg)),
                "n_gen": int(len(gen_scores)),
            })
    return pd.DataFrame(rows)


def plot_group_stats(
    stats: pd.DataFrame,
    group_key: str,
    output_dir: Path,
    split: str = "test",
    auroc: pd.DataFrame | None = None,
    top_n: int = 20,
) -> None:
    """Visualize per-group score stats for one split.

    For group_key="char": grouped bar of mean score (real vs gen) per char, plus a
    per-char AUROC bar if `auroc` is provided.
    For group_key="font_family": horizontal bar of mean real score in lowest /
    middle / highest `top_n` bands (the lowest band = most "fake-looking" real fonts;
    the middle band = typical fonts), with dividers between bands, plus a same-shaped
    per-font AUROC bar (real font vs pooled gen, see per_font_auroc) if `auroc` is
    provided. When there are at most 3*top_n fonts all are shown. The single generated
    bucket is excluded (its mean is the same as the gen baseline).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    s = stats[stats["split"] == split]

    if group_key == "char":
        chars = sorted(s[group_key].unique())
        x = np.arange(len(chars))
        fig, ax = plt.subplots(figsize=(max(6, len(chars) * 0.4), 5))
        for off, lab, color in [(-0.2, "real", REAL_COLOR), (0.2, "gen", GEN_COLOR)]:
            sub = s[s["label"] == lab].set_index(group_key).reindex(chars)
            ax.bar(x + off, sub["mean"].to_numpy(), width=0.4, color=color,
                   yerr=sub["std"].to_numpy(), capsize=2, label=lab)
        ax.set_xticks(x)
        ax.set_xticklabels([str(c) for c in chars])
        ax.set_ylabel("mean realness score")
        _legend(ax)
        fig.tight_layout()
        fig.savefig(output_dir / f"per_char_score_{split}.png", dpi=150,
                    bbox_inches="tight")
        plt.close(fig)

        if auroc is not None and not auroc.empty:
            a = auroc[auroc["split"] == split].sort_values("char")
            fig, ax = plt.subplots(figsize=(max(6, len(a) * 0.4), 5))
            ax.bar(a["char"].astype(str), a["auroc"].to_numpy(), color="#607D8B")
            ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
            ax.set_ylabel("real-vs-gen AUROC")
            fig.tight_layout()
            fig.savefig(output_dir / f"per_char_auroc_{split}.png", dpi=150)
            plt.close(fig)
        return

    if group_key == "font_family":
        real = s[(s["label"] == "real")].copy()
        real = real[real[group_key].isin(["generated", "guided"]) == False]  # noqa: E712
        if real.empty:
            return

        def _banded_barh(df: pd.DataFrame, value_col: str, err_col: str | None,
                          xlabel: str, out_name: str, color: str) -> None:
            df = df.sort_values(value_col).reset_index(drop=True)
            # lowest / middle / highest top_n bands so the "typical" fonts are visible
            # too (not just the extremes). Fall back to all fonts when too few.
            bands: list[tuple[str, pd.DataFrame]] = []
            if len(df) <= 3 * top_n:
                bands.append(("all", df))
            else:
                mid_start = len(df) // 2 - top_n // 2
                bands = [
                    ("lowest", df.iloc[:top_n]),
                    ("middle", df.iloc[mid_start:mid_start + top_n]),
                    ("highest", df.iloc[len(df) - top_n:]),
                ]
            # draw bottom→top so "lowest" sits at the bottom; track band boundaries.
            sel = pd.concat([b for _, b in bands]).drop_duplicates(subset=group_key)
            fig, ax = plt.subplots(figsize=(7, max(4, len(sel) * 0.3)))
            y = np.arange(len(sel))
            xerr = sel[err_col].to_numpy() if err_col else None
            ax.barh(y, sel[value_col].to_numpy(), xerr=xerr, color=color, capsize=2)
            ax.set_yticks(y)
            ax.set_yticklabels(sel[group_key].astype(str))
            ax.set_ylim(-0.5, len(sel) - 0.5)
            if len(bands) > 1:
                offset = 0
                for name, b in bands:
                    n = len(b)
                    center = offset + n / 2 - 0.5
                    ax.text(1.005, center, name, transform=ax.get_yaxis_transform(),
                            rotation=90, va="center", ha="left", fontsize=_fs(8), color="gray")
                    if offset > 0:  # divider between bands
                        ax.axhline(offset - 0.5, color="gray", linestyle="--", linewidth=0.6)
                    offset += n
            ax.set_xlabel(xlabel)
            fig.tight_layout()
            fig.savefig(output_dir / out_name, dpi=150)
            plt.close(fig)

        _banded_barh(real, "mean", "std", "mean realness score",
                     f"per_font_real_score_{split}.png", REAL_COLOR)

        if auroc is not None and not auroc.empty:
            a = auroc[auroc["split"] == split]
            if not a.empty:
                # lowest AUROC = font the scorer confuses most with the gen pool.
                _banded_barh(a, "auroc", None, "real-vs-gen AUROC (real font vs pooled gen)",
                             f"per_font_auroc_{split}.png", "#607D8B")
        return

    raise ValueError(f"Unknown group_key: {group_key!r} (char | font_family)")


# --------------------------------------------------------------------------- #
# Score / logit distributions
# --------------------------------------------------------------------------- #
def plot_score_distributions(
    df_meta: pd.DataFrame,
    output_dir: Path,
    split: str | None = None,
    bins: int = 40,
) -> None:
    """Overlaid real/gen histograms of the realness score (and logit if present).

    Args:
        df_meta:  metadata DataFrame with columns label, score (and optionally logit, split).
        output_dir: where to write the PNGs.
        split:    restrict to one split; None uses all rows. Filenames are suffixed
                  with the split when given.
        bins:     histogram bin count.

    Writes score_dist[_<split>].png and, if a "logit" column exists, logit_dist[_<split>].png.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    df = df_meta if split is None else df_meta[df_meta["split"] == split]
    suffix = f"_{split}" if split is not None else ""
    cols = [c for c in ("score", "logit") if c in df.columns]
    for col in cols:
        fig, ax = plt.subplots(figsize=(7, 5))
        rng = (float(df[col].min()), float(df[col].max()))
        for lab, name, color in [(1, "real", REAL_COLOR), (0, "gen", GEN_COLOR)]:
            vals = df[df["label"] == lab][col].to_numpy()
            if len(vals) == 0:
                continue
            ax.hist(vals, bins=bins, range=rng, color=color, alpha=0.55,
                    label=f"{name} (n={len(vals)})", density=True)
        if col == "score":
            ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel(col)
        ax.set_ylabel("density")
        _legend(ax)
        fig.tight_layout()
        fig.savefig(output_dir / f"{col}_dist{suffix}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


# --------------------------------------------------------------------------- #
# Score KDE: real/gen densities, means, tails and the confusion (overlap) zone
# --------------------------------------------------------------------------- #
_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz  # renamed in NumPy 2.0


def _kde_eval(values: np.ndarray, grid: np.ndarray, bw: float | None = None) -> np.ndarray | None:
    """Gaussian-KDE `values` evaluated on `grid`, or None if KDE is undefined.

    Returns None when there are fewer than 5 samples or zero variance (gaussian_kde
    needs a non-singular covariance), so callers can skip gracefully.
    """
    v = np.asarray(values, dtype=float)
    if len(v) < 5 or float(np.std(v)) == 0.0:
        return None
    return gaussian_kde(v, bw_method=bw)(grid)


def _score_kde(
    df: pd.DataFrame, col: str = "score", n_grid: int = 512, bw: float | None = None
) -> dict:
    """Per-class KDE of `col` plus real/gen separation stats.

    Returns a dict with always-present scalars (n_real, n_gen, real_mean, gen_mean,
    real_p5, gen_p95) and the KDE-dependent fields (grid, d_real, d_gen, s_star, ovl).
    The KDE fields are None / NaN when either class has <5 samples or zero variance.
    s_star = argmax_s min(p_real, p_gen) (the confusion point); ovl = ∫min (overlap
    coefficient, a 0..1 separability summary).
    """
    real = df[df["label"] == 1][col].to_numpy(dtype=float)
    gen = df[df["label"] == 0][col].to_numpy(dtype=float)
    out: dict = {
        "n_real": int(len(real)), "n_gen": int(len(gen)),
        "real_mean": float(real.mean()) if len(real) else float("nan"),
        "gen_mean": float(gen.mean()) if len(gen) else float("nan"),
        "real_p5": float(np.percentile(real, 5)) if len(real) else float("nan"),
        "gen_p95": float(np.percentile(gen, 95)) if len(gen) else float("nan"),
        "grid": None, "d_real": None, "d_gen": None,
        "s_star": float("nan"), "ovl": float("nan"),
    }
    if len(real) == 0 or len(gen) == 0:
        return out
    lo = min(real.min(), gen.min())
    hi = max(real.max(), gen.max())
    pad = 0.02 * (hi - lo) if hi > lo else 0.01
    grid = np.linspace(lo - pad, hi + pad, n_grid)
    d_real = _kde_eval(real, grid, bw)
    d_gen = _kde_eval(gen, grid, bw)
    if d_real is None or d_gen is None:
        return out
    ov = np.minimum(d_real, d_gen)
    out.update(
        grid=grid, d_real=d_real, d_gen=d_gen,
        s_star=float(grid[int(np.argmax(ov))]),
        ovl=float(_trapz(ov, grid)),
    )
    return out


def score_separation_stats(
    df_meta: pd.DataFrame, split: str | None = None, col: str = "score",
    bw: float | None = None,
) -> dict:
    """Scalar real/gen separation stats for one split (rows of score_separation.csv).

    Keys: real_mean, gen_mean, s_star, ovl, n_real, n_gen. s_star/ovl are NaN when
    KDE is undefined (a class with <5 samples or zero variance).
    """
    df = df_meta if split is None else df_meta[df_meta["split"] == split]
    k = _score_kde(df, col=col, bw=bw)
    return {
        "real_mean": k["real_mean"], "gen_mean": k["gen_mean"],
        "s_star": k["s_star"], "ovl": k["ovl"],
        "n_real": k["n_real"], "n_gen": k["n_gen"],
    }


def plot_score_kde(
    df_meta: pd.DataFrame, output_dir: Path, split: str | None = None,
    n_grid: int = 512, bw: float | None = None,
) -> None:
    """Overlaid real/gen KDE of the score (and logit if present), annotated.

    Marks each class mean, the outer tails (real p5 / gen p95), and the confusion
    zone: the min(p_real, p_gen) curve, its peak s* and a shaded band where it is at
    least half its peak. The overlap coefficient OVL=∫min is shown in the title.
    Writes score_kde[_<split>].png and logit_kde[_<split>].png. A column is skipped
    (with a printed warning) when either class has <5 samples or zero variance.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    df = df_meta if split is None else df_meta[df_meta["split"] == split]
    suffix = f"_{split}" if split is not None else ""
    title_split = split if split is not None else "all splits"
    cols = [c for c in ("score", "logit") if c in df.columns]
    for col in cols:
        k = _score_kde(df, col=col, n_grid=n_grid, bw=bw)
        if k["grid"] is None:
            print(f"  KDE skipped for {col} ({title_split}): too few samples / zero variance")
            continue
        grid, d_real, d_gen = k["grid"], k["d_real"], k["d_gen"]
        ov = np.minimum(d_real, d_gen)
        fig, ax = plt.subplots(figsize=(7.5, 5))
        ax.plot(grid, d_real, color=REAL_COLOR, label=f"real (mu={k['real_mean']:.2f})")
        ax.fill_between(grid, d_real, color=REAL_COLOR, alpha=0.2)
        ax.plot(grid, d_gen, color=GEN_COLOR, label=f"gen (mu={k['gen_mean']:.2f})")
        ax.fill_between(grid, d_gen, color=GEN_COLOR, alpha=0.2)
        ax.axvline(k["real_mean"], color=REAL_COLOR, linewidth=1.0, alpha=0.7)
        ax.axvline(k["gen_mean"], color=GEN_COLOR, linewidth=1.0, alpha=0.7)
        ax.axvline(k["real_p5"], color=REAL_COLOR, linestyle=":", linewidth=1.0,
                   alpha=0.6, label=f"real tail p5={k['real_p5']:.2f}")
        ax.axvline(k["gen_p95"], color=GEN_COLOR, linestyle=":", linewidth=1.0,
                   alpha=0.6, label=f"gen tail p95={k['gen_p95']:.2f}")
        ax.plot(grid, ov, color="gray", linewidth=1.0, label="min(p_real,p_gen)")
        ax.fill_between(grid, ov, where=ov >= 0.5 * ov.max(), color="gray", alpha=0.35)
        ax.axvline(k["s_star"], color="black", linestyle="--", linewidth=1.2,
                   label=f"confusion s*={k['s_star']:.2f}")
        if col == "score":
            ax.axvline(0.5, color="lightgray", linestyle="--", linewidth=0.8, zorder=0)
        ax.set_xlabel(col)
        ax.set_ylabel("density")
        # OVL is not encoded in the filename, so keep it as a legend entry (no title).
        ax.plot([], [], " ", label=f"OVL={k['ovl']:.3f}")
        _legend(ax, fontsize=_fs(8))
        fig.tight_layout()
        fig.savefig(output_dir / f"{col}_kde{suffix}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


# --------------------------------------------------------------------------- #
# Image grids (model-input thumbnails via the image_path column)
# --------------------------------------------------------------------------- #
def _load_thumb(image_root: Path, rel_path: str | float, size: int | None = None) -> np.ndarray | None:
    """Load a saved input thumbnail (grayscale) as a float array in [0, 1], or None.

    Returns None when the path is missing/NaN or the file does not exist, so callers
    can degrade gracefully (e.g. old parquet without image_path).
    """
    if rel_path is None or (isinstance(rel_path, float) and np.isnan(rel_path)):
        return None
    path = image_root / str(rel_path)
    if not path.exists():
        return None
    img = Image.open(path).convert("L")
    if size is not None:
        img = img.resize((size, size))
    return np.asarray(img, dtype=np.float32) / 255.0


def has_images(df_meta: pd.DataFrame, image_root: Path) -> bool:
    """True if df_meta has an image_path column and at least one thumbnail exists."""
    if "image_path" not in df_meta.columns:
        return False
    for rel in df_meta["image_path"].head(20):
        if _load_thumb(image_root, rel) is not None:
            return True
    return False


def _image_grid(
    images: list[np.ndarray | None],
    titles: list[str],
    output_path: Path,
    ncols: int,
    suptitle: str = "",
    title_fontsize: float = 8.0,
    row_labels: list[str] | None = None,
) -> None:
    """Render thumbnails in a grid and save. Missing images (None) draw an empty cell.

    row_labels (one per row) annotate the left margin (e.g. "top"/"middle"/"bottom").
    """
    n = len(images)
    if n == 0:
        return
    nrows = (n + ncols - 1) // ncols
    # constrained_layout (not tight_layout) so the suptitle, per-cell titles and
    # per-row ylabels each get reserved space and never overlap each other.
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(ncols * 1.4, nrows * 1.6), squeeze=False,
        layout="constrained",
    )
    for k in range(nrows * ncols):
        ax = axes[k // ncols][k % ncols]
        ax.set_xticks([])
        ax.set_yticks([])
        if k < n and images[k] is not None:
            ax.imshow(images[k], cmap="gray", vmin=0.0, vmax=1.0)
            ax.set_title(titles[k], fontsize=_fs(title_fontsize))
        else:
            ax.axis("off")
            if k < n:
                ax.set_title(titles[k], fontsize=_fs(title_fontsize))
    if row_labels is not None:
        for r, lab in enumerate(row_labels):
            # turn the left-margin axis back on (off cells call axis("off")) so the
            # row label is always visible, while keeping the ticks hidden.
            axes[r][0].axis("on")
            axes[r][0].set_xticks([])
            axes[r][0].set_yticks([])
            axes[r][0].set_ylabel(lab, fontsize=_fs(9), rotation=90)
    if suptitle:
        fig.suptitle(suptitle, fontsize=_fs(11))
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _simple_grid(
    items: list[tuple[np.ndarray | None, str]],
    output_path: Path,
    grid_n: int,
    show_labels: bool = True,
    suptitle: str = "",
) -> None:
    """Render up to grid_n*grid_n thumbnails as a clean single-category N x N grid.

    Each item is (image, per-cell label). The label is drawn only when show_labels, so
    OFF yields a pure image grid (the selection is conveyed via the filename instead).
    Trailing empty cells are left blank. Delegates to _image_grid (row_labels=None).
    """
    items = items[: grid_n * grid_n]
    if not items:
        return
    images = [im for im, _ in items]
    titles = [lab if show_labels else "" for _, lab in items]
    _image_grid(images, titles, output_path, ncols=grid_n, suptitle=suptitle)


def plot_score_ranked_images(
    df_meta: pd.DataFrame,
    image_root: Path,
    output_dir: Path,
    split: str = "test",
    grid_n: int = 8,
    show_labels: bool = True,
) -> None:
    """For real and gen separately, write one N x N grid per score band.

    Writes ranked_{real,gen}_{top,middle,bottom}_<split>.png (each a single-category
    grid of up to grid_n*grid_n samples) so a single band can be inspected at scale.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    df = df_meta[df_meta["split"] == split]
    cap = grid_n * grid_n
    for lab, name in [(1, "real"), (0, "gen")]:
        sub = df[df["label"] == lab].sort_values("score", ascending=False).reset_index(drop=True)
        if len(sub) == 0:
            continue
        m = len(sub)
        mid_start = max(0, m // 2 - cap // 2)
        bands = {
            "top": sub.head(cap),
            "middle": sub.iloc[mid_start:mid_start + cap],
            "bottom": sub.tail(cap),
        }
        for band_name, band in bands.items():
            items = [(_load_thumb(image_root, r.image_path), f"{r.char} {r.score:.2f}")
                     for r in band.itertuples()]
            _simple_grid(
                items, output_dir / f"ranked_{name}_{band_name}_{split}.png",
                grid_n, show_labels,
            )


def plot_misclassified_images(
    df_meta: pd.DataFrame,
    image_root: Path,
    output_dir: Path,
    split: str = "test",
    grid_n: int = 8,
    threshold: float = 0.5,
    show_labels: bool = True,
) -> None:
    """Highest-confidence scorer errors as single-category N x N grids.

    Writes misclassified_real_<split>.png (real with lowest score = judged fake) and
    misclassified_gen_<split>.png (gen with highest score = judged real).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    df = df_meta[df_meta["split"] == split]
    cap = grid_n * grid_n
    bands = {
        "real": df[(df["label"] == 1) & (df["score"] < threshold)]
        .sort_values("score").head(cap),
        "gen": df[(df["label"] == 0) & (df["score"] >= threshold)]
        .sort_values("score", ascending=False).head(cap),
    }
    for name, band in bands.items():
        if band.empty:
            continue
        items = [(_load_thumb(image_root, r.image_path), f"{r.char} {r.score:.2f}")
                 for r in band.itertuples()]
        _simple_grid(
            items, output_dir / f"misclassified_{name}_{split}.png", grid_n, show_labels,
        )


def plot_group_representative_images(
    df_meta: pd.DataFrame,
    image_root: Path,
    output_dir: Path,
    group_key: str,
    split: str = "test",
    grid_n: int = 8,
    show_labels: bool = True,
) -> None:
    """Representative thumbnails per score band as single-category N x N grids.

    Writes repr_<group_key>_{lowest,middle,highest}_<split>.png. Each grid shows up to
    grid_n groups (one per row) with grid_n samples nearest each group's mean score, so
    typical/extreme groups can each be inspected at scale. For group_key="font_family":
    real fonts only (generated/guided are a single bucket), matching per_group_stats.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    df = df_meta[df_meta["split"] == split].copy()
    if group_key == "font_family":
        df = df[(df["label"] == 1) & (~df[group_key].isin(["generated", "guided"]))]
    if df.empty:
        return
    means = df.groupby(group_key)["score"].mean().sort_values()
    groups = means.index.tolist()
    n = grid_n
    mid_start = max(0, len(groups) // 2 - n // 2)
    bands = {
        "lowest": groups[:n],
        "middle": groups[mid_start:mid_start + n],
        "highest": groups[-n:],
    }
    for band_name, band_groups in bands.items():
        items: list[tuple[np.ndarray | None, str]] = []
        for grp in band_groups:
            sub = df[df[group_key] == grp]
            target = float(means[grp])
            sub = sub.assign(_d=(sub["score"] - target).abs()).sort_values("_d")
            recs = list(sub.head(n).itertuples())
            # pad each group to a full row so rows align one-per-group in the N x N grid
            for j in range(n):
                if j < len(recs):
                    r = recs[j]
                    items.append((_load_thumb(image_root, r.image_path), f"{grp} {r.score:.2f}"))
                else:
                    items.append((None, ""))
        _simple_grid(
            items, output_dir / f"repr_{group_key}_{band_name}_{split}.png", grid_n,
            show_labels,
        )


def plot_worst_auroc_char_examples(
    df_meta: pd.DataFrame,
    auroc: pd.DataFrame,
    image_root: Path,
    output_dir: Path,
    split: str = "test",
    grid_n: int = 8,
    show_labels: bool = True,
) -> None:
    """Lowest per-char AUROC chars, real and gen as separate single-category grids.

    Writes worst_auroc_real_<split>.png and worst_auroc_gen_<split>.png (each up to
    grid_n worst chars, one per row, with grid_n samples each). Compare the two files
    side by side for the real-vs-gen contrast.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    a = auroc[auroc["split"] == split].sort_values("auroc") if auroc is not None else None
    if a is None or a.empty:
        return
    df = df_meta[df_meta["split"] == split]
    chars = a["char"].head(grid_n).tolist()
    for lab, name in [(1, "real"), (0, "gen")]:
        items: list[tuple[np.ndarray | None, str]] = []
        for ch in chars:
            recs = list(df[(df["char"] == ch) & (df["label"] == lab)]
                        .sort_values("score").head(grid_n).itertuples())
            # pad each char to a full row so rows align one-per-char in the N x N grid
            for j in range(grid_n):
                if j < len(recs):
                    r = recs[j]
                    items.append((_load_thumb(image_root, r.image_path), f"{ch} {r.score:.2f}"))
                else:
                    items.append((None, ""))
        _simple_grid(
            items, output_dir / f"worst_auroc_{name}_{split}.png", grid_n, show_labels,
        )


# --------------------------------------------------------------------------- #
# Embedding with image thumbnails overlaid
# --------------------------------------------------------------------------- #
def plot_embedding_thumbnails(
    emb: np.ndarray,
    meta: pd.DataFrame,
    image_root: Path,
    output_dir: Path,
    reducer_name: str,
    max_thumbs: int = 200,
    grid: int = 24,
    zoom: float = 0.5,
) -> None:
    """Overlay model-input glyph thumbnails at their 2D embedding coordinates.

    `emb` (M, 2) and `meta` must be row-aligned (e.g. df_meta.iloc[idx]). To avoid
    clutter, the plane is binned into a grid x grid lattice and at most one thumbnail
    per occupied cell is drawn (capped at max_thumbs). Thumbnail border encodes
    real/gen. Writes embed_<reducer>_thumbnails.png.
    """
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage

    output_dir.mkdir(parents=True, exist_ok=True)
    meta = meta.reset_index(drop=True)
    if "image_path" not in meta.columns or len(emb) == 0:
        return

    # bin to a lattice and keep one representative per cell
    mins = emb.min(axis=0)
    span = np.where(emb.max(axis=0) - mins == 0, 1.0, emb.max(axis=0) - mins)
    cells = np.floor((emb - mins) / span * (grid - 1e-9)).astype(int)
    seen: set[tuple[int, int]] = set()
    keep: list[int] = []
    for i in range(len(emb)):
        key = (int(cells[i, 0]), int(cells[i, 1]))
        if key in seen:
            continue
        seen.add(key)
        keep.append(i)
        if len(keep) >= max_thumbs:
            break

    fig, ax = plt.subplots(figsize=(10, 9))
    ax.scatter(emb[:, 0], emb[:, 1], s=3, c="lightgray", alpha=0.4, linewidths=0)
    labels = meta["label"].to_numpy()
    for i in keep:
        thumb = _load_thumb(image_root, meta.iloc[i]["image_path"])
        if thumb is None:
            continue
        color = REAL_COLOR if labels[i] == 1 else GEN_COLOR
        ab = AnnotationBbox(
            OffsetImage(thumb, cmap="gray", zoom=zoom), (emb[i, 0], emb[i, 1]),
            frameon=True, pad=0.1, bboxprops=dict(edgecolor=color, linewidth=1.2),
        )
        ax.add_artist(ab)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    # No title (filename conveys it); keep the real/gen frame-colour key as a legend.
    ax.plot([], [], "s", color=REAL_COLOR, label="real")
    ax.plot([], [], "s", color=GEN_COLOR, label="gen")
    _legend(ax)
    fig.tight_layout()
    fig.savefig(output_dir / f"embed_{reducer_name}_thumbnails.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)


def plot_kde_region_images(
    df_meta: pd.DataFrame,
    image_root: Path,
    output_dir: Path,
    split: str = "test",
    grid_n: int = 8,
    bw: float | None = None,
    show_labels: bool = True,
) -> None:
    """Single-category N x N grids of samples from KDE-defined score regions.

    Writes kde_{real,gen}_{mean,tail}_<split>.png and, when KDE is defined,
    kde_{real,gen}_overlap_<split>.png (samples nearest the confusion point
    s* = argmax_s min(p_real, p_gen)). The two overlap files cover the same score band
    for real vs gen — the glyphs the scorer cannot tell apart. mean = nearest the class
    mean; tail = the outer tail (real low / gen high).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    df = df_meta[df_meta["split"] == split]
    real = df[df["label"] == 1]
    gen = df[df["label"] == 0]
    if real.empty or gen.empty:
        return
    k = _score_kde(df, col="score", bw=bw)

    regions: list[tuple[str, pd.DataFrame]] = [
        ("real_mean",
         real.assign(_d=(real["score"] - k["real_mean"]).abs()).sort_values("_d")),
        ("real_tail", real.sort_values("score")),
        ("gen_mean",
         gen.assign(_d=(gen["score"] - k["gen_mean"]).abs()).sort_values("_d")),
        ("gen_tail", gen.sort_values("score", ascending=False)),
    ]
    if not np.isnan(k["s_star"]):
        s = k["s_star"]
        regions.append(("real_overlap",
                        real.assign(_d=(real["score"] - s).abs()).sort_values("_d")))
        regions.append(("gen_overlap",
                        gen.assign(_d=(gen["score"] - s).abs()).sort_values("_d")))

    cap = grid_n * grid_n
    for name, band in regions:
        items = [(_load_thumb(image_root, r.image_path), f"{r.char} {r.score:.2f}")
                 for r in band.head(cap).itertuples()]
        _simple_grid(
            items, output_dir / f"kde_{name}_{split}.png", grid_n, show_labels,
        )
