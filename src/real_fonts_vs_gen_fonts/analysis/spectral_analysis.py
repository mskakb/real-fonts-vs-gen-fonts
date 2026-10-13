"""Pure functions for frequency-domain (2D FFT) analysis of Scorer input glyphs.

Consumes the same parquet contract as feature_analysis.py (extract_scorer_features.py
output): per-image metadata (split/label/char/font_family/score/logit/image_path) plus
the saved model-input thumbnails (inputs/<split>/*.png). No model / GPU / checkpoint is
required — this module reads the thumbnails directly and computes radial/angular power
spectra and summary scalars per image, then correlates them with the realness score.

Interpretation caveat (see CLAUDE.md "Scorer の周波数解析"): the thumbnails are the
model's *decoded* input (e.g. sdf_decode="hard" -> binarized glyph at both real and gen),
not the raw rendered / SDF files on disk. This analysis therefore measures residual
geometric differences after binarization (stroke width, curvature, aliasing at the
threshold), not the raw rendering-pipeline antialiasing asymmetry between real and gen.

Ink coverage (ink_fraction) is the primary confound: a stroke-width difference alone
shifts the whole spectrum. Every analysis here reports an ink_fraction-only baseline
(probe / AUROC) alongside the spectral ones so a spectral effect can be judged relative
to that baseline rather than in isolation.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from real_fonts_vs_gen_fonts.analysis.feature_analysis import (
    GEN_COLOR,
    REAL_COLOR,
    _fs,
    _legend,
    _load_thumb,
    _simple_grid,
    fit_linear_probe,
    layer_probe_auroc,
)

# 96px input -> Nyquist radius 48; matches the shipped Scorer's image_size.
RADIAL_BINS = 48
ANGULAR_BINS = 18  # 10-degree sectors over [0, pi) (power spectrum is symmetric)

SCALAR_COLS = [
    "ink_fraction", "band_low", "band_mid", "band_high", "hf_ratio",
    "centroid", "slope", "aniso_index", "hv_ratio",
]
RADIAL_COLS = [f"r{i:03d}" for i in range(RADIAL_BINS)]
ANGULAR_COLS = [f"a{i:02d}" for i in range(ANGULAR_BINS)]

FEATURE_SETS: dict[str, list[str]] = {
    "ink_only": ["ink_fraction"],
    "radial": RADIAL_COLS,
    "angular": ANGULAR_COLS,
    "scalars": SCALAR_COLS,
    "all": RADIAL_COLS + ANGULAR_COLS + SCALAR_COLS,
}


# --------------------------------------------------------------------------- #
# Per-image spectrum
# --------------------------------------------------------------------------- #
def image_power_spectrum(img: np.ndarray, window: str = "hann") -> np.ndarray:
    """(H, W) float image in [0, 1] -> fftshifted power spectrum |F|^2, shape (H, W).

    The image mean (DC term) is removed before transforming: DC is orders of magnitude
    larger than any AC bin and would otherwise dominate every radial/angular profile.
    A Hann window is applied after mean-removal to suppress the border discontinuity a
    flat/glyph image would otherwise create (which leaks into a cross/plus artifact in
    the raw FFT of a non-periodic signal).
    """
    x = img.astype(np.float64) - img.mean()
    if window == "hann":
        h = np.hanning(x.shape[0])
        w = np.hanning(x.shape[1])
        x = x * np.outer(h, w)
    elif window not in (None, "none"):
        raise ValueError(f"Unknown window: {window!r} (hann | none)")
    F = np.fft.fftshift(np.fft.fft2(x))
    return np.abs(F) ** 2


def _radial_grid(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    """Integer radius map (H, W) and angle map in [0, pi) (H, W), centered at (h//2, w//2)."""
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    dy, dx = y - cy, x - cx
    r = np.sqrt(dx.astype(np.float64) ** 2 + dy.astype(np.float64) ** 2)
    theta = np.mod(np.arctan2(dy, dx), np.pi)  # power spectrum is symmetric under theta+pi
    return r, theta


def radial_profile(power: np.ndarray, n_bins: int = RADIAL_BINS) -> np.ndarray:
    """Mean power per integer-radius ring, bins 0..n_bins-1 (bin 0 excludes the DC pixel)."""
    h, w = power.shape
    r, _ = _radial_grid(h, w)
    r_int = r.round().astype(int)
    profile = np.zeros(n_bins, dtype=np.float64)
    for k in range(n_bins):
        mask = r_int == k
        if k == 0:
            # exclude the exact-center DC pixel itself (already mean-removed, but its
            # bin would otherwise be a near-zero outlier that distorts log-log fits)
            cy, cx = h // 2, w // 2
            mask = mask.copy()
            mask[cy, cx] = False
        if mask.any():
            profile[k] = power[mask].mean()
    return profile


def angular_profile(power: np.ndarray, n_bins: int = ANGULAR_BINS, r_min: float = 2.0) -> np.ndarray:
    """Mean power per angular sector over [0, pi), excluding a small r < r_min disk (DC-dominated)."""
    h, w = power.shape
    r, theta = _radial_grid(h, w)
    sector = np.clip((theta / np.pi * n_bins).astype(int), 0, n_bins - 1)
    valid = r >= r_min
    profile = np.zeros(n_bins, dtype=np.float64)
    for k in range(n_bins):
        mask = valid & (sector == k)
        if mask.any():
            profile[k] = power[mask].mean()
    return profile


def spectral_features(img: np.ndarray, window: str = "hann", normalize: str = "per-image") -> dict:
    """Compute the radial/angular profiles + summary scalars for one grayscale image.

    Args:
        img:       (H, W) float array in [0, 1], black-ink-on-white-background convention.
        window:    passed to image_power_spectrum.
        normalize: "per-image" divides the radial/angular profiles by their own sum
                   before returning (so heavier-ink images don't dominate an average of
                   many images purely by having more total power); "none" returns raw
                   mean power per bin.
    Returns:
        dict with keys "radial" (n_bins,), "angular" (n_bins,), and the SCALAR_COLS
        scalars (ink_fraction, band_low/mid/high, hf_ratio, centroid, slope,
        aniso_index, hv_ratio).
    """
    power = image_power_spectrum(img, window=window)
    radial = radial_profile(power)
    angular = angular_profile(power)

    ink_fraction = float((1.0 - img).mean())  # black ink on white background

    if normalize == "per-image":
        r_total = radial.sum()
        radial_n = radial / r_total if r_total > 0 else radial
        a_total = angular.sum()
        angular_n = angular / a_total if a_total > 0 else angular
    elif normalize == "none":
        radial_n, angular_n = radial, angular
    else:
        raise ValueError(f"Unknown normalize: {normalize!r} (per-image | none)")

    k = np.arange(RADIAL_BINS)
    low_mask, mid_mask, high_mask = k < 8, (k >= 8) & (k < 20), k >= 20
    total = radial_n.sum()
    band_low = float(radial_n[low_mask].sum() / total) if total > 0 else float("nan")
    band_mid = float(radial_n[mid_mask].sum() / total) if total > 0 else float("nan")
    band_high = float(radial_n[high_mask].sum() / total) if total > 0 else float("nan")
    hf_ratio = float(band_high / (band_low + band_mid)) if (band_low + band_mid) > 0 else float("nan")

    centroid = float((k * radial_n).sum() / total) if total > 0 else float("nan")

    fit_mask = (k >= 2) & (k <= 40)
    kk, pp = k[fit_mask], radial[fit_mask]
    valid = pp > 0
    if valid.sum() >= 2:
        slope = float(np.polyfit(np.log(kk[valid]), np.log(pp[valid]), 1)[0])
    else:
        slope = float("nan")

    a_total_raw = angular.sum()
    if a_total_raw > 0:
        aniso_index = float((angular.max() - angular.min()) / angular.mean())
        # axis-aligned sectors (near 0deg and 90deg, i.e. bins near 0, n/2, n-1) vs
        # diagonal sectors (near 45deg/135deg, bins near n/4 and 3n/4)
        n = ANGULAR_BINS
        axis_idx = [0, n // 2, n - 1]
        diag_idx = [n // 4, (3 * n) // 4]
        axis_e = float(np.mean(angular[axis_idx]))
        diag_e = float(np.mean(angular[diag_idx]))
        hv_ratio = float(axis_e / diag_e) if diag_e > 0 else float("nan")
    else:
        aniso_index = float("nan")
        hv_ratio = float("nan")

    return {
        "radial": radial_n, "angular": angular_n,
        "ink_fraction": ink_fraction,
        "band_low": band_low, "band_mid": band_mid, "band_high": band_high,
        "hf_ratio": hf_ratio, "centroid": centroid, "slope": slope,
        "aniso_index": aniso_index, "hv_ratio": hv_ratio,
    }


# --------------------------------------------------------------------------- #
# Table construction (extract-once contract)
# --------------------------------------------------------------------------- #
def compute_spectral_table(
    df_meta: pd.DataFrame,
    image_root: Path,
    window: str = "hann",
    normalize: str = "per-image",
) -> pd.DataFrame:
    """One row per image: metadata columns + radial (r000..) + angular (a00..) + scalars.

    Rows whose thumbnail is missing (has_images()==False for that row) are dropped
    with a printed count.
    """
    meta_cols = [c for c in df_meta.columns if c != "image_path"]
    rows: list[dict] = []
    n_missing = 0
    for r in df_meta.itertuples(index=False):
        rel = getattr(r, "image_path", None)
        img = _load_thumb(image_root, rel)
        if img is None:
            n_missing += 1
            continue
        feat = spectral_features(img, window=window, normalize=normalize)
        row = {c: getattr(r, c) for c in meta_cols}
        row["image_path"] = rel
        for i, v in enumerate(feat["radial"]):
            row[f"r{i:03d}"] = float(v)
        for i, v in enumerate(feat["angular"]):
            row[f"a{i:02d}"] = float(v)
        for c in SCALAR_COLS:
            row[c] = feat[c]
        rows.append(row)
    if n_missing:
        print(f"  compute_spectral_table: skipped {n_missing} rows with missing thumbnails")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Radial / angular profile plots
# --------------------------------------------------------------------------- #
def plot_radial_profiles(df: pd.DataFrame, output_dir: Path, split: str | None = None) -> None:
    """Mean radial profile (real vs gen, log-y) on top; gen/real ratio (log-y) below.

    The ratio panel is the point: mean-profile overlays at this sample size look
    identical by eye even when a real difference exists, but the ratio makes small,
    consistent deviations visible.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    d = df if split is None else df[df["split"] == split]
    suffix = f"_{split}" if split is not None else ""
    k = np.arange(RADIAL_BINS)

    real = d[d["label"] == 1][RADIAL_COLS].to_numpy()
    gen = d[d["label"] == 0][RADIAL_COLS].to_numpy()
    if len(real) == 0 or len(gen) == 0:
        return
    real_mean, gen_mean = real.mean(axis=0), gen.mean(axis=0)
    real_sem = real.std(axis=0) / np.sqrt(len(real))
    gen_sem = gen.std(axis=0) / np.sqrt(len(gen))

    fig, axes = plt.subplots(2, 1, figsize=(7, 7), sharex=True, height_ratios=[2, 1])
    ax = axes[0]
    ax.plot(k, real_mean, color=REAL_COLOR, label=f"real (n={len(real)})")
    ax.fill_between(k, real_mean - 1.96 * real_sem, real_mean + 1.96 * real_sem,
                     color=REAL_COLOR, alpha=0.25)
    ax.plot(k, gen_mean, color=GEN_COLOR, label=f"gen (n={len(gen)})")
    ax.fill_between(k, gen_mean - 1.96 * gen_sem, gen_mean + 1.96 * gen_sem,
                     color=GEN_COLOR, alpha=0.25)
    ax.set_yscale("log")
    ax.set_ylabel("mean normalized power")
    _legend(ax, fontsize=_fs(8))

    ax2 = axes[1]
    ratio = np.where(real_mean > 0, gen_mean / np.clip(real_mean, 1e-12, None), np.nan)
    ax2.plot(k, ratio, color="#607D8B")
    ax2.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ax2.set_yscale("log")
    ax2.set_xlabel("radial frequency bin (k, px^-1)")
    ax2.set_ylabel("gen / real")
    fig.tight_layout()
    fig.savefig(output_dir / f"radial_profile{suffix}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_radial_profiles_by_char(df: pd.DataFrame, output_dir: Path, split: str = "test") -> None:
    """Small-multiples grid of the mean radial profile (real vs gen) per character."""
    output_dir.mkdir(parents=True, exist_ok=True)
    d = df[df["split"] == split]
    chars = sorted(d["char"].unique())
    if not chars:
        return
    ncols = 6
    nrows = (len(chars) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.2, nrows * 1.8),
                              sharex=True, squeeze=False)
    k = np.arange(RADIAL_BINS)
    for i, ch in enumerate(chars):
        ax = axes[i // ncols][i % ncols]
        sub = d[d["char"] == ch]
        real = sub[sub["label"] == 1][RADIAL_COLS].to_numpy()
        gen = sub[sub["label"] == 0][RADIAL_COLS].to_numpy()
        if len(real):
            ax.plot(k, real.mean(axis=0), color=REAL_COLOR, linewidth=1.0)
        if len(gen):
            ax.plot(k, gen.mean(axis=0), color=GEN_COLOR, linewidth=1.0)
        ax.set_yscale("log")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel(str(ch), fontsize=_fs(8), rotation=0, labelpad=10)
    for i in range(len(chars), nrows * ncols):
        axes[i // ncols][i % ncols].axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / f"radial_profile_by_char_{split}.png", dpi=150)
    plt.close(fig)


def plot_angular_profiles(df: pd.DataFrame, output_dir: Path, split: str | None = None) -> None:
    """Polar mean angular profile, real vs gen (anisotropy: stroke direction bias)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    d = df if split is None else df[df["split"] == split]
    suffix = f"_{split}" if split is not None else ""
    real = d[d["label"] == 1][ANGULAR_COLS].to_numpy()
    gen = d[d["label"] == 0][ANGULAR_COLS].to_numpy()
    if len(real) == 0 or len(gen) == 0:
        return
    theta = (np.arange(ANGULAR_BINS) + 0.5) / ANGULAR_BINS * np.pi
    theta_full = np.concatenate([theta, theta + np.pi])  # mirror [0,pi) -> [0,2pi) for display

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "polar"})
    for vals, color, name in [(real, REAL_COLOR, "real"), (gen, GEN_COLOR, "gen")]:
        m = vals.mean(axis=0)
        m_full = np.concatenate([m, m])
        ax.plot(theta_full, m_full, color=color, label=name)
        ax.fill(theta_full, m_full, color=color, alpha=0.15)
    _legend(ax, fontsize=_fs(8))
    fig.savefig(output_dir / f"angular_profile{suffix}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_profile_by_score_bin(df: pd.DataFrame, output_dir: Path, split: str = "test",
                               n_bins: int = 4) -> None:
    """Mean radial profile split into within-class score quartiles.

    Quartiles are cut *within* each class (real, gen separately) so the comparison is
    "does the spectrum shift with score inside one class" rather than accidentally
    re-deriving the real/gen label split (which trivially correlates with score).
    Writes profile_by_score_bin_{real,gen}_<split>.png.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    d = df[df["split"] == split]
    k = np.arange(RADIAL_BINS)
    for lab, name, cmap_name in [(1, "real", "Blues"), (0, "gen", "Oranges")]:
        sub = d[d["label"] == lab].copy()
        if len(sub) < n_bins * 5:
            continue
        sub["_bin"] = pd.qcut(sub["score"], n_bins, labels=False, duplicates="drop")
        fig, ax = plt.subplots(figsize=(7, 5))
        cmap = plt.get_cmap(cmap_name)
        for b in sorted(sub["_bin"].dropna().unique()):
            g = sub[sub["_bin"] == b]
            mean_score = g["score"].mean()
            color = cmap(0.3 + 0.6 * (b + 1) / n_bins)
            ax.plot(k, g[RADIAL_COLS].to_numpy().mean(axis=0), color=color,
                    label=f"q{int(b)} (score~{mean_score:.2f}, n={len(g)})")
        ax.set_yscale("log")
        ax.set_xlabel("radial frequency bin (k, px^-1)")
        ax.set_ylabel("mean normalized power")
        _legend(ax, fontsize=_fs(8))
        fig.tight_layout()
        fig.savefig(output_dir / f"profile_by_score_bin_{name}_{split}.png", dpi=150,
                    bbox_inches="tight")
        plt.close(fig)


# --------------------------------------------------------------------------- #
# Scalar-metric distributions and separation stats
# --------------------------------------------------------------------------- #
def plot_metric_distributions(df: pd.DataFrame, output_dir: Path, split: str | None = None,
                               bins: int = 40) -> None:
    """Overlaid real/gen histogram per scalar metric (ink_fraction, hf_ratio, ...)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    d = df if split is None else df[df["split"] == split]
    suffix = f"_{split}" if split is not None else ""
    for col in SCALAR_COLS:
        vals_all = d[col].replace([np.inf, -np.inf], np.nan).dropna()
        if vals_all.empty:
            continue
        rng = (float(vals_all.min()), float(vals_all.max()))
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        for lab, name, color in [(1, "real", REAL_COLOR), (0, "gen", GEN_COLOR)]:
            vals = d[d["label"] == lab][col].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
            if len(vals) == 0:
                continue
            ax.hist(vals, bins=bins, range=rng, color=color, alpha=0.55,
                    label=f"{name} (n={len(vals)})", density=True)
        ax.set_xlabel(col)
        ax.set_ylabel("density")
        _legend(ax)
        fig.tight_layout()
        fig.savefig(output_dir / f"{col}_dist{suffix}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled_std = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                          / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / pooled_std) if pooled_std > 0 else float("nan")


def metric_separation_stats(df: pd.DataFrame, split: str | None = None) -> pd.DataFrame:
    """Per-scalar-metric real/gen separation: mean(real), mean(gen), Cohen's d, single-metric AUROC.

    Judge "does frequency differ" via this table, not p-values: at test-split sample
    sizes (tens of thousands) any nonzero difference is statistically significant, so
    effect size (d) and AUROC are the only informative numbers.
    """
    from sklearn.metrics import roc_auc_score

    d = df if split is None else df[df["split"] == split]
    y = d["label"].to_numpy()
    rows = []
    for col in SCALAR_COLS:
        vals = d[col].to_numpy(dtype=float)
        finite = np.isfinite(vals)
        real = vals[finite & (y == 1)]
        gen = vals[finite & (y == 0)]
        if len(real) < 2 or len(gen) < 2:
            continue
        auroc = float(roc_auc_score(y[finite], vals[finite])) if len(np.unique(y[finite])) > 1 else float("nan")
        rows.append({
            "metric": col, "real_mean": float(real.mean()), "gen_mean": float(gen.mean()),
            "cohens_d": _cohens_d(real, gen), "auroc": auroc,
            "n_real": int(len(real)), "n_gen": int(len(gen)),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Linear probes on spectral feature sets (vs the ink-only baseline)
# --------------------------------------------------------------------------- #
def spectral_probe(
    df: pd.DataFrame, fit_split: str = "train", eval_splits: list[str] | None = None,
) -> dict[str, dict[str, tuple]]:
    """Fit a linear probe per FEATURE_SETS entry on `fit_split`; evaluate on each of eval_splits.

    Returns {feature_set_name: {split_name: (fpr, tpr, auroc)}}. Compare "radial" (or
    "all") against "ink_only": if they're close, an apparent spectral signal is just an
    ink-coverage difference in disguise.
    """
    eval_splits = eval_splits or ["train", "val", "test"]
    fit_df = df[df["split"] == fit_split]
    out: dict[str, dict[str, tuple]] = {}
    for name, cols in FEATURE_SETS.items():
        X_fit = fit_df[cols].to_numpy(dtype=np.float64)
        y_fit = fit_df["label"].to_numpy()
        finite = np.isfinite(X_fit).all(axis=1)
        if finite.sum() < 10 or len(np.unique(y_fit[finite])) < 2:
            continue
        scaler, clf = fit_linear_probe(X_fit[finite], y_fit[finite])
        out[name] = {}
        for sp in eval_splits:
            sub = df[df["split"] == sp]
            X = sub[cols].to_numpy(dtype=np.float64)
            y = sub["label"].to_numpy()
            f = np.isfinite(X).all(axis=1)
            if f.sum() < 2 or len(np.unique(y[f])) < 2:
                continue
            out[name][sp] = layer_probe_auroc(scaler, clf, X[f], y[f])
    return out


def probe_results_table(probes: dict[str, dict[str, tuple]]) -> pd.DataFrame:
    """Flatten spectral_probe() output into a DataFrame [feature_set, split, auroc, n]."""
    rows = []
    for name, per_split in probes.items():
        for sp, (fpr, tpr, auroc) in per_split.items():
            rows.append({"feature_set": name, "split": sp, "auroc": auroc})
    return pd.DataFrame(rows)


def plot_spectral_roc(
    probes: dict[str, dict[str, tuple]], output_dir: Path, split: str,
    scorer_curve: tuple | None = None,
) -> None:
    """ROC of each spectral feature set on one split, with the Scorer's own ROC overlaid."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=0.8, label="chance")
    for name, per_split in probes.items():
        if split not in per_split:
            continue
        fpr, tpr, auroc = per_split[split]
        ax.plot(fpr, tpr, label=f"{name} (AUROC={auroc:.3f})")
    if scorer_curve is not None:
        fpr, tpr, auroc = scorer_curve
        ax.plot(fpr, tpr, color="black", linewidth=2, label=f"scorer (AUROC={auroc:.3f})")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    _legend(ax, fontsize=_fs(8))
    fig.tight_layout()
    fig.savefig(output_dir / f"roc_spectral_{split}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def per_char_spectral_auroc(df: pd.DataFrame, split: str = "test",
                             feature_set: str = "radial") -> pd.DataFrame:
    """Per-char real-vs-gen AUROC of a linear probe fit on that char's own split rows.

    Mirrors feature_analysis.per_char_auroc but for a spectral feature set fit
    per-char (small-sample probes; only chars with enough rows of both classes are
    included).
    """
    from sklearn.metrics import roc_auc_score

    cols = FEATURE_SETS[feature_set]
    d = df[df["split"] == split]
    rows = []
    for ch, grp in d.groupby("char"):
        y = grp["label"].to_numpy()
        if len(np.unique(y)) < 2 or len(grp) < 10:
            continue
        X = grp[cols].to_numpy(dtype=np.float64)
        finite = np.isfinite(X).all(axis=1)
        if finite.sum() < 10 or len(np.unique(y[finite])) < 2:
            continue
        scaler, clf = fit_linear_probe(X[finite], y[finite])
        proba = clf.predict_proba(scaler.transform(X[finite]))[:, 1]
        rows.append({
            "char": ch, "auroc": float(roc_auc_score(y[finite], proba)), "n": int(finite.sum()),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Score <-> frequency relationship
# --------------------------------------------------------------------------- #
def score_metric_correlations(df: pd.DataFrame, split: str = "test") -> pd.DataFrame:
    """Spearman rho of score vs each scalar metric: overall, per-label, and ink-controlled.

    "ink-controlled" is the partial correlation of score vs metric holding ink_fraction
    fixed (via residualizing both on ink_fraction with a linear fit, then correlating
    the residuals) — the check for "is this just ink coverage again".
    """
    d = df[df["split"] == split]
    rows = []
    for scope_name, scope_df in [("all", d), ("real", d[d["label"] == 1]), ("gen", d[d["label"] == 0])]:
        for col in SCALAR_COLS:
            x = scope_df[col].to_numpy(dtype=float)
            s = scope_df["score"].to_numpy(dtype=float)
            finite = np.isfinite(x) & np.isfinite(s)
            if finite.sum() < 10:
                continue
            xf, sf = x[finite], s[finite]
            rho, p = spearmanr(xf, sf)
            row = {"scope": scope_name, "metric": col, "spearman_rho": float(rho),
                   "p_value": float(p), "n": int(finite.sum())}
            if col != "ink_fraction":
                ink = scope_df["ink_fraction"].to_numpy(dtype=float)[finite]
                if np.isfinite(ink).all() and ink.std() > 0 and xf.std() > 0:
                    x_resid = xf - np.polyval(np.polyfit(ink, xf, 1), ink)
                    s_resid = sf - np.polyval(np.polyfit(ink, sf, 1), ink)
                    rho_p, _ = pearsonr(x_resid, s_resid)
                    row["partial_rho_ink_controlled"] = float(rho_p)
                else:
                    row["partial_rho_ink_controlled"] = float("nan")
            else:
                row["partial_rho_ink_controlled"] = float("nan")
            rows.append(row)
    return pd.DataFrame(rows)


def plot_score_vs_metric(df: pd.DataFrame, output_dir: Path, split: str = "test") -> None:
    """Scatter of scorer score vs each scalar metric, colored by class, Spearman rho in legend."""
    output_dir.mkdir(parents=True, exist_ok=True)
    d = df[df["split"] == split]
    for col in SCALAR_COLS:
        fig, ax = plt.subplots(figsize=(6, 5))
        for lab, name, color in [(1, "real", REAL_COLOR), (0, "gen", GEN_COLOR)]:
            sub = d[d["label"] == lab]
            x = sub[col].to_numpy(dtype=float)
            y = sub["score"].to_numpy(dtype=float)
            finite = np.isfinite(x) & np.isfinite(y)
            ax.scatter(x[finite], y[finite], s=4, alpha=0.15, color=color, linewidths=0, label=name)
        x_all = d[col].to_numpy(dtype=float)
        y_all = d["score"].to_numpy(dtype=float)
        finite = np.isfinite(x_all) & np.isfinite(y_all)
        if finite.sum() >= 10:
            rho, _ = spearmanr(x_all[finite], y_all[finite])
            ax.plot([], [], " ", label=f"Spearman rho={rho:.3f}")
        ax.set_xlabel(col)
        ax.set_ylabel("realness score")
        _legend(ax, fontsize=_fs(8), markerscale=3)
        fig.tight_layout()
        fig.savefig(output_dir / f"score_vs_{col}_{split}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def plot_score_vs_probe(
    df: pd.DataFrame, probe_proba: np.ndarray, output_dir: Path, split: str = "test",
    feature_set_name: str = "radial",
) -> None:
    """Scatter of scorer realness score vs spectral-probe p(real), colored by class."""
    output_dir.mkdir(parents=True, exist_ok=True)
    d = df[df["split"] == split].reset_index(drop=True)
    if len(d) != len(probe_proba):
        raise ValueError(f"row mismatch: df has {len(d)} rows, probe_proba has {len(probe_proba)}")
    fig, ax = plt.subplots(figsize=(6, 5))
    for lab, name, color in [(1, "real", REAL_COLOR), (0, "gen", GEN_COLOR)]:
        mask = (d["label"] == lab).to_numpy()
        ax.scatter(probe_proba[mask], d["score"].to_numpy()[mask], s=4, alpha=0.15,
                   color=color, linewidths=0, label=name)
    rho, _ = spearmanr(probe_proba, d["score"].to_numpy())
    ax.plot([], [], " ", label=f"Spearman rho={rho:.3f}")
    ax.set_xlabel(f"spectral probe p(real) [{feature_set_name}]")
    ax.set_ylabel("realness score")
    _legend(ax, fontsize=_fs(8), markerscale=3)
    fig.tight_layout()
    fig.savefig(output_dir / f"score_vs_spectral_probe_{split}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Image grids ranked by a spectral metric
# --------------------------------------------------------------------------- #
def plot_metric_ranked_images(
    df: pd.DataFrame, image_root: Path, output_dir: Path, metric: str = "hf_ratio",
    split: str = "test", grid_n: int = 8, show_labels: bool = True,
) -> None:
    """Real and gen images ranked by a scalar metric, top/bottom N x N grids.

    Writes ranked_{real,gen}_{top,bottom}_<metric>_<split>.png.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    d = df[df["split"] == split]
    cap = grid_n * grid_n
    for lab, name in [(1, "real"), (0, "gen")]:
        sub = d[d["label"] == lab].sort_values(metric, ascending=False).reset_index(drop=True)
        if sub.empty:
            continue
        bands = {"top": sub.head(cap), "bottom": sub.tail(cap)}
        for band_name, band in bands.items():
            items = [(_load_thumb(image_root, r.image_path), f"{r.char} {getattr(r, metric):.3f}")
                     for r in band.itertuples()]
            _simple_grid(
                items, output_dir / f"ranked_{name}_{band_name}_{metric}_{split}.png",
                grid_n, show_labels,
            )
