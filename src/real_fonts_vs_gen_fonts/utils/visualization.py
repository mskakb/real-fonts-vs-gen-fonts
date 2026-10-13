from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def plot_reduction(
    reduced: np.ndarray,
    labels: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    """2D 次元圧縮結果を real/gen の散布図として保存する。

    Args:
        reduced:     2D 座標配列 (N, 2)。
        labels:      各点のラベル (N,)。1=real, 0=generated。
        title:       図のタイトル。
        output_path: 保存先パス。
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    for label, color, name in [(1, "#2196F3", "Real"), (0, "#FF5722", "Generated")]:
        mask = labels == label
        ax.scatter(
            reduced[mask, 0], reduced[mask, 1],
            c=color, label=name, alpha=0.4, s=8, linewidths=0,
        )
    ax.set_title(title)
    ax.legend(markerscale=3)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def visualize_features(
    real_feats: torch.Tensor,
    gen_feats: torch.Tensor,
    output_dir: Path,
    tsne_max_samples: int = 2000,
    tsne_perplexity: float = 30.0,
) -> None:
    """PCA と t-SNE で real/gen 特徴量を可視化して output_dir に保存する。

    Args:
        real_feats:       実フォントの特徴量 (N_real, D)。
        gen_feats:        生成フォントの特徴量 (N_gen, D)。
        output_dir:       pca.png と tsne.png の保存先。
        tsne_max_samples: t-SNE のサブサンプリング上限。
        tsne_perplexity:  t-SNE の perplexity。
    """
    real_np = real_feats.numpy()
    gen_np = gen_feats.numpy()
    feats = np.concatenate([real_np, gen_np], axis=0)
    labels = np.concatenate([np.ones(len(real_np)), np.zeros(len(gen_np))])

    print("Running PCA...")
    pca = PCA(n_components=2, random_state=42)
    reduced_pca = pca.fit_transform(feats)
    var_ratio = pca.explained_variance_ratio_
    plot_reduction(
        reduced_pca, labels,
        title=f"PCA  (var: PC1={var_ratio[0]:.1%}, PC2={var_ratio[1]:.1%})",
        output_path=output_dir / "pca.png",
    )

    n_total = len(feats)
    if n_total > tsne_max_samples:
        print(f"Subsampling {tsne_max_samples} / {n_total} for t-SNE...")
        rng = np.random.default_rng(42)
        idx = rng.choice(n_total, tsne_max_samples, replace=False)
        feats_tsne, labels_tsne = feats[idx], labels[idx]
    else:
        feats_tsne, labels_tsne = feats, labels

    print(f"Running t-SNE (n={len(feats_tsne)}, perplexity={tsne_perplexity})...")
    tsne = TSNE(n_components=2, perplexity=tsne_perplexity, random_state=42, n_jobs=-1)
    reduced_tsne = tsne.fit_transform(feats_tsne)
    plot_reduction(
        reduced_tsne, labels_tsne,
        title=f"t-SNE  (perplexity={tsne_perplexity})",
        output_path=output_dir / "tsne.png",
    )


def make_grid(
    images: list[torch.Tensor],
    overlays: list[np.ndarray],
    labels: list[str],
    title: str,
    output_path: Path,
    cols: int = 5,
) -> None:
    """画像グリッドを保存する。

    Args:
        images:   元画像のリスト (C, H, W) float [0, 1]。
        overlays: 表示する numpy 配列 (H, W, 3) のリスト（オーバーレイ済み画像）。
        labels:   各画像のサブタイトル文字列リスト。
        title:    図全体のタイトル。
        output_path: 保存先パス。
        cols:     グリッドの列数。
    """
    n = len(images)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2.3))
    axes = np.array(axes).reshape(rows, cols)

    for idx, (overlay, label) in enumerate(zip(overlays, labels)):
        r, c = divmod(idx, cols)
        axes[r, c].imshow(overlay)
        axes[r, c].set_title(label, fontsize=7)
        axes[r, c].axis("off")

    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].axis("off")

    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")
