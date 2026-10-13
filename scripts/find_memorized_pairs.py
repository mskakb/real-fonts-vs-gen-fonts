"""生成画像が学習データを記憶（memorize）していないか検証する。

文字ごとに、各 GENERATED 画像に最も近い REAL 画像（同一文字、pixel MSE 最小）を探し、
MSE が特に小さいペアを記憶の疑いとして提示する。比較の物差しとして real→real の
最近傍 MSE（同一文字内、自分自身を除く）をベースライン分布として併記し、生成画像の
最近傍実フォントがどの split（train/val/test）由来かを注釈する。

MSE は全ペアで計算できるが、LPIPS（知覚距離）は各画像 1 回の forward が重いため、
MSE 最小の上位 top_k ペアのみに絞って計算する（shortlist）。

Usage:
    uv run python scripts/find_memorized_pairs.py --config config/eval/find_memorized_pairs.yaml

    uv run python scripts/find_memorized_pairs.py \\
        --config config/eval/find_memorized_pairs.yaml charset=H top_k=16
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from piq import LPIPS
from torch.utils.data import DataLoader

from real_fonts_vs_gen_fonts.datamodule import (
    FontDataset,
    _load_gen_meta,
    _load_real_meta,
    _load_split,
    _make_label_map,
    _resolve_codepoints,
)
from real_fonts_vs_gen_fonts.utils.config import load_config, save_run_info
from real_fonts_vs_gen_fonts.utils.image import make_image_transform
from real_fonts_vs_gen_fonts.utils.torch_utils import get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find nearest-real-image matches for generated fonts (memorization check)"
    )
    parser.add_argument("--config", required=True, metavar="FILE",
                        help="YAML config file (see config/eval/find_memorized_pairs.yaml)")
    parser.add_argument("overrides", nargs="*", metavar="key=value",
                        help="Config overrides, e.g. charset=H top_k=32")
    return parser.parse_args()


def _nearest_neighbor(
    query: torch.Tensor,
    ref: torch.Tensor,
    n_pixels: int,
    chunk_size: int,
    ref_block: int,
    exclude_self: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """query の各行に対し ref 中の最近傍（最小 MSE）の値とインデックスを返す。

    query/ref は同一次元のフラット化画像テンソル (N, D)。exclude_self=True の場合、
    query と ref は同一テンソル（同一インデックス=自分自身のペア）を想定し、対角成分を除外する
    （real→real ベースライン用）。メモリを抑えるため両軸をチャンク分割し、ブロックごとの
    cdist から実行時最小値を追跡する。
    """
    n_q = query.shape[0]
    n_r = ref.shape[0]
    best_val = torch.full((n_q,), float("inf"), device=query.device)
    best_idx = torch.zeros(n_q, dtype=torch.long, device=query.device)

    for qi in range(0, n_q, chunk_size):
        q_chunk = query[qi : qi + chunk_size]
        chunk_best_val = torch.full((q_chunk.shape[0],), float("inf"), device=query.device)
        chunk_best_idx = torch.zeros(q_chunk.shape[0], dtype=torch.long, device=query.device)

        for ri in range(0, n_r, ref_block):
            r_chunk = ref[ri : ri + ref_block]
            d = torch.cdist(q_chunk, r_chunk) ** 2 / n_pixels
            if exclude_self:
                q_global = torch.arange(qi, qi + q_chunk.shape[0], device=query.device)
                r_global = torch.arange(ri, ri + r_chunk.shape[0], device=query.device)
                mask = q_global.unsqueeze(1) == r_global.unsqueeze(0)
                d = d.masked_fill(mask, float("inf"))
            vals, idxs = d.min(dim=1)
            better = vals < chunk_best_val
            chunk_best_val = torch.where(better, vals, chunk_best_val)
            chunk_best_idx = torch.where(better, idxs + ri, chunk_best_idx)

        best_val[qi : qi + chunk_size] = chunk_best_val
        best_idx[qi : qi + chunk_size] = chunk_best_idx

    return best_val, best_idx


def _load_char_images(
    meta: pd.DataFrame,
    images_dir: Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> torch.Tensor:
    """meta（単一文字に絞った metadata）の画像を meta の行順で (N, H*W) テンソルにロードする。"""
    transform = make_image_transform(image_size)
    ds = FontDataset(meta, images_dir, transform, image_channels=1)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    chunks = []
    for batch in loader:
        chunks.append(batch["image"].flatten(1))
    return torch.cat(chunks, dim=0).to(device)


def _load_images_by_file(
    files: list[str], images_dir: Path, image_size: int
) -> dict[str, torch.Tensor]:
    """指定ファイル名のみをディスクから再ロードする（shortlist / top_k プロット用、全量キャッシュ回避）。"""
    transform = make_image_transform(image_size)
    result = {}
    for f in files:
        img = Image.open(images_dir / f).convert("L")
        result[f] = transform(img)
    return result


def _annotate_real_split(real_meta: pd.DataFrame, split_file: Path | None) -> pd.DataFrame:
    """real_meta に split 列（train/val/test、split_file が無ければ "all"）を付与する。"""
    real_meta = real_meta.copy()
    if split_file is None:
        real_meta["split"] = "all"
        return real_meta
    train_df, val_df, test_df = _load_split(split_file, real_meta)
    train_fams = set(train_df["font_family"])
    val_fams = set(val_df["font_family"])
    test_fams = set(test_df["font_family"])

    def _split_of(font_family: str) -> str:
        if font_family in train_fams:
            return "train"
        if font_family in val_fams:
            return "val"
        if font_family in test_fams:
            return "test"
        return "unknown"

    real_meta["split"] = real_meta["font_family"].map(_split_of)
    return real_meta


def _make_pair_thumbnail(gen_img: torch.Tensor, real_img: torch.Tensor) -> np.ndarray:
    """gen/real 画像 (1, H, W) を薄い区切り列付きで横に並べた (H, W*2+gap) の numpy 配列を返す。"""
    gen_np = gen_img.squeeze(0).cpu().numpy()
    real_np = real_img.squeeze(0).cpu().numpy()
    gap = np.ones((gen_np.shape[0], 2), dtype=gen_np.dtype)
    return np.concatenate([gen_np, gap, real_np], axis=1)


def _plot_top_matches(
    rows: pd.DataFrame,
    gen_images: dict[str, torch.Tensor],
    real_images: dict[str, torch.Tensor],
    output_path: Path,
    cols: int = 4,
) -> None:
    """MSE 昇順で並べた上位ペアを gen|real 並置サムネのグリッドで保存する。"""
    n = len(rows)
    grid_cols = cols
    grid_rows = (n + grid_cols - 1) // grid_cols
    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(grid_cols * 3, grid_rows * 2.6))
    axes = np.array(axes).reshape(grid_rows, grid_cols)

    for idx in range(grid_rows * grid_cols):
        r, c = divmod(idx, grid_cols)
        ax = axes[r, c]
        ax.axis("off")
        if idx >= n:
            continue
        row = rows.iloc[idx]
        thumb = _make_pair_thumbnail(gen_images[row["gen_file"]], real_images[row["nn_real_file"]])
        ax.imshow(thumb, cmap="gray", vmin=0, vmax=1)
        lpips_str = f"{row['lpips']:.3f}" if pd.notna(row["lpips"]) else "n/a"
        ax.set_title(
            f"{row['gen_char']}  mse={row['mse']:.4f}  lpips={lpips_str}\n"
            f"split={row['nn_real_split']}  font={row['nn_real_font_family']}",
            fontsize=6,
        )

    fig.suptitle("gen (left) | nearest real (right), sorted by MSE ascending", fontsize=9)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def _plot_distance_hist(
    gen_mse: np.ndarray,
    real_mse: np.ndarray,
    baseline_p1: float,
    output_path: Path,
) -> None:
    """gen→real と real→real（ベースライン）の最近傍 MSE 分布を重ねて保存する。"""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.linspace(0, max(gen_mse.max(), real_mse.max()), 80)
    ax.hist(real_mse, bins=bins, alpha=0.5, label=f"real→real baseline (n={len(real_mse)})",
            color="tab:blue", density=True)
    ax.hist(gen_mse, bins=bins, alpha=0.5, label=f"gen→real nearest (n={len(gen_mse)})",
            color="tab:orange", density=True)
    ax.axvline(baseline_p1, color="tab:blue", linestyle="--", linewidth=1,
               label=f"baseline p1={baseline_p1:.4f}")
    ax.set_xlabel("nearest-neighbor MSE")
    ax.set_ylabel("density")
    ax.legend(fontsize=8)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main() -> None:
    raw = parse_args()
    cfg = load_config(raw.config, raw.overrides)
    args = SimpleNamespace(
        real_dir=Path(cfg["real_dir"]),
        gen_dir=Path(cfg["gen_dir"]),
        output_dir=Path(cfg["output_dir"]),
        charset=cfg["charset"],
        image_size=cfg["image_size"],
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        split_file=Path(cfg["split_file"]) if cfg.get("split_file") else None,
        top_k=cfg["top_k"],
        real_block=cfg["real_block"],
    )

    device = get_device()
    print(f"Device: {device}")

    codepoints = _resolve_codepoints(args.charset)
    label_map = _make_label_map(codepoints)

    real_meta = _load_real_meta(args.real_dir, "diffusion", label_map)
    real_meta = _annotate_real_split(real_meta, args.split_file)
    gen_meta = _load_gen_meta(args.gen_dir, label_map)

    n_pixels = args.image_size * args.image_size

    gen_rows: list[dict] = []
    real_baseline_mse: list[float] = []
    real_baseline_split: list[str] = []

    for cp in codepoints:
        char = chr(cp)
        real_c = real_meta[real_meta["codepoint"] == cp].reset_index(drop=True)
        gen_c = gen_meta[gen_meta["codepoint"] == cp].reset_index(drop=True)
        if len(real_c) == 0 or len(gen_c) == 0:
            print(f"Skip char={char!r}: n_real={len(real_c)} n_gen={len(gen_c)}")
            continue

        real_imgs = _load_char_images(
            real_c, args.real_dir / "images", args.image_size, args.batch_size,
            args.num_workers, device,
        )
        gen_imgs = _load_char_images(
            gen_c, args.gen_dir / "images", args.image_size, args.batch_size,
            args.num_workers, device,
        )

        # gen → real 最近傍
        gen_min, gen_idx = _nearest_neighbor(
            gen_imgs, real_imgs, n_pixels, args.batch_size, args.real_block, exclude_self=False
        )
        gen_min = gen_min.cpu().numpy()
        gen_idx = gen_idx.cpu().numpy()
        for i in range(len(gen_c)):
            real_row = real_c.iloc[int(gen_idx[i])]
            gen_rows.append({
                "gen_char": char,
                "gen_file": gen_c.iloc[i]["file"],
                "nn_real_file": real_row["file"],
                "nn_real_font_family": real_row["font_family"],
                "nn_real_split": real_row["split"],
                "mse": float(gen_min[i]),
            })

        # real → real ベースライン（同一文字内、自分自身を除く最近傍）
        if len(real_c) >= 2:
            real_min, _ = _nearest_neighbor(
                real_imgs, real_imgs, n_pixels, args.real_block, args.real_block, exclude_self=True
            )
            real_baseline_mse.extend(real_min.cpu().numpy().tolist())
            real_baseline_split.extend(real_c["split"].tolist())

    df = pd.DataFrame(gen_rows)
    df["lpips"] = float("nan")

    # LPIPS はコストが高いので MSE 最小の上位 top_k のみ計算するショートリスト。
    # 画像は全量を常駐させず、shortlist に必要な分だけディスクから再ロードする。
    df = df.sort_values("mse", ascending=True).reset_index(drop=True)
    shortlist = df.head(args.top_k)

    gen_images_by_file = _load_images_by_file(
        shortlist["gen_file"].tolist(), args.gen_dir / "images", args.image_size
    )
    real_images_by_file = _load_images_by_file(
        shortlist["nn_real_file"].tolist(), args.real_dir / "images", args.image_size
    )

    lpips_net = LPIPS(reduction="none").to(device).eval()
    with torch.no_grad():
        gen_batch = torch.stack(
            [gen_images_by_file[f] for f in shortlist["gen_file"]]
        ).to(device).repeat(1, 3, 1, 1)
        real_batch = torch.stack(
            [real_images_by_file[f] for f in shortlist["nn_real_file"]]
        ).to(device).repeat(1, 3, 1, 1)
        lpips_vals = lpips_net(gen_batch, real_batch).cpu().numpy()
    df.loc[shortlist.index, "lpips"] = lpips_vals

    args.output_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output_dir / "nn_pairs.parquet")

    real_baseline_mse_arr = np.array(real_baseline_mse)
    gen_mse_arr = df["mse"].to_numpy()

    _plot_top_matches(
        df.head(args.top_k), gen_images_by_file, real_images_by_file,
        args.output_dir / "top_matches.png",
    )

    baseline_p1 = float(np.quantile(real_baseline_mse_arr, 0.01)) if len(real_baseline_mse_arr) else float("nan")
    _plot_distance_hist(gen_mse_arr, real_baseline_mse_arr, baseline_p1, args.output_dir / "nn_distance_hist.png")

    def _quantiles(x: np.ndarray) -> dict[str, float]:
        return {
            "min": float(np.min(x)),
            "p1": float(np.quantile(x, 0.01)),
            "p5": float(np.quantile(x, 0.05)),
            "median": float(np.median(x)),
        }

    split_counts = df["nn_real_split"].value_counts(normalize=True).to_dict()
    n_candidates = int((gen_mse_arr < baseline_p1).sum()) if len(real_baseline_mse_arr) else None

    summary = {
        "charset": args.charset,
        "n_real": len(real_meta),
        "n_gen": len(gen_meta),
        "n_gen_matched": len(df),
        "n_real_baseline_pairs": len(real_baseline_mse_arr),
        "gen_to_real_mse": _quantiles(gen_mse_arr),
        "real_to_real_baseline_mse": _quantiles(real_baseline_mse_arr) if len(real_baseline_mse_arr) else None,
        "nn_split_fraction": {k: float(v) for k, v in split_counts.items()},
        "n_memorization_candidates": n_candidates,
        "memorization_candidate_threshold_mse": baseline_p1,
    }
    with open(args.output_dir / "summary.yaml", "w") as f:
        yaml.dump(summary, f, allow_unicode=True, sort_keys=False)

    with open(args.output_dir / "config.yaml", "w") as f:
        yaml.dump({**cfg}, f, allow_unicode=True, sort_keys=False)
    save_run_info(args.output_dir)

    print(f"\nReal: {len(real_meta):,} images | Gen: {len(gen_meta):,} images")
    print(f"gen→real MSE: min={gen_mse_arr.min():.5f} median={np.median(gen_mse_arr):.5f}")
    if len(real_baseline_mse_arr):
        print(f"real→real baseline MSE: p1={baseline_p1:.5f} median={np.median(real_baseline_mse_arr):.5f}")
        print(f"Memorization candidates (gen mse < baseline p1): {n_candidates}")
    print(f"Saved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
