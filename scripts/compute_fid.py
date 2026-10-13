"""Compute FID between real and generated font images.

FID (Fréchet Inception Distance) を実フォントと生成フォントの間で計算する。
InceptionV3 特徴量を使用 (piq.feature_extractors.InceptionV3 + piq.FID)。
PCA・t-SNE による特徴量の可視化も保存する。

Note: FID の絶対値はフォント画像と自然画像の分布差により参考値。
      実験間の相対比較に使うこと。
      統計的安定性のため 2,000 枚以上を推奨。

Usage:
    uv run python scripts/compute_fid.py \\
        --real-dir /path/to/real_fonts \\
        --gen-dir  /path/to/generated \\
        --output-dir /path/to/output

    uv run python scripts/compute_fid.py \\
        --real-dir /path/to/real_fonts \\
        --gen-dir  /path/to/generated \\
        --charset H --batch-size 32 \\
        --output-dir /path/to/output
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml
from piq import FID
from piq.feature_extractors import InceptionV3
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
from real_fonts_vs_gen_fonts.utils.torch_utils import get_device, extract_features
from real_fonts_vs_gen_fonts.utils.visualization import visualize_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute FID between real and generated fonts")
    parser.add_argument("--config", required=True, metavar="FILE",
                        help="YAML config file (see config/eval/compute_fid.yaml)")
    parser.add_argument("overrides", nargs="*", metavar="key=value",
                        help="Config overrides, e.g. split=val batch_size=32")
    return parser.parse_args()


def main() -> None:
    raw = parse_args()
    cfg = load_config(raw.config, raw.overrides)
    # SimpleNamespace adapter: keep the body referring to args.* unchanged.
    args = SimpleNamespace(
        real_dir=Path(cfg["real_dir"]),
        gen_dir=Path(cfg["gen_dir"]),
        charset=cfg["charset"],
        image_size=cfg["image_size"],
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        tsne_max_samples=cfg["tsne_max_samples"],
        tsne_perplexity=cfg["tsne_perplexity"],
        split_file=Path(cfg["split_file"]) if cfg.get("split_file") else None,
        split=cfg["split"],
        output_dir=Path(cfg["output_dir"]),
    )

    device = get_device()
    print(f"Device: {device}")

    codepoints = _resolve_codepoints(args.charset)
    label_map = _make_label_map(codepoints)

    # FID 計算時は binarize しない（InceptionV3 は連続値画像を期待）
    transform = make_image_transform(args.image_size)

    real_meta = _load_real_meta(args.real_dir, "diffusion", label_map)

    if args.split_file is not None and args.split != "all":
        train_df, val_df, test_df = _load_split(args.split_file, real_meta)
        split_map = {"train": train_df, "val": val_df, "test": test_df}
        real_meta = split_map[args.split]
        print(f"Split: {args.split} ({len(real_meta):,} samples from {args.split_file.name})")

    gen_meta = _load_gen_meta(args.gen_dir, label_map)

    real_ds = FontDataset(real_meta, args.real_dir / "images", transform)
    gen_ds = FontDataset(gen_meta, args.gen_dir / "images", transform)

    print(f"Real : {len(real_ds):,} images  (split={args.split})")
    print(f"Gen  : {len(gen_ds):,} images")
    if min(len(real_ds), len(gen_ds)) < 2000:
        print("WARNING: sample count < 2000. FID may be statistically unstable.")

    loader_kwargs = dict(
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    real_loader = DataLoader(real_ds, **loader_kwargs)
    gen_loader = DataLoader(gen_ds, **loader_kwargs)

    # InceptionV3 特徴抽出
    encoder = InceptionV3(normalize_input=True).to(device).eval()
    real_feats = extract_features(real_loader, encoder, device, desc="Real", output_index=0)
    gen_feats = extract_features(gen_loader, encoder, device, desc="Gen ", output_index=0)

    # FID 計算
    fid_metric = FID()
    score = fid_metric(real_feats, gen_feats).item()
    print(f"\nFID: {score:.4f}")

    # 保存
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 可視化
    visualize_features(
        real_feats, gen_feats,
        output_dir=args.output_dir,
        tsne_max_samples=args.tsne_max_samples,
        tsne_perplexity=args.tsne_perplexity,
    )

    result = {
        "fid": round(score, 4),
        "charset": args.charset,
        "split": args.split,
        "split_file": str(args.split_file) if args.split_file else None,
        "n_real": len(real_ds),
        "n_gen": len(gen_ds),
        "image_size": args.image_size,
        "real_dir": str(args.real_dir),
        "gen_dir": str(args.gen_dir),
    }
    with open(args.output_dir / "fid_result.yaml", "w") as f:
        yaml.dump(result, f, allow_unicode=True, sort_keys=False)
    save_run_info(args.output_dir)
    print(f"Saved: {args.output_dir}/fid_result.yaml")


if __name__ == "__main__":
    main()
