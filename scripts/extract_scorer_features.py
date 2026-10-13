"""Extract intermediate features from a trained ScorerModule and save to parquet.

This parquet is the single source of truth for the CPU-side analysis in
`analyze_scorer_features.py` / `notebooks/scorer_feature_analysis.ipynb`
(combined-space embeddings, per-layer probes, per-char/per-font stats).

For each sample in train/val/test, records:
  - GAP features from all ResNet stages (layer1-layer4), column-prefixed l1_/l2_/l3_/l4_
    (dims are backbone-dependent: resnet18 64/128/256/512, resnet50 256/512/1024/2048)
  - realness score (sigmoid output) and the raw logit (pre-sigmoid)
  - metadata: split, label (1=real / 0=gen), char, font_family
  - image_path: relative path to a saved thumbnail of the model-input image
    (the post-transform tensor the model actually saw — SDF-decoded if use_sdf),
    written under <output_dir>/inputs/<split>/. Disable with --no-save-inputs.
    Note: train uses the augmenting+shuffling train_dataloader, so its saved
    thumbnails are augmented (image plots in analyze default to the test split).

Usage:
    uv run python scripts/extract_scorer_features.py \\
        --checkpoint outputs/<exp>/checkpoints/last.ckpt \\
        --config config/scorer/scorer.yaml \\
        data.real_data_dir=data/raw/real \\
        data.gen_data_dir=data/raw/generated

    # custom output dir
    uv run python scripts/extract_scorer_features.py \\
        --checkpoint last.ckpt \\
        --output-dir results/features \\
        --split val

    # override config values (same as train_scorer.py)
    uv run python scripts/extract_scorer_features.py \\
        --checkpoint last.ckpt \\
        data.batch_size=128
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from real_fonts_vs_gen_fonts.datamodule import FontDataModule
from real_fonts_vs_gen_fonts.models.scorer_module import ScorerModule
from real_fonts_vs_gen_fonts.utils.config import load_config, save_run_info
from real_fonts_vs_gen_fonts.utils.torch_utils import get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract Scorer intermediate features to parquet"
    )
    parser.add_argument("--checkpoint", required=True, metavar="CKPT",
                        help="ScorerModule checkpoint (.ckpt)")
    parser.add_argument("--config", default="config/scorer/scorer.yaml", metavar="FILE",
                        help="YAML config file (default: config/scorer/scorer.yaml)")
    parser.add_argument("--output-dir", metavar="DIR",
                        help="Output directory for parquet (default: <ckpt_dir>/features/)")
    parser.add_argument("--split", default="all",
                        choices=["train", "val", "test", "all"],
                        help="Which split(s) to process (default: all)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size (default: from config)")
    parser.add_argument(
        "--use-ema", action="store_true", default=True,
        help="Use EMA-averaged Scorer weights (default)",
    )
    parser.add_argument(
        "--no-ema", dest="use_ema", action="store_false",
        help="Use raw (non-EMA) Scorer weights",
    )
    parser.add_argument(
        "--save-inputs", action="store_true", default=True,
        help="Save a thumbnail of each model-input image for analyze image plots (default)",
    )
    parser.add_argument(
        "--no-save-inputs", dest="save_inputs", action="store_false",
        help="Do not save input thumbnails (no image_path column; image plots disabled)",
    )
    parser.add_argument("overrides", nargs="*", metavar="key=value",
                        help="Config overrides, e.g. data.real_data_dir=/path/to/real")
    return parser.parse_args()


def _save_input_thumb(img: torch.Tensor, path: Path) -> None:
    """Save a model-input tensor (C, H, W) in [0, 1] as an 8-bit grayscale PNG.

    Collapses channels (the 3ch input is a replicated grayscale glyph / SDF) so the
    saved thumbnail is exactly what the model saw, decoded for human viewing.
    """
    arr = img.mean(dim=0).clamp(0.0, 1.0).numpy()
    Image.fromarray((arr * 255.0).round().astype(np.uint8), mode="L").save(path)


@torch.no_grad()
def _extract_split(
    loader: torch.utils.data.DataLoader,
    scorer_module: ScorerModule,
    device: torch.device,
    split_name: str,
    inputs_dir: Path | None = None,
) -> pd.DataFrame:
    records: list[dict] = []
    scorer = scorer_module.scorer
    # layer name -> column prefix (layer1 -> "l1_", ...); keep in sync with
    # real_fonts_vs_gen_fonts.analysis.feature_analysis.LAYER_PREFIXES
    layer_prefix = {f"layer{i}": f"l{i}_" for i in range(1, 5)}

    if inputs_dir is not None:
        split_inputs_dir = inputs_dir / split_name
        split_inputs_dir.mkdir(parents=True, exist_ok=True)

    counter = 0
    for batch in tqdm(loader, desc=split_name, leave=False):
        cpu_imgs = batch["image"]                              # (B, C, H, W) in [0, 1]
        imgs = cpu_imgs.to(device)
        inter = scorer.extract_intermediate(imgs)             # dict layer -> (B, C_l)
        inter = {k: v.cpu().numpy() for k, v in inter.items()}
        logits = scorer.forward_logits(imgs).cpu().numpy()    # (B,)
        scores = torch.sigmoid(torch.from_numpy(logits)).numpy()
        labels = batch["label"].numpy()
        chars = batch["char"]
        families = batch["font_family"]

        for i in range(len(labels)):
            row: dict = {
                "split": split_name,
                "label": int(labels[i]),
                "char": chars[i],
                "font_family": families[i],
                "score": float(scores[i]),
                "logit": float(logits[i]),
            }
            if inputs_dir is not None:
                rel = Path("inputs") / split_name / f"{counter:06d}.png"
                _save_input_thumb(cpu_imgs[i], inputs_dir.parent / rel)
                row["image_path"] = str(rel)
            for layer, prefix in layer_prefix.items():
                for j, v in enumerate(inter[layer][i]):
                    row[f"{prefix}{j}"] = float(v)
            records.append(row)
            counter += 1

    return pd.DataFrame(records)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.overrides)

    if args.batch_size is not None:
        cfg["data"]["batch_size"] = args.batch_size

    ckpt_path = Path(args.checkpoint)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = ckpt_path.parent.parent / "features"
    output_dir.mkdir(parents=True, exist_ok=True)

    save_run_info(output_dir, extra={
        "checkpoint": str(ckpt_path),
        "split": args.split,
        "use_ema": args.use_ema,
        "save_inputs": args.save_inputs,
        "backbone": cfg.get("model", {}).get("backbone", "resnet18"),
    })

    device = get_device()

    # load model (map to CPU first, then move to device for flexibility)
    scorer_module: ScorerModule = ScorerModule.load_from_checkpoint(
        ckpt_path, map_location="cpu"
    )
    scorer_module.eval().to(device)
    if args.use_ema:
        scorer_module._ema.shadow_params = [p.to(device) for p in scorer_module._ema.shadow_params]
        scorer_module._ema.copy_to(scorer_module.scorer.parameters())

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
        # SDF kwargs must mirror train_scorer.py: an SDF-trained Scorer expects SDF
        # input, so omitting these silently feeds plain glyphs and corrupts every
        # extracted feature/score/logit (and the saved input thumbnails).
        use_sdf=cfg["data"].get("use_sdf", False),
        sdf_max_dist=cfg["data"].get("sdf_max_dist", 10.0),
        sdf_source=cfg["data"].get("sdf_source", "auto"),
        sdf_decode=cfg["data"].get("sdf_decode"),
        sdf_sharpness=cfg["data"].get("sdf_sharpness"),
        sdf_filter_width_px=cfg["data"].get("sdf_filter_width_px", 1.5),
    )
    dm.setup(None)

    splits_to_run = (
        ["train", "val", "test"] if args.split == "all" else [args.split]
    )
    loader_map = {
        "train": dm.train_dataloader,
        "val": dm.val_dataloader,
        "test": dm.test_dataloader,
    }

    inputs_dir = output_dir / "inputs" if args.save_inputs else None

    dfs: list[pd.DataFrame] = []
    for split_name in splits_to_run:
        loader = loader_map[split_name]()
        df = _extract_split(loader, scorer_module, device, split_name, inputs_dir)
        print(f"  {split_name}: {len(df):,} samples")
        dfs.append(df)

    result = pd.concat(dfs, ignore_index=True)
    out_path = output_dir / "scorer_features.parquet"
    result.to_parquet(out_path, index=False)
    print(f"\nSaved {len(result):,} rows × {len(result.columns)} cols → {out_path}")

    meta_cols = [c for c in result.columns if not c.split("_")[0] in {"l1", "l2", "l3", "l4"}]
    print(f"  metadata cols: {meta_cols}")
    if inputs_dir is not None:
        print(f"  input thumbnails: {inputs_dir} (image_path column relative to parquet)")
    for layer in ("l1", "l2", "l3", "l4"):
        n = sum(c.startswith(f"{layer}_") for c in result.columns)
        if n:
            print(f"  {layer}_* : {n} cols")


if __name__ == "__main__":
    main()
