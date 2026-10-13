"""Compute realness scores for all images in a dataset directory.

Loads metadata.parquet, scores every image with a trained ScorerModule,
and writes results to an output file (or back to metadata.parquet in-place).

Works on any directory that follows the font2dataset format:
  images/ + metadata.parquet

Usage:
    # in-place write (legacy)
    uv run python scripts/score_images.py \\
        --scorer outputs/<exp>/checkpoints/last.ckpt \\
        --data-dir data/raw/generated

    # write to a separate file per data-dir
    uv run python scripts/score_images.py \\
        --scorer path/to/scorer.ckpt \\
        --data-dir data/raw/real data/raw/generated \\
        --output-dir results/scores/2026-05-26

    # single output file (only works with one --data-dir)
    uv run python scripts/score_images.py \\
        --scorer path/to/scorer.ckpt \\
        --data-dir data/raw/real \\
        --output results/scores/real_scores.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import torch

from real_fonts_vs_gen_fonts.models.scorer_module import ScorerModule
from real_fonts_vs_gen_fonts.utils.config import load_config, save_run_info
from real_fonts_vs_gen_fonts.utils.image import make_image_transform
from real_fonts_vs_gen_fonts.utils.scoring import score_directory
from real_fonts_vs_gen_fonts.utils.torch_utils import get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute realness scores and save results"
    )
    parser.add_argument("--config", required=True, metavar="FILE",
                        help="YAML config file (see config/eval/score_images.yaml)")
    parser.add_argument("overrides", nargs="*", metavar="key=value",
                        help="Config overrides, e.g. batch_size=256 overwrite=true")
    return parser.parse_args()


def main() -> None:
    raw = parse_args()
    cfg = load_config(raw.config, raw.overrides)
    # SimpleNamespace adapter: keep the body referring to args.* unchanged.
    args = SimpleNamespace(
        scorer=cfg["scorer_ckpt"],
        data_dir=cfg["data_dirs"],
        image_size=cfg["image_size"],
        batch_size=cfg["batch_size"],
        overwrite=cfg["overwrite"],
        binarize_threshold=cfg["binarize_threshold"],
        use_ema=cfg["use_ema"],
        output=None,
        output_dir=cfg["output_dir"],
    )

    if args.output and len(args.data_dir) > 1:
        raise SystemExit("--output can only be used with a single --data-dir. "
                         "Use --output-dir for multiple directories.")

    device = get_device()

    scorer = ScorerModule.load_from_checkpoint(args.scorer)
    scorer.eval().to(device)
    if args.use_ema:
        scorer._ema.shadow_params = [p.to(device) for p in scorer._ema.shadow_params]
        scorer._ema.copy_to(scorer.scorer.parameters())

    transform = make_image_transform(args.image_size, args.binarize_threshold)

    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        save_run_info(output_dir, extra=cfg)

    for data_dir in args.data_dir:
        data_dir_path = Path(data_dir)

        if args.output:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
        elif output_dir:
            out_path = output_dir / f"{data_dir_path.name}_scores.parquet"
        else:
            out_path = None  # in-place

        score_directory(
            data_dir_path,
            scorer,
            transform,
            args.batch_size,
            device,
            args.overwrite,
            output_path=out_path,
        )


if __name__ == "__main__":
    main()
