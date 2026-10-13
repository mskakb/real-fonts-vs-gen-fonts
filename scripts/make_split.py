"""実フォントデータセットの train/val/test スプリットを計算して保存する。

生成したスプリットファイルを train_diffusion.py / train_scorer.py に渡すことで、
全実験で同一スプリットを使用でき、公平な比較が可能になる。

Usage:
    uv run python scripts/make_split.py \\
        --real-dir /path/to/real_fonts \\
        --charset uppercase \\
        --output /path/to/split.yaml
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import yaml

from real_fonts_vs_gen_fonts.datamodule import (
    _load_real_meta,
    _make_label_map,
    _resolve_codepoints,
    _split_by_family,
)
from real_fonts_vs_gen_fonts.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute and save train/val/test split")
    parser.add_argument("--config", required=True, metavar="FILE",
                        help="YAML config file (see config/splits/make_split.yaml)")
    parser.add_argument("overrides", nargs="*", metavar="key=value",
                        help="Config overrides, e.g. seed=43 charset=uppercase")
    return parser.parse_args()


def main() -> None:
    raw = parse_args()
    cfg = load_config(raw.config, raw.overrides)
    # SimpleNamespace adapter: keep the body referring to args.* unchanged.
    args = SimpleNamespace(
        real_dir=Path(cfg["real_dir"]),
        charset=cfg["charset"],
        val_split=cfg["val_split"],
        test_split=cfg["test_split"],
        seed=cfg["seed"],
        output=Path(cfg["output"]),
    )

    codepoints = _resolve_codepoints(args.charset)
    label_map = _make_label_map(codepoints)
    meta = _load_real_meta(args.real_dir, "diffusion", label_map)

    train_df, val_df, test_df = _split_by_family(
        meta, args.val_split, args.test_split, args.seed
    )

    def families(df):
        return sorted(df["font_family"].unique().tolist())

    split = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "real_dir": str(args.real_dir),
        "charset": args.charset,
        "val_split": args.val_split,
        "test_split": args.test_split,
        "seed": args.seed,
        "n_families": {
            "train": len(families(train_df)),
            "val": len(families(val_df)),
            "test": len(families(test_df)),
        },
        "n_samples": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
        },
        "families": {
            "train": families(train_df),
            "val": families(val_df),
            "test": families(test_df),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        yaml.dump(split, f, allow_unicode=True, sort_keys=False)

    print(f"Split saved: {args.output}")
    print(f"  train: {split['n_families']['train']} families / {split['n_samples']['train']} samples")
    print(f"  val  : {split['n_families']['val']} families / {split['n_samples']['val']} samples")
    print(f"  test : {split['n_families']['test']} families / {split['n_samples']['test']} samples")


if __name__ == "__main__":
    main()
