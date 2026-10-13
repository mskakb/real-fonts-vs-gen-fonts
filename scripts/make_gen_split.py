"""生成フォントデータセットの train/val/test スプリットを計算して保存する。

生成したスプリットファイルを train_scorer.py に渡すことで、
実験をまたいで同一スプリットを再現できる。

Usage:
    uv run python scripts/make_gen_split.py \\
        --gen-dir /path/to/generated \\
        --charset uppercase \\
        --output config/splits/gen_split_H_seed42.yaml
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import yaml

from real_fonts_vs_gen_fonts.datamodule import (
    _load_gen_meta,
    _make_label_map,
    _resolve_codepoints,
    _split_gen_by_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute and save generated data train/val/test split")
    parser.add_argument("--gen-dir", required=True, type=Path, metavar="DIR",
                        help="生成フォントディレクトリ (images/ + metadata.parquet)")
    parser.add_argument("--charset", default="uppercase",
                        help="文字セット (default: uppercase)")
    parser.add_argument("--val-split", type=float, default=0.15,
                        help="val セットの割合 (default: 0.15)")
    parser.add_argument("--test-split", type=float, default=0.15,
                        help="test セットの割合 (default: 0.15)")
    parser.add_argument("--seed", type=int, default=42,
                        help="ランダムシード (default: 42)")
    parser.add_argument("--output", required=True, type=Path, metavar="FILE",
                        help="スプリットファイルの保存先 (例: config/splits/gen_split_H_seed42.yaml)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    codepoints = _resolve_codepoints(args.charset)
    label_map = _make_label_map(codepoints)
    gen_meta = _load_gen_meta(args.gen_dir, label_map)

    train_df, val_df, test_df = _split_gen_by_file(
        gen_meta, args.val_split, args.test_split, args.seed
    )

    split = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "gen_dir": str(args.gen_dir),
        "charset": args.charset,
        "val_split": args.val_split,
        "test_split": args.test_split,
        "seed": args.seed,
        "n_samples": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
        },
        "files": {
            "train": sorted(train_df["file"].tolist()),
            "val": sorted(val_df["file"].tolist()),
            "test": sorted(test_df["file"].tolist()),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        yaml.dump(split, f, allow_unicode=True, sort_keys=False)

    print(f"Gen split saved: {args.output}")
    print(f"  train: {split['n_samples']['train']} samples")
    print(f"  val  : {split['n_samples']['val']} samples")
    print(f"  test : {split['n_samples']['test']} samples")


if __name__ == "__main__":
    main()
