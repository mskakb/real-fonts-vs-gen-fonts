"""Occlusion sensitivity による Scorer の判断根拠可視化。

入力の一部をパッチで遮蔽し、スコアの変化量を寄与マップにする。Grad-CAM と違い
レイヤ解像度に縛られず（stride で任意解像度）、backbone 非依存で SDF 入力にも対応する。

入力は extract_scorer_features.py と同じく `FontDataModule(mode="scorer")` から取り、
`batch["image"]`＝モデルが実際に見た入力（use_sdf なら SDF decode 済み）を遮蔽する。
このため --config は「学習済み Scorer の config.yaml」を指す（SDF kwarg 再現用）。

Usage:
    uv run python scripts/visualize_occlusion.py \\
        --checkpoint outputs/<exp>/checkpoints/last.ckpt \\
        --config     outputs/<exp>/config.yaml \\
        --output-dir /path/to/output

    # 単発の上書き（解像度・パッチ・遮蔽値）
    uv run python scripts/visualize_occlusion.py \\
        --checkpoint last.ckpt --config config.yaml \\
        --patch 12 --stride 6 --baseline mean --split val

    # 特定の文字だけ可視化（省略時はランダム全文字）
    uv run python scripts/visualize_occlusion.py \\
        --checkpoint last.ckpt --config config.yaml --char A B

    # 文字別に一括実行（モデル/データを使い回し、文字ごとにサブディレクトリへ出力）
    uv run python scripts/visualize_occlusion.py \\
        --checkpoint last.ckpt --config config.yaml \\
        --output-dir /path/to/output --all-chars
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib import cm
from matplotlib.colors import Normalize

from real_fonts_vs_gen_fonts.datamodule import FontDataModule
from real_fonts_vs_gen_fonts.models.occlusion import occlusion_sensitivity, overlay_occlusion
from real_fonts_vs_gen_fonts.models.scorer_module import ScorerModule
from real_fonts_vs_gen_fonts.utils.config import load_config, save_run_info, setup_logging
from real_fonts_vs_gen_fonts.utils.torch_utils import get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Occlusion sensitivity for Scorer")
    parser.add_argument("--checkpoint", required=True, metavar="CKPT",
                        help="ScorerModule checkpoint (.ckpt)")
    parser.add_argument("--config", required=True, metavar="FILE",
                        help="Trained Scorer config.yaml (for data / SDF kwargs)")
    parser.add_argument("--output-dir", required=True, metavar="DIR",
                        help="Output directory for occlusion figures")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"],
                        help="Which split to sample from (default: test)")
    parser.add_argument("--n-samples", type=int, default=25,
                        help="Number of real / gen samples each (default: 10)")
    parser.add_argument("--patch", type=int, default=8,
                        help="Occlusion patch size in px (default: 8)")
    parser.add_argument("--stride", type=int, default=4,
                        help="Occlusion stride in px; smaller = finer/slower (default: 4)")
    parser.add_argument("--baseline", default="background",
                        help="Patch fill: background/mean/min/max or a float (default: background)")
    parser.add_argument("--target", default="logit", choices=["logit", "score"],
                        help="Sensitivity target: logit (pre-sigmoid) or score (default: logit)")
    parser.add_argument("--select", default="random", choices=["random"],
                        help="Sample selection strategy (default: random)")
    parser.add_argument("--alpha", type=float, default=0.6,
                        help="Heatmap overlay opacity (default: 0.6)")
    parser.add_argument("--grid-cols", type=int, default=None,
                        help="Grid columns. Default: ceil(sqrt(n_samples)) for a square NxN "
                             "grid (poster-friendly); pass e.g. 5 for the old fixed layout")
    parser.add_argument("--cell-labels", action="store_true", default=True,
                        help="Show char/score caption per cell (default)")
    parser.add_argument("--no-cell-labels", dest="cell_labels", action="store_false",
                        help="Hide per-cell captions for a clean poster grid")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Occlusion forward chunk size (default: from config data.batch_size)")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed (default: 0)")
    parser.add_argument("--char", nargs="+", default=None, metavar="CHAR",
                        help="Restrict to specific character(s), e.g. --char A B "
                             "(default: random, no filter)")
    parser.add_argument("--all-chars", action="store_true",
                        help="Loop internally over all characters (or --char list), writing "
                             "per-character subdirectories instead of one combined grid")
    parser.add_argument("--use-ema", action="store_true", default=True,
                        help="Use EMA-averaged Scorer weights (default)")
    parser.add_argument("--no-ema", dest="use_ema", action="store_false",
                        help="Use raw (non-EMA) Scorer weights")
    parser.add_argument("overrides", nargs="*", metavar="key=value",
                        help="Config overrides, e.g. data.real_data_dir=/path")
    return parser.parse_args()


def _resolve_baseline_arg(baseline: str) -> str | float:
    """"background"/"mean"/... はそのまま、数値文字列は float に。"""
    try:
        return float(baseline)
    except ValueError:
        return baseline


def _occlusion_grid(
    images: list[torch.Tensor],
    heats: list[torch.Tensor],
    labels: list[str],
    alpha: float,
    output_path: Path,
    cols: int,
    cell_labels: bool = True,
) -> None:
    """符号付き感度マップのオーバレイをグリッド + 共有カラーバーで保存する。

    赤 = そこを隠すとスコア低下（realness を支持）/ 青 = スコア上昇（realness を抑制）。
    各セルは自身の max|Δ| で対称正規化するため、色は「符号と相対的な効き」を表す。
    """
    n = len(images)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2.3 + 0.4))
    axes = np.array(axes).reshape(rows, cols)

    for idx in range(rows * cols):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        ax.axis("off")
        if idx >= n:
            continue
        img = images[idx].permute(1, 2, 0).numpy()
        overlay = overlay_occlusion(img, heats[idx].numpy(), alpha=alpha,
                                    target_size=img.shape[:2])
        ax.imshow(overlay)
        if cell_labels:
            ax.set_title(labels[idx], fontsize=7)

    # 発散カラーマップの意味を示す共有カラーバー（各図は個別正規化なので単位なし）。
    sm = cm.ScalarMappable(norm=Normalize(vmin=-1.0, vmax=1.0), cmap="RdBu_r")
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    cbar.set_label("occlusion Δ (red: supports realness →, blue: opposes)",
                   fontsize=8)
    cbar.set_ticks([-1.0, 0.0, 1.0])
    cbar.set_ticklabels(["- (raises score)", "0", "+ (lowers score)"])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def _collect_split(loader: torch.utils.data.DataLoader) -> dict[int, list[dict]]:
    """split の全サンプルを label(1=real/0=gen) 別に集める。"""
    by_label: dict[int, list[dict]] = {0: [], 1: []}
    for batch in loader:
        for i in range(len(batch["label"])):
            lab = int(batch["label"][i])
            by_label[lab].append({
                "image": batch["image"][i],
                "char": str(batch["char"][i]),
            })
    return by_label


def main() -> None:
    setup_logging()
    args = parse_args()
    cfg = load_config(args.config, args.overrides)

    chunk = args.batch_size if args.batch_size is not None else cfg["data"]["batch_size"]
    baseline = _resolve_baseline_arg(args.baseline)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump({**cfg, "occlusion": vars(args)}, f, allow_unicode=True, sort_keys=False)
    save_run_info(output_dir, extra={
        "checkpoint": args.checkpoint, "split": args.split, "use_ema": args.use_ema,
        "patch": args.patch, "stride": args.stride, "baseline": args.baseline,
        "target": args.target, "n_samples": args.n_samples, "char": args.char,
    })

    device = get_device()
    print(f"Device: {device}")

    module: ScorerModule = ScorerModule.load_from_checkpoint(
        args.checkpoint, map_location="cpu"
    )
    module.eval().to(device)
    if args.use_ema:
        module._ema.shadow_params = [p.to(device) for p in module._ema.shadow_params]
        module._ema.copy_to(module.scorer.parameters())
    scorer = module.scorer

    # SDF kwargs must mirror train_scorer.py / extract_scorer_features.py: an SDF-trained
    # Scorer expects SDF input, so omitting these silently feeds plain glyphs and makes
    # every occluded score (and thus the whole heatmap) meaningless.
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
        use_sdf=cfg["data"].get("use_sdf", False),
        sdf_max_dist=cfg["data"].get("sdf_max_dist", 10.0),
        sdf_source=cfg["data"].get("sdf_source", "auto"),
        sdf_decode=cfg["data"].get("sdf_decode"),
        sdf_sharpness=cfg["data"].get("sdf_sharpness"),
        sdf_filter_width_px=cfg["data"].get("sdf_filter_width_px", 1.5),
    )
    dm.setup(None)
    loader = {"train": dm.train_dataloader, "val": dm.val_dataloader,
              "test": dm.test_dataloader}[args.split]()

    by_label = _collect_split(loader)
    rng = random.Random(args.seed)

    def run(label: int, name: str, out_path: Path, char_filter: list[str] | None) -> list[float]:
        pool = by_label[label]
        if char_filter is not None:
            pool = [p for p in pool if p["char"] in char_filter]
            if not pool:
                available = sorted({p["char"] for p in by_label[label]})
                raise ValueError(
                    f"No samples for char={char_filter} label={label}. Available: {available}"
                )
        picks = rng.sample(pool, min(args.n_samples, len(pool)))
        imgs = torch.stack([p["image"] for p in picks]).to(device)
        heats = occlusion_sensitivity(
            scorer, imgs, patch_size=args.patch, stride=args.stride,
            baseline=baseline, target=args.target, chunk_size=chunk,
        )
        with torch.no_grad():
            scores = torch.sigmoid(scorer.forward_logits(imgs)).cpu().tolist()
        labels = [f"{p['char']}  s={s:.3f}" for p, s in zip(picks, scores)]
        cols = args.grid_cols if args.grid_cols is not None else math.ceil(math.sqrt(len(picks)))
        _occlusion_grid([p["image"].cpu() for p in picks], list(heats), labels,
                        args.alpha, out_path, cols=cols, cell_labels=args.cell_labels)
        print(f"{name}: n={len(picks)}  mean score={np.mean(scores):.4f}")
        return scores

    if args.all_chars:
        chars = args.char if args.char is not None else sorted(
            {p["char"] for pool in by_label.values() for p in pool}
        )
        for c in chars:
            char_dir = output_dir / c
            char_dir.mkdir(parents=True, exist_ok=True)
            print(f"Processing char={c}...")
            real_scores = run(1, "Real", char_dir / "occlusion_real.png", [c])
            gen_scores = run(0, "Gen ", char_dir / "occlusion_gen.png", [c])
            print(f"  char={c}  Real mean score: {np.mean(real_scores):.4f}"
                  f"  Gen mean score: {np.mean(gen_scores):.4f}")
    else:
        print("Processing real images...")
        real_scores = run(1, "Real", output_dir / "occlusion_real.png", args.char)
        print("Processing generated images...")
        gen_scores = run(0, "Gen ", output_dir / "occlusion_gen.png", args.char)

        print(f"\nReal mean score: {np.mean(real_scores):.4f} ± {np.std(real_scores):.4f}")
        print(f"Gen  mean score: {np.mean(gen_scores):.4f} ± {np.std(gen_scores):.4f}")


if __name__ == "__main__":
    main()
