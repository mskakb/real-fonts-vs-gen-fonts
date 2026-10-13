"""ViT Scorer の attention 系 attribution マップを occlusion sensitivity で忠実性検証する。

`models/attention_maps.py` の 5 手法（cls_attention / attention_rollout / chefer_relevance /
token_grad_cam / token_norm_map）は、いずれも「attention（または勾配）からどこを見ているかを
"相関的に" 推定する」手法であり、attention weight が feature importance の忠実な説明にならない
ことは繰り返し示されている（Jain & Wallace 2019 / Serrano & Smith 2019 / Wiegreffe & Pinter 2019）。
本スクリプトは各手法を `models/occlusion.py` の occlusion sensitivity（実際にパッチを遮蔽して
スコア変化を測る、**因果的**な基準）と同一の 12x12 トークン/パッチグリッド上で突き合わせ、
一致度（順位相関）と忠実性（deletion/insertion AUC）を定量評価する。

**`occlusion_delta` は忠実性評価の目的関数（累積遮蔽時のスコア変化）に対する貪欲順序そのもの
なので、この指標群の上限リファレンスであり競合手法ではない**。`random`（一様乱数順）は下限。
各手法がこの区間のどこに位置するかで、attention ベースの説明がどれだけ因果的な根拠に近いかを見る。

`--config` は**学習済み Scorer の `config.yaml`**（データ/SDF kwarg 再現用、analyze_occlusion.py
と同じ規約）。対象は ViT backbone の Scorer のみ（ResNet は Grad-CAM/occlusion で対応済み、
`GradCAM` は ViT を渡すと明示的に ValueError を出す）。

`--patch`/`--stride` は指定しない: グリッドを ViT の `patch_size`（config の `model.backbone`
から自動解決）に厳密に一致させることが本スクリプトの前提のため、常に
`patch_size=stride=vit_patch_size, min_change=0` で occlusion を計算する。

Usage:
    uv run python scripts/analyze_attention.py \\
        --checkpoint outputs/<exp>/checkpoints/last.ckpt \\
        --config     outputs/<exp>/config.yaml \\
        --char H --label real --output-dir /path/to/output

    # 全文字 x real/gen、各4枚
    uv run python scripts/analyze_attention.py \\
        --checkpoint last.ckpt --config config.yaml \\
        --all-chars --label both --n-images 4
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.stats import kendalltau, spearmanr

from real_fonts_vs_gen_fonts.datamodule import FontDataModule, _resolve_codepoints
from real_fonts_vs_gen_fonts.models.attention_maps import (
    attention_rollout,
    chefer_relevance,
    cls_attention,
    token_grad_cam,
    token_norm_map,
)
from real_fonts_vs_gen_fonts.models.occlusion import (
    build_occluded_batch,
    cumulative_curve_batches,
)
from real_fonts_vs_gen_fonts.models.scorer_module import ScorerModule
from real_fonts_vs_gen_fonts.models.vit_backbone import VIT_SPECS
from real_fonts_vs_gen_fonts.utils.config import load_config, save_run_info, setup_logging
from real_fonts_vs_gen_fonts.utils.torch_utils import get_device

_WHITE_VALUE = 1.0
"""「消す」ときの塗り値。analyze_occlusion.py の s_blank 基準（sdf_decode=hard/smoothstep や
素の二値画像で background=1.0）と同じ規約。raw SDF は極性が逆になる点に注意。"""

# 勾配を要する手法（no_grad で呼ぶと壊れる）。それ以外は @torch.no_grad() 内蔵。
_GRAD_METHODS = {"chefer_relevance", "token_grad_cam"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Faithfulness check of ViT attention-based attribution vs. occlusion sensitivity"
    )
    parser.add_argument("--checkpoint", required=True, metavar="CKPT",
                        help="ScorerModule checkpoint (.ckpt), ViT backbone")
    parser.add_argument("--config", required=True, metavar="FILE",
                        help="Trained Scorer config.yaml (for data / SDF kwargs)")
    parser.add_argument("--output-dir", default=None, metavar="DIR",
                        help="Output directory (default: <ckpt_dir>/../attention_analysis/)")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"],
                        help="Which split to sample images from (default: test)")
    parser.add_argument("--char", nargs="+", default=None, metavar="CHAR",
                        help="Restrict image selection to specific character(s)")
    parser.add_argument("--all-chars", action="store_true",
                        help="Loop internally over all charset characters (or --char list)")
    parser.add_argument("--label", default="real", choices=["real", "gen", "both"],
                        help="Which class to sample images from (default: real)")
    parser.add_argument("--n-images", type=int, default=4,
                        help="Number of images to analyze per (char, label) (default: 4)")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed (default: 0)")
    parser.add_argument("--baseline", default="background",
                        help="Occlusion patch fill: background/mean/min/max or a float "
                             "(default: background)")
    parser.add_argument("--target", default="logit", choices=["logit", "score"],
                        help="Attribution/curve target: logit (pre-sigmoid) or score (default: logit)")
    parser.add_argument("--grad-cam-layer", default="layer4",
                        choices=["layer1", "layer2", "layer3", "layer4"],
                        help="Stage for token_grad_cam / token_norm_map (default: layer4)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Occlusion/curve forward chunk size (default: from config data.batch_size)")
    parser.add_argument("--use-ema", action="store_true", default=True,
                        help="Use EMA-averaged Scorer weights (default)")
    parser.add_argument("--no-ema", dest="use_ema", action="store_false",
                        help="Use raw (non-EMA) Scorer weights")
    parser.add_argument("overrides", nargs="*", metavar="key=value",
                        help="Config overrides, e.g. data.real_data_dir=/path")
    return parser.parse_args()


def _resolve_baseline_arg(baseline: str) -> str | float:
    try:
        return float(baseline)
    except ValueError:
        return baseline


def _collect_split(loader: torch.utils.data.DataLoader) -> dict[int, list[dict]]:
    by_label: dict[int, list[dict]] = {0: [], 1: []}
    for batch in loader:
        for i in range(len(batch["label"])):
            lab = int(batch["label"][i])
            by_label[lab].append({
                "image": batch["image"][i],
                "char": str(batch["char"][i]),
                "font_family": str(batch["font_family"][i]),
                "image_path": str(batch["image_path"][i]),
            })
    return by_label


def _forward_chunked(scorer, x: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """x を chunk_size ごとに forward し、sigmoid スコアを連結して返す（曲線描画専用、勾配不要）。"""
    scores = []
    with torch.no_grad():
        for start in range(0, len(x), chunk_size):
            logits = scorer.forward_logits(x[start : start + chunk_size])
            scores.append(torch.sigmoid(logits).cpu())
    return torch.cat(scores)


def _to_gray(x: torch.Tensor) -> np.ndarray:
    return x.mean(dim=1).clamp(0.0, 1.0).cpu().numpy()


def _compute_attributions(
    scorer, image: torch.Tensor, args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    """5 手法の attribution を計算し、各 (grid*grid,) の flat array を返す（CLS 除外済み）。"""
    x = image.unsqueeze(0)
    out: dict[str, np.ndarray] = {}
    out["cls_attention"] = cls_attention(scorer, x).squeeze(0).cpu().numpy()
    out["attention_rollout"] = attention_rollout(scorer, x).squeeze(0).cpu().numpy()
    out["chefer_relevance"] = (
        chefer_relevance(scorer, x, target=args.target).squeeze(0).cpu().numpy()
    )
    out["token_grad_cam"] = (
        token_grad_cam(scorer, x, layer=args.grad_cam_layer, target=args.target)
        .squeeze(0).cpu().numpy()
    )
    out["token_norm"] = (
        token_norm_map(scorer, x, layer=args.grad_cam_layer).squeeze(0).cpu().numpy()
    )
    return out


def _plot_attribution_maps(
    image: np.ndarray,
    grid: int,
    attributions: dict[str, np.ndarray],
    occ_delta: np.ndarray,
    output_path: Path,
) -> None:
    """入力 + 各手法の attribution ヒートマップ + occlusion delta（因果的リファレンス）を並べる。

    非負な手法（attention/rollout/chefer/grad_cam/token_norm）は各図ごとに [min,max] 正規化した
    'hot'、符号付きの occlusion delta は 0 中心対称の 'RdBu_r' で描く（既存 occlusion 図の配色
    規約に合わせる）。
    """
    methods = list(attributions) + ["occlusion_delta (reference)"]
    n = len(methods) + 1
    fig, axes = plt.subplots(1, n, figsize=(2.4 * n, 2.6))

    axes[0].imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title("input", fontsize=8)
    axes[0].axis("off")

    for ax, name in zip(axes[1:], methods):
        if name == "occlusion_delta (reference)":
            m = occ_delta.reshape(grid, grid)
            vmax = float(np.abs(m).max()) + 1e-8
            ax.imshow(m, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        else:
            m = attributions[name].reshape(grid, grid)
            lo, hi = float(m.min()), float(m.max())
            ax.imshow((m - lo) / (hi - lo + 1e-8), cmap="hot", vmin=0.0, vmax=1.0)
        ax.set_title(name, fontsize=8)
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def _plot_faithfulness_curves(
    area_frac: np.ndarray,
    curves: dict[str, dict[str, np.ndarray]],
    output_path: Path,
) -> None:
    """手法ごとの deletion/insertion 曲線を重ね描きする。occlusion_delta は上限、random は下限として
    破線で区別する。
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    cmap = plt.get_cmap("tab10")
    for i, (name, c) in enumerate(curves.items()):
        style = "--" if name in ("occlusion_delta", "random") else "-"
        color = cmap(i % 10)
        axes[0].plot(area_frac, c["deletion"], style, color=color,
                    label=f"{name} (AUC={c['deletion_auc']:.3f})", markersize=2)
        axes[1].plot(area_frac, c["insertion"], style, color=color,
                    label=f"{name} (AUC={c['insertion_auc']:.3f})", markersize=2)
    axes[0].set_xlabel("cumulative area fraction (attribution-descending order)")
    axes[0].set_ylabel("score")
    axes[0].set_title("deletion (lower AUC = sharper)", fontsize=9)
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].legend(loc="best", fontsize=6)
    axes[1].set_xlabel("cumulative area fraction (attribution-descending order)")
    axes[1].set_ylabel("score")
    axes[1].set_title("insertion (higher AUC = sharper)", fontsize=9)
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].legend(loc="best", fontsize=6)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def analyze_one_image(
    scorer,
    image: torch.Tensor,
    char: str,
    label_name: str,
    args: argparse.Namespace,
    baseline: str | float,
    patch_size: int,
    grid: int,
    chunk: int,
    output_dir: Path,
    target_meta: dict,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    device = image.device
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- attention-based attributions (5 methods) ---
    attributions = _compute_attributions(scorer, image, args)

    # --- occlusion sensitivity: causal reference (min_change=0 so the grid stays 12x12 dense) ---
    occ, positions = build_occluded_batch(
        image, patch_size=patch_size, stride=patch_size, baseline=baseline, min_change=0.0,
    )
    occ = occ.to(device)
    assert len(positions) == grid * grid, (
        f"occlusion grid ({len(positions)}) != ViT token grid ({grid * grid}); "
        f"patch_size/stride must equal the ViT's own patch_size."
    )
    with torch.no_grad():
        s0_logit = scorer.forward_logits(image.unsqueeze(0)).cpu()
        occ_logits = torch.cat([
            scorer.forward_logits(occ[start : start + chunk]).cpu()
            for start in range(0, len(occ), chunk)
        ])
    target_s0 = float(s0_logit.item()) if args.target == "logit" else float(
        torch.sigmoid(s0_logit).item()
    )
    target_occ = occ_logits if args.target == "logit" else torch.sigmoid(occ_logits)
    occ_delta = (target_s0 - target_occ.numpy())
    attributions["occlusion_delta"] = occ_delta

    # --- random baseline (null model) ---
    rng = np.random.default_rng(args.seed)
    attributions["random"] = rng.permutation(grid * grid).astype(np.float64)

    # --- agreement vs occlusion_delta (causal reference) ---
    agreement_rows = []
    for name in list(attributions):
        if name == "occlusion_delta":
            continue
        vals = attributions[name]
        rho, _ = spearmanr(vals, occ_delta)
        tau, _ = kendalltau(vals, occ_delta)
        k10 = len(set(np.argsort(-vals)[:10]) & set(np.argsort(-occ_delta)[:10])) / 10
        k20 = len(set(np.argsort(-vals)[:20]) & set(np.argsort(-occ_delta)[:20])) / 20
        agreement_rows.append({
            "char": char, "label": label_name, "method": name,
            "spearman_rho": rho, "kendall_tau": tau,
            "topk10_overlap": k10, "topk20_overlap": k20,
        })
    agreement_df = pd.DataFrame(agreement_rows)

    # --- faithfulness: deletion/insertion curves per method ---
    curves: dict[str, dict[str, float]] = {}
    curve_data: dict[str, dict[str, np.ndarray]] = {}
    area_frac_ref = None
    for name, vals in attributions.items():
        order = np.argsort(-vals)
        del_batch, ins_batch, area_frac = cumulative_curve_batches(
            image, positions, order, patch_size, _WHITE_VALUE
        )
        del_curve = _forward_chunked(scorer, del_batch, chunk).numpy()
        ins_curve = _forward_chunked(scorer, ins_batch, chunk).numpy()
        deletion_auc = float(np.trapezoid(del_curve, area_frac))
        insertion_auc = float(np.trapezoid(ins_curve, area_frac))
        curves[name] = {"deletion_auc": deletion_auc, "insertion_auc": insertion_auc}
        curve_data[name] = {
            "deletion": del_curve, "insertion": ins_curve,
            "deletion_auc": deletion_auc, "insertion_auc": insertion_auc,
        }
        area_frac_ref = area_frac

    faithfulness_rows = [
        {"char": char, "label": label_name, "method": name, **metrics}
        for name, metrics in curves.items()
    ]
    faithfulness_df = pd.DataFrame(faithfulness_rows)

    # --- per-token raw values ---
    tokens_df = pd.DataFrame({
        "top": [p[0] for p in positions], "left": [p[1] for p in positions],
        **{name: vals for name, vals in attributions.items()},
    })
    tokens_df.to_csv(output_dir / "attribution_maps.csv", index=False)
    agreement_df.to_csv(output_dir / "attribution_agreement.csv", index=False)
    faithfulness_df.to_csv(output_dir / "faithfulness.csv", index=False)

    img_np = _to_gray(image.unsqueeze(0))[0]
    _plot_attribution_maps(
        img_np, grid, {k: v for k, v in attributions.items() if k not in ("occlusion_delta",)},
        occ_delta, output_dir / "attribution_maps.png",
    )
    _plot_faithfulness_curves(area_frac_ref, curve_data, output_dir / "faithfulness_curves.png")

    with open(output_dir / "target_info.yaml", "w") as f:
        yaml.dump({
            "char": char, "label": label_name,
            "font_family": target_meta["font_family"], "image_path": target_meta["image_path"],
            "split": target_meta["split"], "pick_index": target_meta["pick_index"],
            "s0_target": target_s0,
        }, f, allow_unicode=True, sort_keys=False)

    print(f"{char} {label_name}: agreement(spearman) vs occlusion = " +
          ", ".join(f"{r['method']}={r['spearman_rho']:.3f}" for r in agreement_rows))

    return pd.concat([agreement_df.assign(_kind="agreement"),
                      faithfulness_df.assign(_kind="faithfulness")], ignore_index=True), curves


def main() -> None:
    setup_logging()
    args = parse_args()
    cfg = load_config(args.config, args.overrides)

    backbone = cfg["model"]["backbone"]
    if backbone not in VIT_SPECS:
        raise ValueError(
            f"analyze_attention.py requires a ViT backbone (model.backbone in "
            f"{sorted(VIT_SPECS)}), got '{backbone}'. Use analyze_occlusion.py / "
            f"visualize_gradcam.py for ResNet Scorers."
        )
    patch_size = VIT_SPECS[backbone].patch_size
    image_size = cfg["data"]["image_size"]
    if image_size % patch_size != 0:
        raise ValueError(f"data.image_size={image_size} not divisible by patch_size={patch_size}")
    grid = image_size // patch_size

    chunk = args.batch_size if args.batch_size is not None else cfg["data"]["batch_size"]
    baseline = _resolve_baseline_arg(args.baseline)

    ckpt_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir) if args.output_dir else (
        ckpt_path.parent.parent / "attention_analysis"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump({**cfg, "analyze_attention": vars(args)}, f, allow_unicode=True, sort_keys=False)
    save_run_info(output_dir, extra={
        "checkpoint": args.checkpoint, "split": args.split, "use_ema": args.use_ema,
        "backbone": backbone, "patch_size": patch_size, "grid": grid,
        "baseline": args.baseline, "target": args.target,
        "grad_cam_layer": args.grad_cam_layer,
        "char": args.char, "all_chars": args.all_chars,
        "label": args.label, "n_images": args.n_images,
    })

    device = get_device()
    print(f"Device: {device}  ViT backbone={backbone}  patch={patch_size}  grid={grid}x{grid}")

    module: ScorerModule = ScorerModule.load_from_checkpoint(args.checkpoint, map_location="cpu")
    module.eval().to(device)
    if args.use_ema:
        module._ema.shadow_params = [p.to(device) for p in module._ema.shadow_params]
        module._ema.copy_to(module.scorer.parameters())
    scorer = module.scorer

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

    labels = ["real", "gen"] if args.label == "both" else [args.label]
    all_rows: list[pd.DataFrame] = []
    manifest_rows: list[dict] = []

    def run(label_name: str, char_filter: list[str] | None) -> None:
        label_int = 1 if label_name == "real" else 0
        pool = by_label[label_int]
        if char_filter is not None:
            pool = [p for p in pool if p["char"] in char_filter]
            if not pool:
                available = sorted({p["char"] for p in by_label[label_int]})
                raise ValueError(
                    f"No samples for char={char_filter} label={label_name}. Available: {available}"
                )
        picks = rng.sample(pool, min(args.n_images, len(pool)))
        for i, pick in enumerate(picks):
            image = pick["image"].to(device)
            char = pick["char"]
            image_dir = output_dir / f"{char}_{label_name}_{i}"
            print(f"Analyzing {char} ({label_name}) [{i + 1}/{len(picks)}] -> {image_dir}")
            target_meta = {
                "font_family": pick["font_family"], "image_path": pick["image_path"],
                "split": args.split, "pick_index": i,
            }
            rows, curves = analyze_one_image(
                scorer, image, char, label_name, args, baseline, patch_size, grid, chunk,
                image_dir, target_meta,
            )
            all_rows.append(rows)
            manifest_rows.append({
                "output_dir": str(image_dir.relative_to(output_dir)),
                "char": char, "label": label_name, **target_meta,
            })

    if args.all_chars:
        chars = args.char if args.char is not None else [
            chr(cp) for cp in _resolve_codepoints(cfg["data"]["charset"])
        ]
        for char in chars:
            for label_name in labels:
                run(label_name, [char])
    else:
        for label_name in labels:
            run(label_name, args.char)

    combined = pd.concat(all_rows, ignore_index=True)
    agreement_all = combined[combined["_kind"] == "agreement"].drop(columns=["_kind"])
    faithfulness_all = combined[combined["_kind"] == "faithfulness"].drop(columns=["_kind"])

    agreement_all.to_csv(output_dir / "attribution_agreement_all.csv", index=False)
    faithfulness_all.to_csv(output_dir / "faithfulness_all.csv", index=False)

    summary = (
        faithfulness_all.groupby("method")[["deletion_auc", "insertion_auc"]]
        .agg(["mean", "std"])
    )
    summary.columns = ["_".join(c) for c in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(output_dir / "faithfulness_summary.csv", index=False)
    print("\nFaithfulness summary (mean +/- std over all analyzed images):")
    print(summary.to_string(index=False))

    pd.DataFrame(manifest_rows).to_csv(output_dir / "targets_manifest.csv", index=False)
    print(f"\nSaved: {output_dir / 'targets_manifest.csv'} ({len(manifest_rows)} images)")


if __name__ == "__main__":
    main()
