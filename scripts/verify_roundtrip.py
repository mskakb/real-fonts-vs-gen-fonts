"""Forward→reverse diffusion roundtrip verification for real font images.

For a batch of real images x_0, forward-diffuses analytically to a noise level N
(q_sample) then reverse-diffuses back to x_0 with DDIM using M steps (independent
of N). Measures whether the roundtrip alone introduces artifacts and how it shifts
the Scorer's realness judgement, across a sweep of (N, M) combinations.

Usage:
    uv run python scripts/verify_roundtrip.py \\
        --config config/eval/verify_roundtrip.yaml \\
        diffusion_ckpt=outputs/<exp>/checkpoints/last.ckpt \\
        scorer_ckpt=outputs/<exp>/checkpoints/last.ckpt \\
        real_data_dir=/path/to/real \\
        output_dir=outputs/verify_roundtrip
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import pandas as pd
import torch
import yaml
from lightning import seed_everything
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve

from real_fonts_vs_gen_fonts.datamodule import FontDataModule
from real_fonts_vs_gen_fonts.models.diffusion import normalize_to_neg_one_to_one, unnormalize_to_zero_to_one
from real_fonts_vs_gen_fonts.models.diffusion_module import DiffusionModule
from real_fonts_vs_gen_fonts.models.scorer_module import ScorerModule
from real_fonts_vs_gen_fonts.utils.config import load_config, save_run_info
from real_fonts_vs_gen_fonts.utils.image import tensor_to_pil
from real_fonts_vs_gen_fonts.utils.torch_utils import get_device

_COLORS = ["steelblue", "tomato", "seagreen", "darkorange", "mediumpurple", "brown"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Forward-reverse diffusion roundtrip verification")
    parser.add_argument("--config", required=True, metavar="FILE",
                        help="YAML config file (see config/eval/verify_roundtrip.yaml)")
    parser.add_argument("overrides", nargs="*", metavar="key=value",
                        help="Config overrides, e.g. noise_steps=[100,500] ddim_reverse_steps=[20]")
    return parser.parse_args()


def _score(scorer_module: ScorerModule, images: torch.Tensor) -> torch.Tensor:
    """Scorer の backbone は 3ch 入力前提のため、グレースケール(1ch)なら複製する。"""
    if images.shape[1] == 1:
        images = images.repeat(1, 3, 1, 1)
    return scorer_module.scorer(images.float()).detach().cpu()


def _decode_sdf(images: torch.Tensor, use_sdf: bool, sdf_decode: str,
                 sdf_sharpness: float | None, sdf_filter_width_px: float,
                 sdf_max_dist: float) -> torch.Tensor:
    if not use_sdf:
        return images
    if sdf_decode == "smoothstep":
        from real_fonts_vs_gen_fonts.sdf import sdf_to_smoothstep
        return sdf_to_smoothstep(images, filter_width_px=sdf_filter_width_px, max_dist=sdf_max_dist)
    from real_fonts_vs_gen_fonts.sdf import sdf_to_binary
    return sdf_to_binary(images, sharpness=sdf_sharpness if sdf_decode == "sigmoid" else None)


def _save_grid(originals: torch.Tensor, noisy: torch.Tensor, roundtrips: torch.Tensor, path: Path) -> None:
    """各行 = 1 サンプル、列 = [original, noisy x_N, roundtrip] のグリッド画像を保存する。

    noisy は q_sample 直後・SDF decode 前の生テンソル（[-1,1] 正規化空間）。SDF decode は
    有効な SDF 場を前提とする処理でノイズが乗った x_N には意味を成さないため、正規化解除
    （unnormalize_to_zero_to_one）のみで可視化する。
    """
    n = originals.shape[0]
    cell = 96
    noisy_01 = unnormalize_to_zero_to_one(noisy).clamp(0, 1)
    grid = Image.new("L", (cell * 3, cell * n), color=255)
    for i in range(n):
        grid.paste(tensor_to_pil(originals[i]).convert("L").resize((cell, cell)), (0, cell * i))
        grid.paste(tensor_to_pil(noisy_01[i]).convert("L").resize((cell, cell)), (cell, cell * i))
        grid.paste(tensor_to_pil(roundtrips[i]).convert("L").resize((cell, cell)), (cell * 2, cell * i))
    path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(path)


def main() -> None:
    raw = parse_args()
    cfg = load_config(raw.config, raw.overrides)
    args = SimpleNamespace(**cfg)

    seed_everything(args.seed, workers=True)
    device = get_device()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f, allow_unicode=True, sort_keys=False)
    save_run_info(output_dir)

    # --- モデルのロード ---
    diff_module = DiffusionModule.load_from_checkpoint(args.diffusion_ckpt, strict=False)
    diff_module.eval().to(device)
    if args.use_ema:
        diff_module._ema.shadow_params = [p.to(device) for p in diff_module._ema.shadow_params]
        diff_module._ema.copy_to(diff_module.diffusion.parameters())
    diffusion = diff_module.diffusion

    scorer_module: ScorerModule | None = None
    if args.scorer_ckpt:
        scorer_module = ScorerModule.load_from_checkpoint(args.scorer_ckpt)
        scorer_module.eval().to(device)
        if args.scorer_use_ema:
            scorer_module._ema.shadow_params = [p.to(device) for p in scorer_module._ema.shadow_params]
            scorer_module._ema.copy_to(scorer_module.scorer.parameters())

    use_sdf = bool(getattr(diff_module.hparams, "use_sdf", False))
    sdf_max_dist = float(getattr(diff_module.hparams, "sdf_max_dist", 10.0))
    image_channels = int(getattr(diff_module.hparams, "channels", 3))

    # chunk_size: 1 回の GPU forward に載せる枚数の上限。n_samples をこれより大きくしても
    # チャンク単位で逐次処理するため VRAM 使用量は n_samples に依存しない（各サンプル独立なので
    # チャンク分割しても数値的に等価。DiffusionModule._sample_in_chunks と同じ考え方）。
    chunk_size = int(getattr(args, "chunk_size", 32))

    # --- 実画像のロード（chunk_size 単位でバッチ取得し、CPU 上に集約する）---
    dm = FontDataModule(
        real_data_dir=args.real_data_dir,
        mode="diffusion",
        charset=args.charset,
        image_size=diffusion.image_size,
        image_channels=image_channels,
        batch_size=min(args.n_samples, chunk_size),
        split_file=args.split_file,
        seed=args.seed,
        use_sdf=use_sdf,
        sdf_max_dist=sdf_max_dist,
    )
    dm.setup(stage=None)
    dataloader = {"train": dm.train_dataloader, "val": dm.val_dataloader, "test": dm.test_dataloader}[args.split]()

    images_chunks, label_chunks, chars, font_families = [], [], [], []
    n_collected = 0
    for batch in dataloader:
        take = min(args.n_samples - n_collected, batch["image"].shape[0])
        images_chunks.append(batch["image"][:take])
        label_chunks.append(batch["label"][:take])
        chars.extend(str(c) for c in batch["char"][:take])
        font_families.extend(str(f) for f in batch["font_family"][:take])
        n_collected += take
        if n_collected >= args.n_samples:
            break
    images_01 = torch.cat(images_chunks, dim=0)   # CPU, [0,1]
    classes_all = torch.cat(label_chunks, dim=0)  # CPU
    n = images_01.shape[0]

    # --- x0 の decode・スコア（chunk_size 単位で GPU 推論し、結果は CPU に集約する）---
    x0_decoded_parts, s0_parts = [], []
    for lo in range(0, n, chunk_size):
        hi = min(lo + chunk_size, n)
        dec_chunk = _decode_sdf(images_01[lo:hi].to(device), use_sdf, args.sdf_decode,
                                 args.sdf_sharpness, args.sdf_filter_width_px, sdf_max_dist)
        x0_decoded_parts.append(dec_chunk.cpu())
        if scorer_module is not None:
            s0_parts.append(_score(scorer_module, dec_chunk))
    x0_decoded = torch.cat(x0_decoded_parts, dim=0)
    s0 = torch.cat(s0_parts, dim=0) if scorer_module is not None else torch.full((n,), float("nan"))

    # --- noise_schedule.png（データ非依存、alphas_cumprod から解析的に1回だけ算出）---
    n_grid = min(args.n_grid_images, n)
    alphas_cumprod = diffusion.alphas_cumprod.detach().cpu()
    signal_coef = alphas_cumprod.sqrt()
    noise_coef = (1 - alphas_cumprod).sqrt()
    fig, ax = plt.subplots(figsize=(9, 5))
    ts = torch.arange(alphas_cumprod.shape[0])
    ax.plot(ts, signal_coef, color="steelblue", label=r"signal coef. $\sqrt{\bar\alpha_t}$")
    ax.plot(ts, noise_coef, color="tomato", label=r"noise coef. $\sqrt{1-\bar\alpha_t}$")
    for N in args.noise_steps:
        ax.axvline(N, color="gray", linestyle="--", linewidth=0.8)
        ax.text(N, 1.02, f"N={N}", rotation=90, fontsize=7, ha="right", va="bottom", color="gray")
    ax.set_xlabel("t (forward diffusion timestep)")
    ax.set_ylabel("coefficient")
    ax.set_ylim(0.0, 1.08)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "noise_schedule.png", dpi=150)
    plt.close(fig)

    rows: list[dict] = []
    curve: dict[tuple[int, int], dict[str, float]] = {}
    roc_curves: dict[tuple[int, int], tuple] = {}

    for N in args.noise_steps:
        x_noisy_head_parts = []
        for M in args.ddim_reverse_steps:
            x_rt_decoded_parts, s_rt_parts, mse_parts, pixel_diff_parts = [], [], [], []
            for lo in range(0, n, chunk_size):
                hi = min(lo + chunk_size, n)
                img_chunk = images_01[lo:hi].to(device)
                classes_chunk = classes_all[lo:hi].to(device)
                x0_chunk = normalize_to_neg_one_to_one(img_chunk)
                t = torch.full((hi - lo,), N, device=device, dtype=torch.long)
                xt_chunk = diffusion.q_sample(x0_chunk, t=t)
                if M == args.ddim_reverse_steps[0] and lo < n_grid:
                    x_noisy_head_parts.append(xt_chunk[: n_grid - lo].cpu())
                x_rt_chunk = diffusion.ddim_sample_from(
                    xt_chunk, classes_chunk, t_start=N, sampling_timesteps=M,
                    cond_scale=args.cond_scale, rescaled_phi=args.rescaled_phi,
                )
                x_rt_dec_chunk = _decode_sdf(x_rt_chunk, use_sdf, args.sdf_decode, args.sdf_sharpness,
                                              args.sdf_filter_width_px, sdf_max_dist)
                x0_dec_chunk = x0_decoded[lo:hi].to(device)

                if scorer_module is not None:
                    s_rt_parts.append(_score(scorer_module, x_rt_dec_chunk))
                mse_parts.append((x0_dec_chunk - x_rt_dec_chunk).pow(2).mean(dim=(1, 2, 3)).cpu())
                pixel_diff_parts.append(
                    ((x0_dec_chunk > 0.5) != (x_rt_dec_chunk > 0.5)).float().mean(dim=(1, 2, 3)).cpu()
                )
                x_rt_decoded_parts.append(x_rt_dec_chunk.cpu())

            if M == args.ddim_reverse_steps[0]:
                x_noisy_head = torch.cat(x_noisy_head_parts, dim=0)

            x_rt_decoded = torch.cat(x_rt_decoded_parts, dim=0)
            s_rt = torch.cat(s_rt_parts, dim=0) if scorer_module is not None else torch.full((n,), float("nan"))
            mse = torch.cat(mse_parts, dim=0)
            pixel_diff = torch.cat(pixel_diff_parts, dim=0)

            for i in range(n):
                rows.append({
                    "char": chars[i],
                    "font_family": font_families[i],
                    "N": N,
                    "M": M,
                    "s0": float(s0[i]),
                    "s_rt": float(s_rt[i]),
                    "score_delta": float(s_rt[i] - s0[i]),
                    "mse": float(mse[i]),
                    "pixel_diff_frac": float(pixel_diff[i]),
                })

            # 元画像(s0, label=1) vs roundtrip 画像(s_rt, label=0) を Scorer のスコアだけで
            # どこまで分離できるか。1.0 に近いほど「roundtrip だけで real/fake 判定が変わる」ことを意味する。
            if scorer_module is not None:
                y_true = [1] * n + [0] * n
                y_score = s0.tolist() + s_rt.tolist()
                auroc = float(roc_auc_score(y_true, y_score))
                fpr, tpr, _ = roc_curve(y_true, y_score)
                roc_curves[(N, M)] = (fpr, tpr)
            else:
                auroc = float("nan")

            curve[(N, M)] = {
                "s0_mean": float(s0.mean()), "s0_std": float(s0.std()),
                "s_rt_mean": float(s_rt.mean()), "s_rt_std": float(s_rt.std()),
                "mse_mean": float(mse.mean()), "pixel_diff_mean": float(pixel_diff.mean()),
                "auroc": auroc,
            }

            _save_grid(
                x0_decoded[:n_grid], x_noisy_head, x_rt_decoded[:n_grid],
                output_dir / "grids" / f"N{N}_M{M}_grid.png",
            )
            print(f"N={N} M={M}: s0={s0.mean():.4f} s_rt={s_rt.mean():.4f} auroc={auroc:.4f} "
                  f"mse={mse.mean():.5f} pixel_diff={pixel_diff.mean():.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "roundtrip_scores.csv", index=False)

    if scorer_module is not None:
        auroc_df = pd.DataFrame([
            {"N": N, "M": M, "auroc": curve[(N, M)]["auroc"], "n": n}
            for N in args.noise_steps for M in args.ddim_reverse_steps
        ])
        auroc_df.to_csv(output_dir / "roundtrip_auroc.csv", index=False)

    # --- score_vs_noise.png ---
    if scorer_module is not None:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.axhline(float(s0.mean()), color="gray", linestyle="--", linewidth=0.8, label="s0 (original) mean")
        for idx, M in enumerate(args.ddim_reverse_steps):
            ns = args.noise_steps
            means = [curve[(N, M)]["s_rt_mean"] for N in ns]
            stds = [curve[(N, M)]["s_rt_std"] for N in ns]
            color = _COLORS[idx % len(_COLORS)]
            ax.plot(ns, means, color=color, marker="o", label=f"s_rt (M={M})")
            ax.fill_between(ns, [m - s for m, s in zip(means, stds)],
                             [m + s for m, s in zip(means, stds)], alpha=0.15, color=color)
        ax.set_xlabel("N (forward diffusion noise level)")
        ax.set_ylabel("Realness score")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / "score_vs_noise.png", dpi=150)
        plt.close(fig)

    # --- auroc_vs_noise.png ---
    if scorer_module is not None:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="auroc = 0.5 (indistinguishable)")
        for idx, M in enumerate(args.ddim_reverse_steps):
            ns = args.noise_steps
            aurocs = [curve[(N, M)]["auroc"] for N in ns]
            ax.plot(ns, aurocs, color=_COLORS[idx % len(_COLORS)], marker="o", label=f"M={M}")
        ax.set_xlabel("N (forward diffusion noise level)")
        ax.set_ylabel("AUROC (s0 vs s_rt)")
        ax.set_ylim(0.0, 1.05)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / "auroc_vs_noise.png", dpi=150)
        plt.close(fig)

    # --- roc_curves_M{M}.png（M ごとに 1 枚、N を重ね書き）---
    if scorer_module is not None:
        for M in args.ddim_reverse_steps:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=0.8, label="auroc = 0.5")
            for idx, N in enumerate(args.noise_steps):
                fpr, tpr = roc_curves[(N, M)]
                auroc = curve[(N, M)]["auroc"]
                ax.plot(fpr, tpr, color=_COLORS[idx % len(_COLORS)],
                        label=f"N={N} (auroc={auroc:.3f})")
            ax.set_xlabel("FPR")
            ax.set_ylabel("TPR")
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.05)
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(output_dir / f"roc_curves_M{M}.png", dpi=150)
            plt.close(fig)

    # --- reconstruction_error_vs_noise.png ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for idx, M in enumerate(args.ddim_reverse_steps):
        ns = args.noise_steps
        color = _COLORS[idx % len(_COLORS)]
        axes[0].plot(ns, [curve[(N, M)]["mse_mean"] for N in ns], color=color, marker="o", label=f"M={M}")
        axes[1].plot(ns, [curve[(N, M)]["pixel_diff_mean"] for N in ns], color=color, marker="o", label=f"M={M}")
    axes[0].set_xlabel("N")
    axes[0].set_ylabel("MSE (decoded)")
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("N")
    axes[1].set_ylabel("Pixel disagreement fraction")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "reconstruction_error_vs_noise.png", dpi=150)
    plt.close(fig)

    print(f"\nSaved: {output_dir}")


if __name__ == "__main__":
    main()
