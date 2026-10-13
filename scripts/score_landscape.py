"""Visualize s(x̂_0) vs. timestep t during DDIM sampling (E2 experiment).

Runs DDIM sampling without guidance and records the Scorer output s(x̂_0)
at every denoising step. Supports multiple cond_scale and sampling_timesteps
values; all combinations are plotted on a single figure.

Usage:
    uv run python scripts/score_landscape.py \\
        --diffusion outputs/<exp>/checkpoints/last.ckpt \\
        --scorer outputs/<exp>/checkpoints/last.ckpt

    uv run python scripts/score_landscape.py \\
        --diffusion path/to/diffusion.ckpt \\
        --scorer path/to/scorer.ckpt \\
        --cond-scale 1.0 3.0 6.0 \\
        --sampling-timesteps 50 1000 \\
        --n-samples 32 \\
        --output-dir outputs/score_landscape
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import torch
import yaml
from lightning import seed_everything

from real_fonts_vs_gen_fonts.datamodule import _resolve_codepoints
from real_fonts_vs_gen_fonts.models.diffusion_module import DiffusionModule
from real_fonts_vs_gen_fonts.models.guided_sampler import ClassifierGuidedSampler
from real_fonts_vs_gen_fonts.models.scorer_module import ScorerModule
from real_fonts_vs_gen_fonts.utils.config import load_config, save_run_info

# 組み合わせ数に応じて色・線種を自動割り当て
_COLORS = ["steelblue", "tomato", "seagreen", "darkorange", "mediumpurple", "brown"]
_LINESTYLES = ["-", "--", "-.", ":"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot s(x̂_0) vs. timestep t")
    parser.add_argument("--config", required=True, metavar="FILE",
                        help="YAML config file (see config/eval/score_landscape.yaml)")
    parser.add_argument("overrides", nargs="*", metavar="key=value",
                        help="Config overrides, e.g. n_samples=32 cond_scales=[1.0,6.0]")
    return parser.parse_args()


def main() -> None:
    raw = parse_args()
    cfg = load_config(raw.config, raw.overrides)
    # SimpleNamespace adapter: keep the body referring to args.* unchanged.
    args = SimpleNamespace(
        diffusion=cfg["diffusion_ckpt"],
        scorer=cfg["scorer_ckpt"],
        n_samples=cfg["n_samples"],
        cond_scale=cfg["cond_scales"],
        sampling_timesteps=cfg["sampling_timesteps"],
        charset=cfg["charset"],
        output_dir=cfg["output_dir"],
        seed=cfg["seed"],
        use_ema=cfg["use_ema"],
    )
    seed_everything(args.seed, workers=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f, allow_unicode=True, sort_keys=False)
    save_run_info(output_dir)

    # --- Load models ---
    diff_module = DiffusionModule.load_from_checkpoint(args.diffusion)
    diff_module.eval().to(device)

    scorer_module = ScorerModule.load_from_checkpoint(args.scorer)
    scorer_module.eval().to(device)
    if args.use_ema:
        scorer_module._ema.shadow_params = [p.to(device) for p in scorer_module._ema.shadow_params]
        scorer_module._ema.copy_to(scorer_module.scorer.parameters())

    # --- Setup ---
    codepoints = _resolve_codepoints(args.charset)
    num_classes = len(codepoints)
    classes = torch.tensor(
        [i % num_classes for i in range(args.n_samples)], device=device
    )

    combinations = list(itertools.product(args.sampling_timesteps, args.cond_scale))
    n_st = len(args.sampling_timesteps)
    n_cs = len(args.cond_scale)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="s = 0.5")

    all_results: list[dict] = []

    for idx, (sampling_timesteps, cond_scale) in enumerate(combinations):
        diff_module.set_sampling_timesteps(sampling_timesteps)

        sampler = ClassifierGuidedSampler(
            diffusion=diff_module.diffusion,
            scorer=scorer_module.scorer,
            realness_scale=0.0,
            sampling_timesteps=sampling_timesteps,
            use_sdf=getattr(diff_module.hparams, "use_sdf", False),
        )

        print(f"[{idx+1}/{len(combinations)}] steps={sampling_timesteps}, cond_scale={cond_scale}")
        timesteps, scores = sampler.score_landscape(classes, cond_scale=cond_scale)

        # 色: cond_scale 軸、線種: sampling_timesteps 軸
        color = _COLORS[args.cond_scale.index(cond_scale) % len(_COLORS)]
        ls = _LINESTYLES[args.sampling_timesteps.index(sampling_timesteps) % len(_LINESTYLES)]

        t_arr = list(timesteps)
        mean = scores.mean(dim=1).numpy()
        std = scores.std(dim=1).numpy()

        label = f"steps={sampling_timesteps}, cs={cond_scale:.1f}"
        ax.plot(t_arr, mean, color=color, linestyle=ls, label=label)
        ax.fill_between(t_arr, mean - std, mean + std, alpha=0.15, color=color)

        # 個別の .pt を保存
        tag = f"st{sampling_timesteps}_cs{cond_scale:.1f}"
        torch.save(
            {"timesteps": timesteps, "scores": scores, "tag": tag},
            output_dir / f"scores_{tag}.pt",
        )
        all_results.append({
            "tag": tag,
            "sampling_timesteps": sampling_timesteps,
            "cond_scale": cond_scale,
            "mean_final": float(scores[-1].mean()),
            "mean_initial": float(scores[0].mean()),
        })

        print(f"  t_high={t_arr[0]}: mean={scores[0].mean():.4f} | "
              f"t_low={t_arr[-1]}: mean={scores[-1].mean():.4f}")

    ax.set_xlabel("Timestep t  (high noise → low noise)")
    ax.set_ylabel("Realness score s(x̂_0)")
    title = f"Score landscape  n={args.n_samples}"
    if n_st == 1:
        title += f"  steps={args.sampling_timesteps[0]}"
    if n_cs == 1:
        title += f"  cs={args.cond_scale[0]:.1f}"
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.invert_xaxis()
    fig.tight_layout()

    plot_path = output_dir / "score_landscape.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    print(f"\nPlot saved: {plot_path}")
    for r in all_results:
        print(f"  {r['tag']}: initial={r['mean_initial']:.4f} → final={r['mean_final']:.4f}")


if __name__ == "__main__":
    main()
