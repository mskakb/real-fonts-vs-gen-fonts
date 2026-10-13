"""Generate font images with Classifier Guidance from RealnessScorer (E3 experiment).

Sweeps over realness_scale values and saves images + metadata in the same
format as generate_fonts.py so FontDataModule(mode="scorer") can read them.

Usage:
    uv run python scripts/guided_sample.py \\
        --diffusion outputs/<exp>/checkpoints/last.ckpt \\
        --scorer outputs/<exp>/checkpoints/last.ckpt

    uv run python scripts/guided_sample.py \\
        --diffusion path/to/diffusion.ckpt \\
        --scorer path/to/scorer.ckpt \\
        --config config/generate/guided_sample.yaml \\
        realness_scales=[-2.0,0.0,2.0] n_samples_per_class=10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from lightning import seed_everything

from real_fonts_vs_gen_fonts.datamodule import _resolve_codepoints
from real_fonts_vs_gen_fonts.models.diffusion_module import DiffusionModule
from real_fonts_vs_gen_fonts.models.guided_sampler import ClassifierGuidedSampler
from real_fonts_vs_gen_fonts.models.scorer_module import ScorerModule
from real_fonts_vs_gen_fonts.utils.config import load_config, save_run_info
from real_fonts_vs_gen_fonts.utils.image import tensor_to_pil
from real_fonts_vs_gen_fonts.utils.io import save_metadata_parquet
from real_fonts_vs_gen_fonts.utils.torch_utils import get_device


def parse_args() -> tuple[argparse.Namespace, argparse.ArgumentParser]:
    parser = argparse.ArgumentParser(
        description="Generate font images with Classifier Guidance"
    )
    parser.add_argument("--config", default="config/generate/guided_sample.yaml", metavar="FILE",
                        help="YAML config file (default: config/generate/guided_sample.yaml)")
    # default=None: distinguishes "not given on CLI" from an explicit value, so the
    # config value can fill in. CLI flag wins over config when provided.
    parser.add_argument("--diffusion", default=None, metavar="CKPT",
                        help="DiffusionModule checkpoint (overrides config 'diffusion')")
    parser.add_argument("--scorer", default=None, metavar="CKPT",
                        help="ScorerModule checkpoint (overrides config 'scorer')")
    parser.add_argument("overrides", nargs="*", metavar="key=value",
                        help="Config overrides, e.g. realness_scales=[0.0,2.0]")
    parser.add_argument(
        "--use-ema", dest="use_ema", action="store_true", default=None,
        help="Use EMA-averaged Scorer weights (overrides config 'scorer_use_ema')",
    )
    parser.add_argument(
        "--no-ema", dest="use_ema", action="store_false",
        help="Use raw (non-EMA) Scorer weights",
    )
    return parser.parse_args(), parser





def main() -> None:
    args, parser = parse_args()
    cfg = load_config(args.config, args.overrides)

    # Resolve CLI > config > default for values that may live in either place.
    diffusion = args.diffusion or cfg.get("diffusion")
    scorer = args.scorer or cfg.get("scorer")
    if not diffusion:
        parser.error("diffusion checkpoint must be given via --diffusion or config 'diffusion'")
    if not scorer:
        parser.error("scorer checkpoint must be given via --scorer or config 'scorer'")
    scorer_use_ema = args.use_ema if args.use_ema is not None else bool(cfg.get("scorer_use_ema", True))

    seed_everything(cfg["seed"], workers=True)
    device = get_device()

    # --- Load models ---
    diff_module = DiffusionModule.load_from_checkpoint(diffusion)
    diff_module.eval().to(device)

    scorer_module = ScorerModule.load_from_checkpoint(scorer)
    scorer_module.eval().to(device)
    if scorer_use_ema:
        scorer_module._ema.shadow_params = [p.to(device) for p in scorer_module._ema.shadow_params]
        scorer_module._ema.copy_to(scorer_module.scorer.parameters())

    # --- Output directory ---
    output_dir = Path(cfg["output_dir"])
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    cfg_to_save = dict(cfg)
    cfg_to_save["diffusion_ckpt"] = str(diffusion)
    cfg_to_save["scorer_ckpt"] = str(scorer)
    cfg_to_save["scorer_use_ema"] = scorer_use_ema
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(cfg_to_save, f, allow_unicode=True, sort_keys=False)
    save_run_info(output_dir)

    # --- Setup ---
    codepoints: list[int] = _resolve_codepoints(cfg["charset"])
    num_classes = len(codepoints)
    realness_scales: list[float] = cfg["realness_scales"]
    _st = cfg["sampling_timesteps"]
    sampling_timesteps_list: list[int] = _st if isinstance(_st, list) else [_st]
    _cs = cfg["cond_scales"]
    cond_scales: list[float] = _cs if isinstance(_cs, list) else [_cs]
    n_per_class: int = cfg["n_samples_per_class"]
    batch_size: int = cfg["batch_size"]

    sdf_sharpness: float | None = cfg.get("sdf_sharpness")
    records: list[dict] = []

    for sampling_timesteps in sampling_timesteps_list:
        diff_module.set_sampling_timesteps(sampling_timesteps)

        for cond_scale in cond_scales:
            for realness_scale in realness_scales:
                sampler = ClassifierGuidedSampler(
                    diffusion=diff_module.diffusion,
                    scorer=scorer_module.scorer,
                    realness_scale=realness_scale,
                    sampling_timesteps=sampling_timesteps,
                    use_sdf=getattr(diff_module.hparams, "use_sdf", False),
                )

                all_classes = [
                    class_idx
                    for class_idx in range(num_classes)
                    for _ in range(n_per_class)
                ]
                scale_tag = f"{realness_scale:+.1f}".replace("+", "p").replace("-", "n")

                print(f"steps={sampling_timesteps}  cs={cond_scale:.1f}  rs={realness_scale:+.1f}  ({len(all_classes)} images)")

                for batch_start in range(0, len(all_classes), batch_size):
                    batch_classes = all_classes[batch_start : batch_start + batch_size]
                    classes_t = torch.tensor(batch_classes, device=device)

                    images = sampler.sample(classes_t, cond_scale=cond_scale)
                    if getattr(diff_module.hparams, "use_sdf", False):
                        from real_fonts_vs_gen_fonts.sdf import sdf_to_binary
                        images = sdf_to_binary(images, sharpness=sdf_sharpness)

                    with torch.no_grad():
                        scores = scorer_module.scorer(images.float()).cpu().tolist()

                    for i, (class_idx, img_t, score) in enumerate(zip(batch_classes, images, scores)):
                        cp = codepoints[class_idx]
                        sample_idx = batch_start + i
                        filename = (
                            f"{cp:04X}_guided"
                            f"_st{sampling_timesteps}"
                            f"_cs{cond_scale:.1f}"
                            f"_rs{scale_tag}"
                            f"_{sample_idx:06d}.png"
                        )
                        tensor_to_pil(img_t).save(images_dir / filename)

                        records.append({
                            "file": filename,
                            "char": chr(cp),
                            "unicode": f"U+{cp:04X}",
                            "codepoint": cp,
                            "font_path": str(diffusion),
                            "font_family": "guided",
                            "font_style": f"st{sampling_timesteps}_cs{cond_scale:.1f}_rs{realness_scale:+.1f}",
                            "sampling_timesteps": sampling_timesteps,
                            "cond_scale": cond_scale,
                            "realness_scale": realness_scale,
                            "realness_score": float(score),
                        })

    meta_path = save_metadata_parquet(records, output_dir)
    print(f"Generated {len(records)} images → {output_dir}")
    print(f"metadata: {meta_path}")


if __name__ == "__main__":
    main()
