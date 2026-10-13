"""Generate font images from a trained DiffusionModule checkpoint.

Saves images and metadata.parquet to output_dir in the same format
as font2dataset output, so FontDataModule(mode="scorer") can read it directly.

Usage:
    uv run python scripts/generate_fonts.py --checkpoint outputs/<exp>/checkpoints/last.ckpt
    uv run python scripts/generate_fonts.py --config config/generate/generate_fonts.yaml \\
        --checkpoint outputs/<exp>/checkpoints/last.ckpt
    uv run python scripts/generate_fonts.py --checkpoint path/to/last.ckpt \\
        n_samples_per_class=20 cond_scales=[1.0,6.0]
    uv run python scripts/generate_fonts.py --checkpoint last.ckpt --no-ema
"""

from __future__ import annotations

import argparse
import contextlib
import warnings
from pathlib import Path

import torch
import yaml
from lightning import seed_everything
from tqdm import tqdm

from real_fonts_vs_gen_fonts.datamodule import _resolve_codepoints
from real_fonts_vs_gen_fonts.models.diffusion_module import DiffusionModule
from real_fonts_vs_gen_fonts.models.scorer_module import ScorerModule
from real_fonts_vs_gen_fonts.utils.config import load_config, save_run_info
from real_fonts_vs_gen_fonts.utils.image import tensor_to_pil
from real_fonts_vs_gen_fonts.utils.io import save_metadata_parquet
from real_fonts_vs_gen_fonts.utils.torch_utils import get_device


def parse_args() -> tuple[argparse.Namespace, argparse.ArgumentParser]:
    parser = argparse.ArgumentParser(description="Generate font images from a trained DiffusionModule")
    parser.add_argument(
        "--config", default="config/generate/generate_fonts.yaml", metavar="FILE",
        help="YAML config file (default: config/generate/generate_fonts.yaml)",
    )
    # default=None: distinguishes "not given on CLI" from an explicit value, so the
    # config value can fill in. CLI flag wins over config when provided.
    parser.add_argument(
        "--checkpoint", default=None, metavar="CKPT",
        help="Path to trained DiffusionModule checkpoint (overrides config 'checkpoint')",
    )
    parser.add_argument(
        "--scorer", default=None, metavar="CKPT",
        help="(optional) ScorerModule checkpoint to compute realness_score (overrides config 'scorer')",
    )
    parser.add_argument(
        "--use-ema", dest="use_ema", action="store_true", default=None,
        help="Use EMA parameters for sampling (overrides config 'use_ema')",
    )
    parser.add_argument(
        "--no-ema", dest="use_ema", action="store_false",
        help="Use raw model parameters instead of EMA",
    )
    parser.add_argument(
        "--scorer-use-ema", dest="scorer_use_ema", action="store_true", default=None,
        help="Use EMA-averaged Scorer weights (overrides config 'scorer_use_ema')",
    )
    parser.add_argument(
        "--scorer-no-ema", dest="scorer_use_ema", action="store_false",
        help="Use raw (non-EMA) Scorer weights",
    )
    parser.add_argument(
        "overrides", nargs="*", metavar="key=value",
        help="Config overrides, e.g. n_samples_per_class=20 cond_scales=[1.0,6.0]",
    )
    return parser.parse_args(), parser


def main() -> None:
    args, parser = parse_args()
    cfg = load_config(args.config, args.overrides)

    # Resolve CLI > config > default for values that may live in either place.
    checkpoint = args.checkpoint or cfg.get("checkpoint")
    if not checkpoint:
        parser.error("checkpoint must be given via --checkpoint or config 'checkpoint'")
    scorer = args.scorer or cfg.get("scorer")
    use_ema = args.use_ema if args.use_ema is not None else bool(cfg.get("use_ema", True))
    scorer_use_ema = (
        args.scorer_use_ema if args.scorer_use_ema is not None
        else bool(cfg.get("scorer_use_ema", True))
    )

    seed_everything(cfg["seed"], workers=True)
    device = get_device()

    # --- モデルのロード（sampling_timesteps は後でループ内で切り替える）---
    ckpt_path = Path(checkpoint)
    # strict=False: FID 計算中に保存された _inception.* キーを持つ旧チェックポイントとの互換性維持
    module = DiffusionModule.load_from_checkpoint(ckpt_path, strict=False)
    module.eval().to(device)

    scorer_module: ScorerModule | None = None
    if scorer:
        scorer_module = ScorerModule.load_from_checkpoint(scorer)
        scorer_module.eval().to(device)
        if scorer_use_ema:
            scorer_module._ema.shadow_params = [p.to(device) for p in scorer_module._ema.shadow_params]
            scorer_module._ema.copy_to(scorer_module.scorer.parameters())

    binarize_threshold: float | None = cfg.get("binarize_threshold")
    sdf_decode: str = cfg.get("sdf_decode", "hard")
    sdf_sharpness: float | None = cfg.get("sdf_sharpness")
    sdf_filter_width_px: float = float(cfg.get("sdf_filter_width_px", 1.0))
    min_score_threshold: float | None = cfg.get("min_score_threshold")
    save_sdf: bool = bool(cfg.get("save_sdf", False))

    if min_score_threshold is not None and scorer_module is None:
        warnings.warn(
            "min_score_threshold is set but no scorer is specified. Score filtering will be skipped.",
            stacklevel=1,
        )
        min_score_threshold = None

    model_use_sdf: bool = bool(getattr(module.hparams, "use_sdf", False))
    if save_sdf and not model_use_sdf:
        warnings.warn(
            "save_sdf is set but the model was trained with use_sdf=false. "
            "There is no SDF representation to save; skipping SDF output.",
            stacklevel=1,
        )
        save_sdf = False

    # --- 出力ディレクトリの準備と config の保存 ---
    output_dir = Path(cfg["output_dir"])
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    sdf_dir = output_dir / "sdf"
    if save_sdf:
        sdf_dir.mkdir(parents=True, exist_ok=True)
    cfg_to_save = dict(cfg)
    cfg_to_save["checkpoint"] = str(ckpt_path)
    cfg_to_save["scorer"] = str(scorer) if scorer else None
    cfg_to_save["use_ema"] = use_ema
    cfg_to_save["scorer_use_ema"] = scorer_use_ema if scorer else None
    # Record SDF provenance so downstream precomputed-SDF readers know the normalization.
    cfg_to_save["use_sdf"] = model_use_sdf
    if model_use_sdf:
        cfg_to_save["sdf_max_dist"] = float(getattr(module.hparams, "sdf_max_dist", 10.0))
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(cfg_to_save, f, allow_unicode=True, sort_keys=False)
    save_run_info(output_dir)

    # --- charset からクラス → codepoint のマッピングを作成 ---
    codepoints: list[int] = _resolve_codepoints(cfg["charset"])

    cond_scales: list[float] = cfg["cond_scales"]
    # int 単体で指定された場合もリストとして扱う
    _st = cfg["sampling_timesteps"]
    sampling_timesteps_list: list[int] = _st if isinstance(_st, list) else [_st]
    n_per_class: int = cfg["n_samples_per_class"]
    batch_size: int = cfg["batch_size"]
    num_classes = len(codepoints)

    ema_suffix = "_ema" if use_ema else ""
    records: list[dict] = []

    for sampling_timesteps in sampling_timesteps_list:
        module.set_sampling_timesteps(sampling_timesteps)

        for cond_scale in cond_scales:
            all_classes = [
                class_idx
                for class_idx in range(num_classes)
                for _ in range(n_per_class)
            ]

            pbar = tqdm(
                total=len(all_classes),
                desc=f"steps={sampling_timesteps} cond_scale={cond_scale:.1f}",
            )

            for batch_start in range(0, len(all_classes), batch_size):
                batch_classes = all_classes[batch_start : batch_start + batch_size]
                classes_t = torch.tensor(batch_classes, device=device)

                ema_ctx = module._ema.average_parameters() if use_ema else contextlib.nullcontext()
                with ema_ctx, torch.no_grad():
                    images = module.diffusion.sample(
                        classes_t, cond_scale=cond_scale
                    )
                # Raw SDF before decode (decode below overwrites `images`).
                sdf_raw = images if save_sdf else None
                if getattr(module.hparams, "use_sdf", False):
                    sdf_max_dist = float(getattr(module.hparams, "sdf_max_dist", 10.0))
                    if sdf_decode == "smoothstep":
                        from real_fonts_vs_gen_fonts.sdf import sdf_to_smoothstep
                        images = sdf_to_smoothstep(images, filter_width_px=sdf_filter_width_px, max_dist=sdf_max_dist)
                    else:
                        from real_fonts_vs_gen_fonts.sdf import sdf_to_binary
                        images = sdf_to_binary(images, sharpness=sdf_sharpness if sdf_decode == "sigmoid" else None)

                scores = (
                    scorer_module.scorer(images.float()).cpu().tolist()
                    if scorer_module else [None] * len(batch_classes)
                )

                for i, (class_idx, img_t, score) in enumerate(zip(batch_classes, images, scores)):
                    if min_score_threshold is not None and score is not None and score < min_score_threshold:
                        continue

                    if binarize_threshold is not None:
                        img_t = (img_t > binarize_threshold).float()

                    cp = codepoints[class_idx]
                    sample_idx = batch_start + i
                    filename = (
                        f"{cp:04X}_generated"
                        f"_st{sampling_timesteps}"
                        f"_cs{cond_scale:.1f}"
                        f"{ema_suffix}"
                        f"_{sample_idx:06d}.png"
                    )
                    tensor_to_pil(img_t).save(images_dir / filename)

                    record = {
                        "file": filename,
                        "char": chr(cp),
                        "unicode": f"U+{cp:04X}",
                        "codepoint": cp,
                        "font_path": str(ckpt_path),
                        "font_family": "generated",
                        "font_style": f"st{sampling_timesteps}_cs{cond_scale:.1f}",
                        "sampling_timesteps": sampling_timesteps,
                        "cond_scale": cond_scale,
                        "use_ema": use_ema,
                        "realness_score": float(score) if score is not None else None,
                    }

                    if save_sdf:
                        # Raw SDF in [0, 1]; same filename as images/ (mirrors real dataset layout).
                        tensor_to_pil(sdf_raw[i]).save(sdf_dir / filename)
                        record["sdf_png_file"] = filename

                    records.append(record)

                pbar.update(len(batch_classes))

            pbar.close()

    # --- metadata.parquet の保存 ---
    meta_path = save_metadata_parquet(records, output_dir)
    print(f"Generated {len(records)} images → {output_dir}")
    print(f"metadata: {meta_path}")


if __name__ == "__main__":
    main()
