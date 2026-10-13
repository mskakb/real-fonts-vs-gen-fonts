"""Diffusion model training entry point.

Usage:
    uv run python scripts/train_diffusion.py
    uv run python scripts/train_diffusion.py --config config/diffusion/diffusion.yaml
    uv run python scripts/train_diffusion.py model.dim=128 trainer.max_steps=200000
    uv run python scripts/train_diffusion.py --resume outputs/<exp>/checkpoints/last.ckpt
    uv run python scripts/train_diffusion.py --weights-only outputs/<exp>/checkpoints/last.ckpt trainer.max_steps=300000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from real_fonts_vs_gen_fonts.datamodule import FontDataModule
from real_fonts_vs_gen_fonts.models.diffusion_module import DiffusionModule
from real_fonts_vs_gen_fonts.utils.config import load_config, setup_experiment_dir, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GaussianDiffusion model")
    parser.add_argument(
        "--config", default="config/diffusion/diffusion.yaml", metavar="FILE",
        help="YAML config file (default: config/diffusion/diffusion.yaml)",
    )
    parser.add_argument(
        "--resume", metavar="CKPT",
        help="Resume training from checkpoint (restores weights, optimizer, scheduler, global_step)",
    )
    parser.add_argument(
        "--weights-only", metavar="CKPT",
        help="Load model weights only from checkpoint (optimizer and scheduler are reset)",
    )
    parser.add_argument(
        "overrides", nargs="*", metavar="key=value",
        help="Config overrides, e.g. model.dim=128 trainer.max_steps=200000",
    )
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    if args.resume and args.weights_only:
        raise ValueError("--resume and --weights-only are mutually exclusive")
    cfg = load_config(args.config, args.overrides)
    resume_from = Path(args.resume) if args.resume else None
    weights_only_from = Path(args.weights_only) if args.weights_only else None

    seed_everything(cfg["data"]["seed"], workers=True)

    # --- 出力ディレクトリと config の保存 ---
    output_dir = setup_experiment_dir(cfg, extra={
        "resume": str(args.resume) if args.resume else None,
        "weights_only": str(args.weights_only) if args.weights_only else None,
    })

    # --- DataModule ---
    dm = FontDataModule(
        real_data_dir=cfg["data"]["real_data_dir"],
        mode="diffusion",
        charset=cfg["data"]["charset"],
        image_size=cfg["data"]["image_size"],
        image_channels=cfg["data"].get("image_channels", 3),
        batch_size=cfg["data"]["batch_size"],
        val_split=cfg["data"]["val_split"],
        test_split=cfg["data"]["test_split"],
        num_workers=cfg["data"]["num_workers"],
        seed=cfg["data"]["seed"],
        output_dir=output_dir,
        split_file=cfg["data"].get("split_file"),
        use_sdf=cfg["data"].get("use_sdf", False),
        sdf_max_dist=cfg["data"].get("sdf_max_dist", 10.0),
        sdf_source=cfg["data"].get("sdf_source", "auto"),
    )

    # --- Model ---
    # lr_total_steps は trainer.max_steps から注入（YAML に二重管理しない）
    # model.use_lr_scheduler=false を指定すると lr_total_steps=0 を渡してスケジューラを無効化
    use_lr_scheduler = cfg["model"].pop("use_lr_scheduler", True)
    lr_total_steps = cfg["trainer"]["max_steps"] if use_lr_scheduler else 0

    if weights_only_from:
        model = DiffusionModule.load_from_checkpoint(
            weights_only_from,
            num_classes=dm.num_classes,
            lr_total_steps=lr_total_steps,
            **cfg["model"],
        )
    else:
        model = DiffusionModule(
            num_classes=dm.num_classes,
            lr_total_steps=lr_total_steps,
            **cfg["model"],
        )

    # --- Callbacks ---
    # --- Trainer ---
    trainer_cfg = cfg["trainer"]

    callbacks = [
        ModelCheckpoint(
            dirpath=output_dir / "checkpoints",
            filename="step{step:06d}-loss{val/loss:.4f}",
            monitor="val/loss",
            mode="min",
            save_top_k=3,
            save_last=True,
            auto_insert_metric_name=False,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]
    save_every = trainer_cfg.get("save_every_n_steps", 0)
    if save_every > 0:
        callbacks.append(
            ModelCheckpoint(
                dirpath=output_dir / "checkpoints",
                filename="periodic_step{step:06d}",
                every_n_train_steps=save_every,
                save_top_k=trainer_cfg.get("save_periodic_top_k", -1),
                save_last=False,
                auto_insert_metric_name=False,
            )
        )
    es_cfg = trainer_cfg.get("early_stopping", {})
    if es_cfg.get("patience", 0) > 0:
        callbacks.append(
            EarlyStopping(
                monitor=es_cfg["monitor"],
                patience=es_cfg["patience"],
                mode=es_cfg["mode"],
                min_delta=es_cfg.get("min_delta", 0.0),
            )
        )

    logger = TensorBoardLogger(save_dir=output_dir / "logs", name="", version="")
    trainer = Trainer(
        max_steps=trainer_cfg["max_steps"],
        val_check_interval=trainer_cfg["val_check_interval"],
        limit_val_batches=trainer_cfg.get("limit_val_batches", 1.0),
        log_every_n_steps=trainer_cfg["log_every_n_steps"],
        precision=trainer_cfg["precision"],
        gradient_clip_val=trainer_cfg["gradient_clip_val"],
        callbacks=callbacks,
        logger=logger,
        default_root_dir=str(output_dir),
    )

    # --resume: 重み・オプティマイザ・スケジューラ・global_step をすべて復元
    # --weights-only: 重みのみ復元済み（model生成時）、スケジューラは新規
    trainer.fit(model, dm, ckpt_path=resume_from)

    best = trainer.checkpoint_callback.best_model_path
    if best:
        (output_dir / "checkpoints" / "best_model_path.txt").write_text(best)


if __name__ == "__main__":
    main()
