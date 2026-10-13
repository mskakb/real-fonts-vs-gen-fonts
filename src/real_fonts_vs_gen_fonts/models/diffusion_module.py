from __future__ import annotations

import math
from pathlib import Path

import torch
from lightning import LightningModule
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch_ema import ExponentialMovingAverage
from torchvision.utils import make_grid, save_image

from .diffusion import GaussianDiffusion, Unet

# t を 3 段階のノイズレベルに分類するバケット名
_NOISE_BUCKETS: list[tuple[str, float, float]] = [
    ("low",  0.0, 1/3),
    ("mid",  1/3, 2/3),
    ("high", 2/3, 1.0),
]


class DiffusionModule(LightningModule):
    """GaussianDiffusion の LightningModule ラッパー。

    Args:
        # --- Unet ---
        dim:               ベースチャネル数。
        num_classes:       条件クラス数 (FontDataModule.num_classes と合わせる)。
        cond_drop_prob:    CFG 用クラス埋め込みドロップアウト率。
        dim_mults:         各スケールのチャネル倍率。
        channels:          入力画像のチャネル数。
        attn_dim_head:     Attention のヘッドあたり次元数。
        attn_heads:        Attention のヘッド数。
        # --- GaussianDiffusion ---
        image_size:        正方形画像の一辺 (px)。
        timesteps:         学習時の拡散ステップ数 T。
        sampling_timesteps: サンプリング時のステップ数 (< T で DDIM)。
        objective:         'pred_noise' | 'pred_x0' | 'pred_v'。
        beta_schedule:     'cosine' | 'linear'。
        offset_noise_strength: オフセットノイズ強度。
        min_snr_loss_weight: Min-SNR 損失重み付けを使うか。
        use_cfg_plus_plus:  CFG++ を使うか。
        # --- optimizer / scheduler ---
        lr:                ピーク学習率。
        weight_decay:      AdamW の weight decay。
        lr_warmup_steps:   linear warmup のステップ数。0 で warmup なし。
        lr_total_steps:    スケジューラの総ステップ数。Trainer.max_steps と合わせる。
    """

    def __init__(
        self,
        # Unet
        dim: int = 64,
        num_classes: int = 26,
        cond_drop_prob: float = 0.5,
        dim_mults: tuple[int, ...] = (1, 2, 4, 8),
        channels: int = 3,
        attn_dim_head: int = 32,
        attn_heads: int = 4,
        # GaussianDiffusion
        image_size: int = 64,
        timesteps: int = 1000,
        sampling_timesteps: int | None = None,
        objective: str = "pred_noise",
        beta_schedule: str = "cosine",
        offset_noise_strength: float = 0.0,
        min_snr_loss_weight: bool = False,
        use_cfg_plus_plus: bool = False,
        # optimizer / scheduler
        lr: float = 1e-4,
        weight_decay: float = 1e-2,
        lr_warmup_steps: int = 1000,
        lr_total_steps: int = 100_000,
        # sampling
        sample_every_n_val_checks: int = 1,
        sample_cond_scale: float = 6.0,
        sample_n_images: int = 16,
        # EMA
        ema_decay: float = 0.9999,
        ema_start_step: int = 0,
        # FID
        fid_every_n_val_checks: int = 0,
        fid_n_samples: int = 256,
        fid_splits: tuple[str, ...] = ("val",),
        # Perceptual loss on x̂_0
        perceptual_weight: float = 0.0,
        perceptual_network: str = "lpips",
        perceptual_t_max_frac: float = 0.5,
        # Stroke-weighted loss
        stroke_weight_alpha: float = 0.0,
        normalize_loss_per_image: bool = False,
        # SDF representation
        use_sdf: bool = False,
        sdf_decode: str = "hard",       # "hard" | "sigmoid" | "smoothstep"
        sdf_sharpness: float = 30.0,    # used when sdf_decode="sigmoid"
        sdf_filter_width_px: float = 1.0,   # used when sdf_decode="smoothstep"
        sdf_max_dist: float = 10.0,     # must match data.sdf_max_dist
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        unet = Unet(
            dim=dim,
            num_classes=num_classes,
            cond_drop_prob=cond_drop_prob,
            dim_mults=dim_mults,
            channels=channels,
            attn_dim_head=attn_dim_head,
            attn_heads=attn_heads,
        )
        self.diffusion = GaussianDiffusion(
            unet,
            image_size=image_size,
            timesteps=timesteps,
            sampling_timesteps=sampling_timesteps,
            objective=objective,
            beta_schedule=beta_schedule,
            offset_noise_strength=offset_noise_strength,
            min_snr_loss_weight=min_snr_loss_weight,
            use_cfg_plus_plus=use_cfg_plus_plus,
            stroke_weight_alpha=stroke_weight_alpha,
            normalize_loss_per_image=normalize_loss_per_image,
        )

        self._ema = ExponentialMovingAverage(self.diffusion.parameters(), decay=ema_decay)
        self._inception = None
        self._perceptual = None
        self._fid_real_cache: dict[str, torch.Tensor] = {}
        self._val_check_count: int = 0

    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        if self.global_step >= self.hparams.ema_start_step:
            self._ema.update()

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        checkpoint["ema_state"] = self._ema.state_dict()
        # 推論時不要なネットワーク（FID 用 InceptionV3・パーセプチュアルロス）を state_dict から除外する
        checkpoint["state_dict"] = {
            k: v for k, v in checkpoint["state_dict"].items()
            if not k.startswith("_inception.") and not k.startswith("_perceptual.")
        }

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        if "ema_state" in checkpoint:
            self._ema.load_state_dict(checkpoint["ema_state"])

    def set_sampling_timesteps(self, n: int) -> None:
        """DDIM のサンプリングステップ数を変更する。

        sampling_timesteps と is_ddim_sampling を同時に更新することで
        生合成を保つ。generate_fonts.py など推論時に使う。
        """
        self.diffusion.sampling_timesteps = n
        self.diffusion.is_ddim_sampling = n < self.diffusion.num_timesteps

    def on_fit_start(self) -> None:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.logger.log_hyperparams({"trainable_params": trainable, "total_params": total})
        self.print(f"Parameters: {trainable:,} trainable / {total:,} total")
        self._ema.shadow_params = [p.to(self.device) for p in self._ema.shadow_params]

    def _get_perceptual(self) -> torch.nn.Module:
        if self._perceptual is None:
            from .perceptual import build_perceptual_loss
            self._perceptual = build_perceptual_loss(self.hparams.perceptual_network)
        return self._perceptual.to(self.device)

    def _loss_with_details(
        self,
        img: torch.Tensor,
        classes: torch.Tensor,
        stroke_density: torch.Tensor | None = None,
        stroke_map: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """拡散損失をサンプルごとに計算して返す。

        Returns:
            loss:            スカラー損失 (学習に使う値、パーセプチュアル損失を含む)
            loss_per_sample: 重み付き per-sample 損失 (B,)
            t:               各サンプルのタイムステップ (B,)
            perc:            パーセプチュアル損失スカラー (perceptual_weight=0 なら None)
        """
        use_perceptual = self.hparams.perceptual_weight > 0
        result = self.diffusion(
            img, classes=classes,
            stroke_density=stroke_density, stroke_map=stroke_map,
            return_per_sample=True, return_x0=use_perceptual,
        )

        if use_perceptual:
            loss, per_sample, t, x0_pred = result
            T = self.diffusion.num_timesteps
            t_max = int(T * self.hparams.perceptual_t_max_frac)
            mask = t < t_max
            if mask.any():
                # x̂_0 は [-1,1] スケール → [0,1] に変換してネットワークへ渡す
                x0_01 = ((x0_pred[mask] + 1) / 2).clamp(0, 1)
                real_01 = img[mask]  # img は DiffusionModule 入力時点で [0,1]
                if x0_01.shape[1] == 1:
                    x0_01 = x0_01.repeat(1, 3, 1, 1)
                    real_01 = real_01.repeat(1, 3, 1, 1)
                # no_grad は使わない — x̂_0 への勾配が必要
                perc = self._get_perceptual()(x0_01, real_01)
            else:
                perc = torch.zeros(1, device=self.device)
            loss = loss + self.hparams.perceptual_weight * perc
        else:
            loss, per_sample, t = result
            perc = None

        return loss, per_sample, t, perc

    def _log_details(self, prefix: str, weighted: torch.Tensor, t: torch.Tensor, batch_size: int) -> None:
        """ノイズレベル別の損失をログする。"""
        T = self.diffusion.num_timesteps
        is_train = prefix == "train"
        for name, lo, hi in _NOISE_BUCKETS:
            mask = (t >= int(T * lo)) & (t < int(T * hi))
            if mask.any():
                self.log(
                    f"{prefix}/loss_{name}_noise",
                    weighted[mask].mean(),
                    on_step=is_train, on_epoch=True,
                    batch_size=batch_size,
                )

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        loss, weighted, t, perc = self._loss_with_details(
            batch["image"], batch["label"],
            stroke_density=batch.get("stroke_density"),
            stroke_map=batch.get("stroke_map"),
        )
        bs = batch["image"].shape[0]
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=bs)
        self._log_details("train", weighted, t, bs)
        if perc is not None:
            self.log("train/perceptual_loss", perc, on_step=True, on_epoch=True, batch_size=bs)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        loss, weighted, t, perc = self._loss_with_details(
            batch["image"], batch["label"],
            stroke_density=batch.get("stroke_density"),
            stroke_map=batch.get("stroke_map"),
        )
        bs = batch["image"].shape[0]
        self.log("val/loss", loss, on_epoch=True, prog_bar=True, batch_size=bs)
        self._log_details("val", weighted, t, bs)
        if perc is not None:
            self.log("val/perceptual_loss", perc, on_epoch=True, batch_size=bs)

    def on_validation_epoch_end(self) -> None:
        self._val_check_count += 1
        count = self._val_check_count

        every = self.hparams.sample_every_n_val_checks
        if every > 0 and count % every == 0:
            self._log_samples()

        every_fid = self.hparams.fid_every_n_val_checks
        if every_fid > 0 and count % every_fid == 0:
            self._log_fid()

    def _sdf_decode(self, images: torch.Tensor) -> torch.Tensor:
        if not self.hparams.use_sdf:
            return images
        mode = self.hparams.sdf_decode
        if mode == "smoothstep":
            from real_fonts_vs_gen_fonts.sdf import sdf_to_smoothstep
            return sdf_to_smoothstep(
                images,
                filter_width_px=self.hparams.sdf_filter_width_px,
                max_dist=self.hparams.sdf_max_dist,
            )
        from real_fonts_vs_gen_fonts.sdf import sdf_to_binary
        sharpness = self.hparams.sdf_sharpness if mode == "sigmoid" else None
        return sdf_to_binary(images, sharpness=sharpness)

    def _inference_batch_size(self) -> int | None:
        """学習ミニバッチサイズ。datamodule が無い経路では None（分割なし）。"""
        dm = getattr(self.trainer, "datamodule", None)
        return getattr(dm, "batch_size", None) if dm is not None else None

    def _sample_in_chunks(self, classes: torch.Tensor, cond_scale: float) -> torch.Tensor:
        """学習ミニバッチサイズを上限に classes を分割してサンプリングし結合する。

        FID 等で大量（fid_n_samples=256 など）を一括サンプリングすると、学習時より
        はるかに大きい VRAM を消費する。チャンク分割は各サンプル独立なので数値的に等価。
        """
        chunk = self._inference_batch_size() or classes.shape[0]
        self.diffusion.eval()
        with self._ema.average_parameters(), torch.no_grad():
            samples = [
                self.diffusion.sample(classes[i : i + chunk], cond_scale=cond_scale)
                for i in range(0, classes.shape[0], chunk)
            ]
        self.diffusion.train()
        return torch.cat(samples, dim=0)

    def _log_samples(self) -> None:
        num_classes = self.hparams.num_classes
        n = self.hparams.sample_n_images
        reps = math.ceil(n / num_classes)
        classes = torch.arange(num_classes, device=self.device).repeat(reps)[:n]

        samples = self._sample_in_chunks(classes, self.hparams.sample_cond_scale)
        samples = self._sdf_decode(samples)

        nrow = int(math.sqrt(n))
        grid = make_grid(samples, nrow=nrow, value_range=(0, 1))

        samples_dir = Path(self.trainer.default_root_dir) / "results" / "samples"
        samples_dir.mkdir(parents=True, exist_ok=True)
        save_image(grid, samples_dir / f"step_{self.global_step:06d}.png")

        if self.logger is not None:
            self.logger.experiment.add_image(
                "samples", grid, global_step=self.global_step
            )

    def _get_inception(self):
        if self._inception is None:
            from piq.feature_extractors import InceptionV3
            self._inception = InceptionV3(normalize_input=True).eval()
        return self._inception.to(self.device)

    def _inception_features(self, images: torch.Tensor) -> torch.Tensor:
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)
        inception = self._get_inception()
        chunk = min(64, self._inference_batch_size() or 64)
        feats: list[torch.Tensor] = []
        with torch.no_grad():
            for i in range(0, len(images), chunk):
                f = inception(images[i : i + chunk].to(self.device))
                if isinstance(f, (list, tuple)):
                    f = f[0]  # piq InceptionV3 returns list[Tensor]
                if f.dim() == 4:
                    f = f.mean(dim=[2, 3])  # (B, 2048, 1, 1) → (B, 2048)
                feats.append(f.cpu())
        return torch.cat(feats, 0)

    def _get_real_features(self, split: str) -> torch.Tensor:
        if split in self._fid_real_cache:
            return self._fid_real_cache[split]
        dm = self.trainer.datamodule
        loader_fn = {"train": dm.train_dataloader, "val": dm.val_dataloader, "test": dm.test_dataloader}[split]
        imgs: list[torch.Tensor] = []
        n = self.hparams.fid_n_samples
        for batch in loader_fn():
            imgs.append(batch["image"])
            if sum(x.shape[0] for x in imgs) >= n:
                break
        # FID is computed in grayscale image space, so decode SDF (or other) real
        # representations the same way generated samples are decoded in _log_fid.
        real_imgs = self._sdf_decode(torch.cat(imgs, 0)[:n])
        feats = self._inception_features(real_imgs)
        self._fid_real_cache[split] = feats
        return feats

    def _log_fid(self) -> None:
        import piq

        n = self.hparams.fid_n_samples
        num_classes = self.hparams.num_classes
        reps = math.ceil(n / num_classes)
        classes = torch.arange(num_classes, device=self.device).repeat(reps)[:n]

        gen_imgs = self._sample_in_chunks(classes, self.hparams.sample_cond_scale)
        gen_imgs = self._sdf_decode(gen_imgs)

        gen_feats = self._inception_features(gen_imgs)
        for split in self.hparams.fid_splits:
            real_feats = self._get_real_features(split)
            fid_score = piq.FID()(real_feats, gen_feats).item()
            self.log(f"fid/{split}", fid_score, on_epoch=True, prog_bar=(split == "val"))

    def configure_optimizers(self):
        optimizer = AdamW(
            self.diffusion.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        warmup = self.hparams.lr_warmup_steps
        total = self.hparams.lr_total_steps

        # lr_total_steps=0 でスケジューラを無効化（定数 LR）
        if total == 0:
            return optimizer

        if warmup > 0:
            # lr: 0 → peak over warmup steps, then cosine decay to 0
            warmup_scheduler = LinearLR(
                optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup
            )
            decay_scheduler = CosineAnnealingLR(optimizer, T_max=total - warmup)
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, decay_scheduler],
                milestones=[warmup],
            )
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=total)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
