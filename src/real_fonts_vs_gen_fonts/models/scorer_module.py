from __future__ import annotations

import torch
import torch.nn.functional as F
from lightning import LightningModule
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    ReduceLROnPlateau,
    SequentialLR,
)
from torch_ema import ExponentialMovingAverage
from torchmetrics.classification import BinaryAUROC

from real_fonts_vs_gen_fonts.losses.realness_loss import (
    ranknet_loss,
    realness_loss,
    smooth_labels,
)
from real_fonts_vs_gen_fonts.metrics.classification import make_scorer_metrics
from real_fonts_vs_gen_fonts.models.scorer import RealnessScorer


class ScorerModule(LightningModule):
    """RealnessScorer の LightningModule ラッパー。

    ranking=True（デフォルト）: RankNet ペアワイズ損失。
        バッチ形式: {"real_image": Tensor, "gen_image": Tensor, ...}
        pair_sampling="in_batch_hard" の場合、training_step 内でバッチ内ハードマイニングを実施。

    ranking=False: BCE 分類損失（従来モード）。
        バッチ形式: {"image": Tensor, "label": Tensor, ...}

    Args:
        ranking:       True で RankNet モード、False で分類モード。
        pair_sampling: ランキングモードのペアリング戦略。
                       "random" | "same_char" | "in_batch_hard"
        backbone:      torchvision ResNet 系の名前（"resnet18" | "resnet50"）または
                       ViT の名前（`vit_backbone.VIT_SPECS`、例: "vit_small_p8"）。
        pretrained:    backbone の ImageNet 事前学習済み重みを使うか（ViT では常に未使用）。
        dropout:       head の Dropout 率。
        hidden_dim:    head の中間層次元数。None で 1 層線形ヘッド (Linear(feature_dim, 1)) になる。
        image_size:    入力解像度（px）。ViT backbone の構築にのみ使う（train_scorer.py /
                       train_scorer_classifier.py が data.image_size から注入する）。
        label_smoothing: 分類モードのラベル平滑化強度（ranking=False 時のみ使用）。
        lr:            head のピーク学習率。
        backbone_lr_scale: backbone の lr = lr × backbone_lr_scale。1.0 で head と同じ lr。
        weight_decay:  AdamW の weight decay。
        lr_warmup_steps: linear warmup のステップ数。0 で warmup なし（scheduler="cosine" 時のみ有効）。
        lr_total_steps:  スケジューラの総ステップ数。train_scorer.py で trainer.max_steps から注入する。
        scheduler:       "cosine"（warmup→CosineAnnealing）または "plateau"（ReduceLROnPlateau）。
                         "plateau" のとき warmup は無効。
        plateau_factor:  plateau 時に lr に掛ける係数。
        plateau_patience: 改善なしを許容する検証回数（val_check_interval 単位）。
        plateau_threshold: 改善とみなす最小変化量（ReduceLROnPlateau の threshold）。
        plateau_min_lr:  lr の下限。
        plateau_cooldown: lr 減衰後に再評価を再開するまで待つ検証回数。
        plateau_monitor: plateau が監視するメトリクス名。
        plateau_mode:    "min"（loss 系）/ "max"（accuracy 系）。
        freeze_backbone_steps: この step 数まで backbone を凍結し head のみ学習。0 で即時全パラメータを学習。
        ema_decay:       EMA 平滑化係数。
        ema_start_step:  EMA 更新を開始するグローバルステップ。
        tie_weight_real: ranking モードで real-real の "tie" ペア損失（RankNet p_ij=0.5）
                         に掛ける重み。0 で無効。
        tie_weight_gen:  ranking モードで gen-gen の "tie" ペア損失（RankNet p_ij=0.5）
                         に掛ける重み。0 で無効。
        no_decay_norm_bias: True のとき、norm/bias/ViT の class_token・pos_embedding パラメータを
                         weight decay 0 の別 param group に分ける。ViT のスクラッチ学習では
                         実質必須だが、既定 False で既存 ResNet 実験の recipe を変えない。
    """

    def __init__(
        self,
        ranking: bool = True,
        pair_sampling: str = "random",
        backbone: str = "resnet18",
        pretrained: bool = True,
        dropout: float = 0.3,
        hidden_dim: int | None = 128,
        image_size: int = 96,
        label_smoothing: float = 0.1,
        lr: float = 1e-4,
        backbone_lr_scale: float = 1.0,
        weight_decay: float = 1e-2,
        lr_warmup_steps: int = 500,
        lr_total_steps: int = 50_000,
        scheduler: str = "cosine",
        plateau_factor: float = 0.5,
        plateau_patience: int = 5,
        plateau_threshold: float = 1e-4,
        plateau_min_lr: float = 1e-6,
        plateau_cooldown: int = 0,
        plateau_monitor: str = "val/loss",
        plateau_mode: str = "min",
        freeze_backbone_steps: int = 0,
        ema_decay: float = 0.999,
        ema_start_step: int = 0,
        tie_weight_real: float = 0.0,
        tie_weight_gen: float = 0.0,
        no_decay_norm_bias: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.scorer = RealnessScorer(
            backbone=backbone,
            pretrained=pretrained,
            dropout=dropout,
            hidden_dim=hidden_dim,
            image_size=image_size,
        )

        if ranking:
            self.val_auroc = BinaryAUROC()
            self.test_auroc = BinaryAUROC()
        else:
            self.val_metrics = make_scorer_metrics("val/")
            self.test_metrics = make_scorer_metrics("test/")

        self._ema = ExponentialMovingAverage(self.scorer.parameters(), decay=ema_decay)

    def _set_backbone_grad(self, requires_grad: bool) -> None:
        for p in self.scorer.backbone.parameters():
            p.requires_grad = requires_grad

    def on_fit_start(self) -> None:
        if self.hparams.freeze_backbone_steps > 0:
            self._set_backbone_grad(False)
            self.print(f"Backbone frozen for first {self.hparams.freeze_backbone_steps} steps")
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.logger.log_hyperparams({"trainable_params": trainable, "total_params": total})
        self.print(f"Parameters: {trainable:,} trainable / {total:,} total")
        self._ema.shadow_params = [p.to(self.device) for p in self._ema.shadow_params]

    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        if self.global_step >= self.hparams.ema_start_step:
            self._ema.update()

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        checkpoint["ema_state"] = self._ema.state_dict()

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        if "ema_state" in checkpoint:
            self._ema.load_state_dict(checkpoint["ema_state"])

    # ------------------------------------------------------------------
    # training
    # ------------------------------------------------------------------

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        freeze_steps = self.hparams.freeze_backbone_steps
        if freeze_steps > 0 and self.global_step == freeze_steps:
            self._set_backbone_grad(True)
            self.print(f"Backbone unfrozen at step {self.global_step}")

        if self.hparams.ranking:
            return self._ranking_train_step(batch)
        return self._classification_train_step(batch)

    def _ranking_train_step(self, batch: dict) -> torch.Tensor:
        tie_weight_real = self.hparams.tie_weight_real
        tie_weight_gen = self.hparams.tie_weight_gen
        use_tie = tie_weight_real > 0 or tie_weight_gen > 0

        # Forward real and gen (and tie partners) in a SINGLE mixed batch so BatchNorm
        # sees both classes together. Forwarding pure-class batches separately lets BN's
        # per-batch statistics leak the real/gen label at train time; the model then
        # collapses to chance at eval time (fixed running stats). Concatenating keeps the
        # train- and eval-time normalization consistent.
        if use_tie:
            logits = self.scorer.forward_logits(torch.cat([
                batch["real_image"], batch["gen_image"],
                batch["real_image2"], batch["gen_image2"],
            ]))
            s_real, s_gen, s_real2, s_gen2 = logits.chunk(4)
        else:
            logits = self.scorer.forward_logits(
                torch.cat([batch["real_image"], batch["gen_image"]])
            )
            s_real, s_gen = logits.chunk(2)

        if self.hparams.pair_sampling == "in_batch_hard":
            # バッチ内で最も違反しているペアに再マッピングする。
            # gen スコアが高い順 ↔ real スコアが低い順で対応させることで
            # ランキング制約の違反量を最大化する。
            gen_order = s_gen.argsort(descending=True)
            real_order = s_real.argsort()
            s_real = s_real[real_order]
            s_gen = s_gen[gen_order]

        rank_loss = ranknet_loss(s_real, s_gen)
        rank_acc = (s_real > s_gen).float().mean()
        loss = rank_loss

        self.log("train/rank_loss", rank_loss, on_step=True, on_epoch=True)
        self.log("train/rank_acc", rank_acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/score_real", torch.sigmoid(s_real).mean(), on_step=True, on_epoch=True)
        self.log("train/score_gen", torch.sigmoid(s_gen).mean(), on_step=True, on_epoch=True)
        self.log("train/logit_real", s_real.mean(), on_step=True, on_epoch=True)
        self.log("train/logit_gen", s_gen.mean(), on_step=True, on_epoch=True)

        if use_tie:
            tie_loss_real = ranknet_loss(s_real, s_real2, p_ij=0.5)
            tie_loss_gen = ranknet_loss(s_gen, s_gen2, p_ij=0.5)
            self.log("train/tie_loss_real", tie_loss_real, on_step=True, on_epoch=True)
            self.log("train/tie_loss_gen", tie_loss_gen, on_step=True, on_epoch=True)
            loss = loss + tie_weight_real * tie_loss_real + tie_weight_gen * tie_loss_gen

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def _classification_train_step(self, batch: dict) -> torch.Tensor:
        images = batch["image"]
        labels = batch["label"].float()
        soft_labels = smooth_labels(labels, smoothing=self.hparams.label_smoothing)
        logits = self.scorer.forward_logits(images)
        loss = realness_loss(logits, soft_labels)
        scores = torch.sigmoid(logits)
        real_mask = labels == 1
        gen_mask = labels == 0

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/accuracy", ((scores > 0.5) == labels).float().mean(),
                 on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/score_real", scores[real_mask].mean(), on_step=True, on_epoch=True)
        self.log("train/score_gen", scores[gen_mask].mean(), on_step=True, on_epoch=True)
        self.log("train/logit_real", logits[real_mask].mean(), on_step=True, on_epoch=True)
        self.log("train/logit_gen", logits[gen_mask].mean(), on_step=True, on_epoch=True)
        return loss

    # ------------------------------------------------------------------
    # validation
    # ------------------------------------------------------------------

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        with self._ema.average_parameters():
            if self.hparams.ranking:
                self._ranking_val_step(batch)
            else:
                self._classification_val_step(batch)

    def _ranking_val_step(self, batch: dict) -> None:
        # Forward both classes together for parity with _ranking_train_step (see the
        # BatchNorm note there). In eval mode BN uses fixed running stats so this is
        # numerically identical to separate forwards, but keeps the convention uniform.
        logits = self.scorer.forward_logits(
            torch.cat([batch["real_image"], batch["gen_image"]])
        )
        s_real, s_gen = logits.chunk(2)

        loss = ranknet_loss(s_real, s_gen)
        rank_acc = (s_real > s_gen).float().mean()
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        self.log("val/rank_acc", rank_acc, on_epoch=True, prog_bar=True)
        self.log("val/score_real", torch.sigmoid(s_real).mean(), on_epoch=True)
        self.log("val/score_gen", torch.sigmoid(s_gen).mean(), on_epoch=True)
        self.log("val/logit_real", s_real.mean(), on_epoch=True)
        self.log("val/logit_gen", s_gen.mean(), on_epoch=True)

        # AUROC: real スコアを正例、gen スコアを負例として評価
        scores = torch.cat([torch.sigmoid(s_real), torch.sigmoid(s_gen)])
        labels = torch.cat([torch.ones_like(s_real, dtype=torch.long),
                            torch.zeros_like(s_gen, dtype=torch.long)])
        self.val_auroc.update(scores, labels)

    def _classification_val_step(self, batch: dict) -> None:
        images = batch["image"]
        labels = batch["label"]
        logits = self.scorer.forward_logits(images)
        loss = F.binary_cross_entropy_with_logits(logits, labels.float())
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        scores = torch.sigmoid(logits)
        self.val_metrics.update(scores, labels)

    def on_validation_epoch_end(self) -> None:
        if self.hparams.ranking:
            self.log("val/auroc", self.val_auroc.compute(), prog_bar=True)
            self.val_auroc.reset()
        else:
            self.log_dict(self.val_metrics.compute(), prog_bar=True)
            self.val_metrics.reset()

    # ------------------------------------------------------------------
    # test
    # ------------------------------------------------------------------

    def test_step(self, batch: dict, batch_idx: int) -> None:
        with self._ema.average_parameters():
            if self.hparams.ranking:
                logits = self.scorer.forward_logits(
                    torch.cat([batch["real_image"], batch["gen_image"]])
                )
                s_real, s_gen = logits.chunk(2)
                scores = torch.cat([torch.sigmoid(s_real), torch.sigmoid(s_gen)])
                labels = torch.cat([torch.ones_like(s_real, dtype=torch.long),
                                    torch.zeros_like(s_gen, dtype=torch.long)])
                self.test_auroc.update(scores, labels)
            else:
                images = batch["image"]
                labels = batch["label"]
                scores = self.scorer(images)
                self.test_metrics.update(scores, labels)

    def on_test_epoch_end(self) -> None:
        if self.hparams.ranking:
            self.log("test/auroc", self.test_auroc.compute())
            self.test_auroc.reset()
        else:
            self.log_dict(self.test_metrics.compute())
            self.test_metrics.reset()

    # ------------------------------------------------------------------
    # optimizer
    # ------------------------------------------------------------------

    def _split_decay(self, params: list[tuple[str, torch.nn.Parameter]]) -> tuple[list, list]:
        """norm/bias/ViT の class_token・pos_embedding を weight-decay 対象から除外する。

        1 次元パラメータ（LayerNorm/BatchNorm の weight・bias、全 Linear の bias）と
        `class_token`/`pos_embedding`（名前で判定、torchvision VisionTransformer 固有の
        バッファ的パラメータ）を no-decay group に振り分ける。ViT のスクラッチ学習では
        これを分けないと汎化が悪化しやすい（標準的な ViT 学習レシピ）。
        """
        decay, no_decay = [], []
        for name, p in params:
            if not p.requires_grad:
                continue
            if p.ndim <= 1 or "class_token" in name or "pos_embedding" in name:
                no_decay.append(p)
            else:
                decay.append(p)
        return decay, no_decay

    def configure_optimizers(self):
        backbone_lr = self.hparams.lr * self.hparams.backbone_lr_scale
        if self.hparams.no_decay_norm_bias:
            bb_decay, bb_no_decay = self._split_decay(
                list(self.scorer.backbone.named_parameters())
            )
            head_decay, head_no_decay = self._split_decay(
                list(self.scorer.head.named_parameters())
            )
            param_groups = [
                {"params": bb_decay, "lr": backbone_lr, "weight_decay": self.hparams.weight_decay},
                {"params": bb_no_decay, "lr": backbone_lr, "weight_decay": 0.0},
                {"params": head_decay, "lr": self.hparams.lr, "weight_decay": self.hparams.weight_decay},
                {"params": head_no_decay, "lr": self.hparams.lr, "weight_decay": 0.0},
            ]
            optimizer = AdamW(param_groups)
        else:
            param_groups = [
                {"params": self.scorer.backbone.parameters(), "lr": backbone_lr},
                {"params": self.scorer.head.parameters(), "lr": self.hparams.lr},
            ]
            optimizer = AdamW(param_groups, weight_decay=self.hparams.weight_decay)

        if self.hparams.scheduler == "plateau":
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode=self.hparams.plateau_mode,
                factor=self.hparams.plateau_factor,
                patience=self.hparams.plateau_patience,
                threshold=self.hparams.plateau_threshold,
                min_lr=self.hparams.plateau_min_lr,
                cooldown=self.hparams.plateau_cooldown,
            )
            # plateau は val 指標を見て step するため検証カデンツに合わせる。
            # val_check_interval が int (ステップ) なら interval="step" で揃え、float
            # (エポック) なら interval="epoch"。max_steps < 1 エポックでも発火させる。
            val_interval = self.trainer.val_check_interval
            interval, frequency = (
                ("step", val_interval) if isinstance(val_interval, int) else ("epoch", 1)
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": interval,
                    "frequency": frequency,
                    "monitor": self.hparams.plateau_monitor,
                },
            }

        if self.hparams.scheduler != "cosine":
            raise ValueError(f"Unknown scheduler: {self.hparams.scheduler!r}")

        warmup = self.hparams.lr_warmup_steps
        total = self.hparams.lr_total_steps

        if warmup > 0:
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
