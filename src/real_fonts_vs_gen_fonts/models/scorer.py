from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
from torchvision.models import (
    ResNet18_Weights,
    ResNet50_Weights,
    resnet18,
    resnet50,
)

from real_fonts_vs_gen_fonts.models.vit_backbone import VIT_SPECS, build_vit, stage_tokens

# Supported ResNet backbones (torchvision ResNet family). Each entry maps a name to
# its constructor and default ImageNet weights. The head input dim is derived from
# the constructed network's `fc.in_features`, so adding a ResNet variant here needs
# no further bookkeeping. `extract_intermediate()` and `GradCAM` assume the ResNet
# attribute layout (conv1/bn1/relu/maxpool/layer1..4), so this table stays ResNet-only.
# ViT backbones are listed separately in `vit_backbone.VIT_SPECS` and go through
# `build_vit`/`stage_tokens` instead; `models/attention_maps.py` and
# `scripts/analyze_attention.py` are the ViT counterpart to Grad-CAM/occlusion.
_BACKBONES: dict[str, tuple[Callable[..., nn.Module], object]] = {
    "resnet18": (resnet18, ResNet18_Weights.IMAGENET1K_V1),
    "resnet50": (resnet50, ResNet50_Weights.IMAGENET1K_V2),
}


class RealnessScorer(nn.Module):
    """ResNet 系 または ViT 系 backbone（既定 resnet18）の実/生成フォント二値分類器。

    4 つのメソッドを用途によって使い分ける:
      - forward()               : sigmoid スコア。Classifier Guidance の勾配計算に使う。
      - forward_logits()        : sigmoid 前のロジット。学習時の損失計算に使う (fp16 安全)。
      - extract_features()      : backbone 出力の特徴量（次元は backbone 依存）。t-SNE / UMAP 可視化に使う。
      - extract_intermediate()  : layer1〜4 相当の GAP 特徴量。レベル別解析に使う
                                   （ViT は patch トークン平均。`vit_backbone.stage_tokens` 参照）。

    Args:
        backbone:   torchvision ResNet 系の名前（`_BACKBONES` のキー）または
                    ViT の名前（`vit_backbone.VIT_SPECS` のキー、例: "vit_small_p8"）。
        pretrained: ImageNet 事前学習済み重みを使うか（ViT では常に未使用、スクラッチ学習）。
        dropout:    head の Dropout 率（hidden_dim が None の場合は未使用）。
        hidden_dim: head の中間層次元数。None で 1 層線形ヘッド (Linear(feature_dim, 1)) になる。
        image_size: 入力解像度（px）。ViT backbone の構築にのみ使う（ResNet では無視）。
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        dropout: float = 0.3,
        hidden_dim: int | None = 128,
        image_size: int = 96,
    ) -> None:
        super().__init__()
        self._is_vit = backbone in VIT_SPECS
        if self._is_vit:
            net = build_vit(backbone, image_size)
            self.feature_dim: int = net.hidden_dim
        elif backbone in _BACKBONES:
            ctor, weights_enum = _BACKBONES[backbone]
            net = ctor(weights=weights_enum if pretrained else None)
            # `fc` を Identity に置換する前に入力次元を取得しておく（backbone ごとに異なる）
            self.feature_dim = net.fc.in_features
            # fc を Identity に置き換えることで avgpool → flatten (B, feature_dim) を出力させる
            net.fc = nn.Identity()
        else:
            raise ValueError(
                f"Unknown backbone '{backbone}'. Supported ResNet: {sorted(_BACKBONES)}; "
                f"supported ViT: {sorted(VIT_SPECS)}"
            )
        self.backbone = net

        if hidden_dim is None:
            self.head: nn.Module = nn.Linear(self.feature_dim, 1)
        else:
            self.head = nn.Sequential(
                nn.Linear(self.feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """avgpool 後の特徴量を返す。

        Args:
            x: 画像テンソル (B, C, H, W)、値域 [0, 1]。
        Returns:
            特徴量 (B, feature_dim)。feature_dim は backbone 依存（resnet18=512 / resnet50=2048）。
        """
        return self.backbone(x)

    def extract_intermediate(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """各段（ResNet: layer1〜4／ViT: 4 等分したブロック）の GAP 済み特徴量を返す。

        ResNet は各 layer の次元が backbone 依存（resnet18: 64/128/256/512、
        resnet50: 256/512/1024/2048）。ViT は全層 `feature_dim` で共通（`vit_backbone.
        stage_tokens` が返すトークン列から CLS を除き、patch トークンを平均する）。

        Args:
            x: 画像テンソル (B, C, H, W)、値域 [0, 1]。
        Returns:
            dict with keys "layer1"〜"layer4"（各 (B, C_l)）。
        """
        if self._is_vit:
            tokens = stage_tokens(self.backbone, x)
            return {name: t[:, 1:, :].mean(dim=1) for name, t in tokens.items()}
        b = self.backbone
        h = b.maxpool(b.relu(b.bn1(b.conv1(x))))
        f1 = b.layer1(h)
        f2 = b.layer2(f1)
        f3 = b.layer3(f2)
        f4 = b.layer4(f3)
        return {
            "layer1": f1.mean(dim=(2, 3)),
            "layer2": f2.mean(dim=(2, 3)),
            "layer3": f3.mean(dim=(2, 3)),
            "layer4": f4.mean(dim=(2, 3)),
        }

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        """sigmoid 前のロジットを返す。

        BCEWithLogitsLoss と組み合わせて使う。fp16 学習で数値安定。

        Args:
            x: 画像テンソル (B, C, H, W)、値域 [0, 1]。
        Returns:
            ロジット (B,)。
        """
        features = self.backbone(x)
        return self.head(features).squeeze(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """本物度スコア s(x) ∈ [0, 1] を返す。

        Classifier Guidance での勾配計算:
            scorer.eval()  # BatchNorm の running stats を固定
            with torch.enable_grad():
                x_t = x_t.detach().requires_grad_(True)
                score = scorer(x_hat_0.float())  # fp16 勾配の精度問題を避けるため float32 で渡す
                grad = torch.autograd.grad(score.sum(), x_t)[0]

        Args:
            x: 画像テンソル (B, C, H, W)、値域 [0, 1]。
        Returns:
            本物度スコア (B,)、1 = 実フォント、0 = 生成フォント。
        """
        return torch.sigmoid(self.forward_logits(x))
