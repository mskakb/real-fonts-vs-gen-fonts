from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from real_fonts_vs_gen_fonts.models.scorer import RealnessScorer


_VALID_LAYERS = ("layer1", "layer2", "layer3", "layer4")


class GradCAM:
    """ResNet の指定レイヤーを対象とした Grad-CAM。

    フック登録 → compute() でヒートマップ取得 → remove() でフック解除。
    layer1〜4 は空間解像度・チャネル数が backbone 依存で異なる（浅い層ほど高解像度・
    低チャネル）。ただし Grad-CAM 本来の重み付け（GAP した勾配をチャネル重みにする）は
    最終 conv 層を想定した手法のため、layer1〜3 での解釈は layer4 ほど確立していない点に注意。

    Args:
        scorer: RealnessScorer インスタンス。
        layer:  フック対象レイヤー名（"layer1"〜"layer4"、既定 "layer4"）。
    """

    def __init__(self, scorer: RealnessScorer, layer: str = "layer4") -> None:
        if scorer._is_vit:
            raise ValueError(
                "GradCAM assumes a ResNet backbone (backbone.layer1..4 as 4D conv feature "
                "maps) and does not support ViT backbones. Use "
                "real_fonts_vs_gen_fonts.models.attention_maps / scripts/analyze_attention.py "
                "for ViT Scorers instead."
            )
        if layer not in _VALID_LAYERS:
            raise ValueError(f"Unknown layer '{layer}'. Use one of {_VALID_LAYERS}.")
        self._feat: torch.Tensor | None = None
        self._grad: torch.Tensor | None = None
        self._scorer = scorer

        target = getattr(scorer.backbone, layer)
        self._fwd = target.register_forward_hook(
            lambda m, i, o: setattr(self, "_feat", o)
        )
        self._bwd = target.register_full_backward_hook(
            lambda m, gi, go: setattr(self, "_grad", go[0])
        )

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        """Grad-CAM ヒートマップを返す。

        Args:
            x: 入力画像 (B, C, H, W)。requires_grad は自動で設定する。
        Returns:
            ヒートマップ (B, H', W')、値域 [0, 1]。
            H', W' はコンストラクタで指定したレイヤーの空間サイズ（浅い層ほど大きい。
            例: 96px 入力・resnet50 なら layer1≈24x24、layer4≈3x3）。
        """
        self._scorer.eval()
        x = x.detach().requires_grad_(True)

        logits = self._scorer.forward_logits(x)  # (B,)
        self._scorer.zero_grad()
        logits.sum().backward()

        # チャネル方向に勾配を GAP → 重みとして feature map に掛ける
        weights = self._grad.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
        cam = torch.relu((weights * self._feat).sum(dim=1))   # (B, H', W')

        # バッチ内で個別に [0, 1] に正規化
        B = cam.shape[0]
        flat = cam.view(B, -1)
        lo = flat.min(dim=1).values.view(B, 1, 1)
        hi = flat.max(dim=1).values.view(B, 1, 1)
        cam = (cam - lo) / (hi - lo + 1e-8)

        return cam.detach().cpu()

    def remove(self) -> None:
        """フックを解除する。使用後に必ず呼ぶこと。"""
        self._fwd.remove()
        self._bwd.remove()


def overlay_heatmap(
    image: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.5,
    target_size: tuple[int, int] | None = None,
) -> np.ndarray:
    """画像にヒートマップをオーバーレイする。

    Args:
        image:       元画像 (H, W, 3)、値域 [0, 1]。
        cam:         Grad-CAM ヒートマップ (H', W')、値域 [0, 1]。
        alpha:       ヒートマップの不透明度。
        target_size: ヒートマップのリサイズ先 (H, W)。None で image と同じサイズ。
    Returns:
        オーバーレイ画像 (H, W, 3)、値域 [0, 1]。
    """
    import matplotlib.cm as cm

    h, w = image.shape[:2]
    size = target_size or (h, w)

    cam_t = torch.tensor(cam).unsqueeze(0).unsqueeze(0)  # (1, 1, H', W')
    cam_resized = F.interpolate(cam_t, size=size, mode="bilinear", align_corners=False)
    cam_np = cam_resized.squeeze().numpy()

    heatmap = cm.hot(cam_np)[..., :3]  # (H, W, 3)
    return np.clip(alpha * heatmap + (1 - alpha) * image, 0, 1)
