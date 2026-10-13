from __future__ import annotations

import torch
from PIL import Image
from torchvision.transforms import v2


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """(C, H, W) float [0, 1] → PIL Image。C=1 で grayscale、C=3 で RGB。"""
    if t.shape[0] == 1:
        arr = (t.clamp(0, 1) * 255).byte().squeeze(0).cpu().numpy()
        return Image.fromarray(arr, mode="L")
    arr = (t.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(arr, mode="RGB")


def make_image_transform(
    image_size: int,
    binarize_threshold: float | None = None,
) -> v2.Transform:
    """標準的な画像前処理パイプラインを返す。

    Args:
        image_size:          リサイズ後の正方形サイズ (px)。
        binarize_threshold:  二値化しきい値。None または負値で無効。
    Returns:
        v2.Compose で束ねた transform。
    """
    transforms: list = [
        v2.Resize((image_size, image_size)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ]
    if binarize_threshold is not None and binarize_threshold >= 0:
        thr = binarize_threshold
        transforms.append(v2.Lambda(lambda x, t=thr: (x > t).float()))
    return v2.Compose(transforms)
