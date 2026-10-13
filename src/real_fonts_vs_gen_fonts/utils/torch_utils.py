from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def get_device() -> torch.device:
    """利用可能な場合は CUDA、なければ CPU を返す。"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def extract_features(
    loader: DataLoader,
    encoder: nn.Module,
    device: torch.device,
    desc: str = "",
    output_index: int = 0,
) -> torch.Tensor:
    """DataLoader から特徴量を抽出して結合する。

    各バッチの画像を encoder に通し、output_index 番目の出力を
    空間方向に GAP してから結合する。

    Args:
        loader:       {"image": Tensor} を返す DataLoader。
        encoder:      特徴抽出器。出力はテンソルまたはテンソルのリスト。
        device:       実行デバイス。
        desc:         tqdm のプログレスバーラベル。
        output_index: encoder の出力がリストの場合に使うインデックス。
    Returns:
        特徴量テンソル (N, D)。
    """
    feats_list = []
    for batch in tqdm(loader, desc=desc):
        imgs = batch["image"].to(device)
        out = encoder(imgs)
        feat = out[output_index] if isinstance(out, (list, tuple)) else out
        # 空間次元 (H, W) があれば GAP で潰す
        if feat.dim() == 4:
            feat = feat.flatten(2).mean(dim=2)
        feats_list.append(feat.cpu())
    return torch.cat(feats_list, dim=0)
