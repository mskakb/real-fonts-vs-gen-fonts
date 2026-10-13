from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torchvision.transforms import v2
from tqdm import tqdm

from real_fonts_vs_gen_fonts.models.scorer_module import ScorerModule


def score_directory(
    data_dir: Path,
    scorer: ScorerModule,
    transform: v2.Transform,
    batch_size: int,
    device: torch.device,
    overwrite: bool = False,
    output_path: Path | None = None,
) -> None:
    """ディレクトリ内の全画像に realness_score を付与して保存する。

    Args:
        data_dir:     images/ + metadata.parquet を含むディレクトリ。
        scorer:       学習済み ScorerModule。
        transform:    画像の前処理 transform。
        batch_size:   スコアリングのバッチサイズ。
        device:       実行デバイス。
        overwrite:    True のとき既存出力を上書きする。
        output_path:  出力先 parquet パス。None のとき metadata.parquet に in-place 書き込み。
    """
    meta_path = data_dir / "metadata.parquet"
    images_dir = data_dir / "images"

    if not meta_path.exists():
        print(f"[SKIP] metadata.parquet not found: {data_dir}")
        return

    # in-place モードでは元ファイルに realness_score が既存かで判定
    if output_path is None:
        dest = meta_path
    else:
        dest = output_path

    if dest != meta_path and dest.exists() and not overwrite:
        print(f"[SKIP] output already exists: {dest}  (use --overwrite to recompute)")
        return

    meta = pd.read_parquet(meta_path)

    if output_path is None and "realness_score" in meta.columns and not overwrite:
        print(f"[SKIP] realness_score already exists: {data_dir}  (use --overwrite to recompute)")
        return

    print(f"Scoring {len(meta):,} images in {data_dir} ...")
    all_scores: list[float] = []

    for batch_start in tqdm(range(0, len(meta), batch_size), unit="batch"):
        batch_meta = meta.iloc[batch_start : batch_start + batch_size]
        imgs: list[torch.Tensor] = []
        for _, row in batch_meta.iterrows():
            img = Image.open(images_dir / row["file"]).convert("RGB")
            imgs.append(transform(img))

        batch_t = torch.stack(imgs).to(device)
        with torch.no_grad():
            scores = scorer.scorer(batch_t.float()).cpu().tolist()
        all_scores.extend(scores)

    meta["realness_score"] = all_scores
    meta.to_parquet(dest, index=False)

    mean_s = sum(all_scores) / len(all_scores)
    print(f"  → saved to {dest}")
    print(f"     mean={mean_s:.4f}  min={min(all_scores):.4f}  max={max(all_scores):.4f}")
