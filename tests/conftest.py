"""Shared fixtures for all tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch
from PIL import Image


@pytest.fixture()
def image_batch() -> torch.Tensor:
    """(B=4, C=3, H=64, W=64) float [0, 1]"""
    return torch.rand(4, 3, 64, 64)


@pytest.fixture()
def binary_labels() -> torch.Tensor:
    """(B=4,) int {0, 1}"""
    return torch.tensor([1, 0, 1, 0])


@pytest.fixture()
def char_labels() -> torch.Tensor:
    """(B=4,) int — uppercase A-Z class indices"""
    return torch.tensor([0, 1, 25, 10])


@pytest.fixture()
def fake_font2dataset_dir(tmp_path: Path) -> Path:
    """font2dataset 形式のディレクトリを tmp_path に作成して返す。"""
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    records = []
    for cp in range(ord("A"), ord("A") + 5):  # A-E の 5 文字
        char = chr(cp)
        for i in range(3):  # フォントファミリー 3 種
            family = f"FakeFont{i}"
            fname = f"{cp:04X}_{family}_000.png"
            Image.new("RGB", (64, 64), color=(255, 255, 255)).save(images_dir / fname)
            records.append({
                "file": fname,
                "char": char,
                "unicode": f"U+{cp:04X}",
                "codepoint": cp,
                "font_path": f"/fake/{family}.ttf",
                "font_family": family,
                "font_style": "Regular",
            })

    pd.DataFrame(records).to_parquet(tmp_path / "metadata.parquet", index=False)
    return tmp_path
