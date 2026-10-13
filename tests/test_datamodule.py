from __future__ import annotations

from pathlib import Path

import pytest
import torch

from real_fonts_vs_gen_fonts.datamodule import (
    CHARSETS,
    FontDataModule,
    _make_label_map,
    _resolve_codepoints,
)


def test_charsets_keys():
    assert set(CHARSETS) == {"uppercase", "lowercase", "digits", "ascii"}
    assert len(CHARSETS["uppercase"]) == 26
    assert len(CHARSETS["digits"]) == 10


def test_resolve_codepoints_by_name():
    cps = _resolve_codepoints("uppercase")
    assert cps[0] == ord("A")
    assert cps[-1] == ord("Z")


def test_resolve_codepoints_by_list():
    cps = _resolve_codepoints([65, 66, 67])
    assert cps == [65, 66, 67]


def test_resolve_codepoints_unknown():
    with pytest.raises(ValueError, match="Unknown charset"):
        _resolve_codepoints("invalid")


def test_make_label_map():
    lm = _make_label_map([65, 66, 67])
    assert lm == {65: 0, 66: 1, 67: 2}


def test_datamodule_diffusion_setup(fake_font2dataset_dir: Path):
    dm = FontDataModule(
        real_data_dir=fake_font2dataset_dir,
        mode="diffusion",
        charset=[ord("A"), ord("B"), ord("C"), ord("D"), ord("E")],
        image_size=64,
        batch_size=2,
        val_split=0.34,
        test_split=0.34,
        num_workers=0,
        seed=0,
    )
    dm.setup()

    assert dm.num_classes == 5

    batch = next(iter(dm.train_dataloader()))
    assert batch["image"].shape == (2, 3, 64, 64)
    assert batch["image"].dtype == torch.float32
    assert batch["image"].min() >= 0.0 and batch["image"].max() <= 1.0
    assert "label" in batch and "char" in batch and "font_family" in batch


def test_datamodule_scorer_setup(fake_font2dataset_dir: Path, tmp_path: Path):
    # generated データ用のダミーディレクトリを作成
    import pandas as pd
    from PIL import Image

    gen_dir = tmp_path / "generated"
    gen_images = gen_dir / "images"
    gen_images.mkdir(parents=True)
    records = []
    for cp in range(ord("A"), ord("A") + 5):
        fname = f"{cp:04X}_generated_000.png"
        Image.new("RGB", (64, 64)).save(gen_images / fname)
        records.append({
            "file": fname,
            "char": chr(cp),
            "unicode": f"U+{cp:04X}",
            "codepoint": cp,
            "font_path": "ckpt",
            "font_family": "generated",
            "font_style": "cond_scale_6.0",
        })
    pd.DataFrame(records).to_parquet(gen_dir / "metadata.parquet", index=False)

    dm = FontDataModule(
        real_data_dir=fake_font2dataset_dir,
        gen_data_dir=gen_dir,
        mode="scorer",
        charset=[ord("A"), ord("B"), ord("C"), ord("D"), ord("E")],
        image_size=64,
        batch_size=2,
        num_workers=0,
        seed=0,
    )
    dm.setup()

    assert dm.num_classes == 2  # scorer: real/gen binary
    batch = next(iter(dm.train_dataloader()))
    assert batch["label"].max() <= 1 and batch["label"].min() >= 0


def test_datamodule_scorer_requires_gen_dir():
    with pytest.raises(ValueError, match="gen_data_dir"):
        FontDataModule(real_data_dir="dummy", mode="scorer")
