"""End-to-end pipeline integration test.

train_diffusion → generate_fonts → train_scorer の 3 ステップを
小規模データ・小規模モデルで通しで実行する。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch
from lightning import Trainer
from PIL import Image

from real_fonts_vs_gen_fonts.datamodule import FontDataModule
from real_fonts_vs_gen_fonts.models.diffusion_module import DiffusionModule
from real_fonts_vs_gen_fonts.models.scorer_module import ScorerModule


IMAGE_SIZE = 16
CHARSET = [ord("A"), ord("B"), ord("C")]
NUM_CLASSES = len(CHARSET)
N_FAMILIES = 10   # font_family ベースの分割で train/val 両方に十分な量を確保


@pytest.fixture()
def real_data_dir(tmp_path: Path) -> Path:
    """A / B / C 3 文字 × N_FAMILIES フォントの最小 font2dataset ディレクトリ。"""
    images_dir = tmp_path / "real" / "images"
    images_dir.mkdir(parents=True)
    records = []
    for cp in CHARSET:
        for i in range(N_FAMILIES):
            family = f"Font{i:02d}"
            fname = f"{cp:04X}_{family}_000.png"
            Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color=200).save(images_dir / fname)
            records.append({
                "file": fname,
                "char": chr(cp),
                "unicode": f"U+{cp:04X}",
                "codepoint": cp,
                "font_path": f"/fake/{family}.ttf",
                "font_family": family,
                "font_style": "Regular",
            })
    pd.DataFrame(records).to_parquet(tmp_path / "real" / "metadata.parquet", index=False)
    return tmp_path / "real"


def _make_trainer(tmp_path: Path, name: str, max_steps: int = 3) -> Trainer:
    return Trainer(
        max_steps=max_steps,
        check_val_every_n_epoch=1,  # エポックごとに検証（steps ベースより安全）
        log_every_n_steps=1,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        default_root_dir=str(tmp_path / name),
    )


@pytest.fixture()
def trained_diffusion_ckpt(real_data_dir: Path, tmp_path: Path) -> Path:
    """Step 1: DiffusionModule を数ステップ学習してチェックポイントを返す。"""
    dm = FontDataModule(
        real_data_dir=real_data_dir,
        mode="diffusion",
        charset=CHARSET,
        image_size=IMAGE_SIZE,
        batch_size=4,
        val_split=0.2,
        test_split=0.1,
        num_workers=0,
        seed=0,
    )
    model = DiffusionModule(
        dim=8,
        num_classes=NUM_CLASSES,
        dim_mults=(1, 2),
        image_size=IMAGE_SIZE,
        timesteps=10,
        sampling_timesteps=5,
        lr_total_steps=3,
        lr_warmup_steps=1,
        sample_every_n_val_checks=0,
    )
    trainer = _make_trainer(tmp_path, "diffusion_out")
    trainer.fit(model, dm)

    ckpt_path = tmp_path / "diffusion_ckpts" / "last.ckpt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(ckpt_path)
    return ckpt_path


@pytest.fixture()
def gen_data_dir(trained_diffusion_ckpt: Path, tmp_path: Path) -> Path:
    """Step 2: チェックポイントから生成フォントを作成して返す。"""
    output_dir = tmp_path / "generated"
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True)

    module = DiffusionModule.load_from_checkpoint(
        trained_diffusion_ckpt,
        sampling_timesteps=5,
    )
    module.eval()

    records = []
    with torch.no_grad():
        classes = torch.tensor([0, 1, 2])
        samples = module.diffusion.sample(classes, cond_scale=1.0)

    for i, (cp, img_t) in enumerate(zip(CHARSET, samples)):
        fname = f"{cp:04X}_generated_{i:06d}.png"
        arr = (img_t.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
        Image.fromarray(arr, mode="RGB").save(images_dir / fname)
        records.append({
            "file": fname,
            "char": chr(cp),
            "unicode": f"U+{cp:04X}",
            "codepoint": cp,
            "font_path": str(trained_diffusion_ckpt),
            "font_family": "generated",
            "font_style": "cond_scale_1.0",
        })

    pd.DataFrame(records).to_parquet(output_dir / "metadata.parquet", index=False)
    return output_dir


def test_step1_diffusion_checkpoint_exists(trained_diffusion_ckpt: Path):
    """Step 1: チェックポイントが生成されているか。"""
    assert trained_diffusion_ckpt.exists()


def test_step2_generated_data_exists(gen_data_dir: Path):
    """Step 2: 生成画像と metadata.parquet が存在するか。"""
    assert (gen_data_dir / "metadata.parquet").exists()
    assert len(list((gen_data_dir / "images").glob("*.png"))) == NUM_CLASSES


def test_step3_scorer_training(real_data_dir: Path, gen_data_dir: Path, tmp_path: Path):
    """Step 3: Scorer が実/生成フォントで正常に学習できるか。"""
    dm = FontDataModule(
        real_data_dir=real_data_dir,
        gen_data_dir=gen_data_dir,
        mode="scorer",
        charset=CHARSET,
        image_size=IMAGE_SIZE,
        batch_size=2,
        val_split=0.2,
        test_split=0.1,
        num_workers=0,
        seed=0,
    )
    model = ScorerModule(pretrained=False, lr_total_steps=3, lr_warmup_steps=1)
    trainer = _make_trainer(tmp_path, "scorer_out")
    trainer.fit(model, dm)
    assert "val/auroc" in trainer.logged_metrics


def test_full_pipeline(
    trained_diffusion_ckpt: Path,
    gen_data_dir: Path,
    real_data_dir: Path,
    tmp_path: Path,
):
    """全 3 ステップが順番に成功するか確認する統合テスト。"""
    assert trained_diffusion_ckpt.exists(), "Step 1 failed: no checkpoint"
    assert (gen_data_dir / "metadata.parquet").exists(), "Step 2 failed: no metadata"

    dm = FontDataModule(
        real_data_dir=real_data_dir,
        gen_data_dir=gen_data_dir,
        mode="scorer",
        charset=CHARSET,
        image_size=IMAGE_SIZE,
        batch_size=2,
        val_split=0.2,
        test_split=0.1,
        num_workers=0,
        seed=0,
    )
    model = ScorerModule(pretrained=False, lr_total_steps=3, lr_warmup_steps=1)
    trainer = _make_trainer(tmp_path, "full_pipeline_out")
    trainer.fit(model, dm)
    assert "val/auroc" in trainer.logged_metrics
