from __future__ import annotations

import torch
import pytest

from real_fonts_vs_gen_fonts.models.diffusion_module import DiffusionModule


@pytest.fixture()
def module() -> DiffusionModule:
    return DiffusionModule(
        dim=16,
        num_classes=4,
        dim_mults=(1, 2),
        image_size=16,
        timesteps=10,
        sampling_timesteps=5,
        lr_total_steps=100,
    )


@pytest.fixture()
def small_batch() -> dict:
    return {
        "image": torch.rand(4, 3, 16, 16),
        "label": torch.randint(0, 4, (4,)),
    }


def test_hparams_saved(module):
    assert "dim" in module.hparams
    assert "lr_total_steps" in module.hparams


def test_training_step_returns_scalar(module, small_batch):
    loss = module.training_step(small_batch, 0)
    assert loss.shape == torch.Size([])
    assert loss.item() > 0


def test_validation_step_runs(module, small_batch):
    module.validation_step(small_batch, 0)


def test_loss_with_details_shapes(module, small_batch):
    loss, per_sample, t = module._loss_with_details(
        small_batch["image"], small_batch["label"]
    )
    B = small_batch["image"].shape[0]
    assert loss.shape == torch.Size([])
    assert per_sample.shape == (B,)
    assert t.shape == (B,)
    assert t.min() >= 0 and t.max() < module.diffusion.num_timesteps


def test_return_per_sample_api(module, small_batch):
    """GaussianDiffusion.forward の return_per_sample=True が正しく動作するか。"""
    img = small_batch["image"]
    classes = small_batch["label"]

    loss_scalar = module.diffusion(img, classes=classes)
    loss, per_sample, t = module.diffusion(img, classes=classes, return_per_sample=True)

    assert loss_scalar.shape == torch.Size([])
    assert per_sample.shape == (img.shape[0],)
    assert t.shape == (img.shape[0],)
