"""Config loading and CLI override tests."""

from __future__ import annotations

import sys


def test_diffusion_config_loads():
    sys.argv = ["train_diffusion.py"]
    from scripts.train_diffusion import load_config
    cfg, resume = load_config([])
    assert "experiment_name" in cfg
    assert "model" in cfg and "trainer" in cfg and "data" in cfg
    assert resume is None


def test_scorer_config_loads():
    sys.argv = ["train_scorer.py"]
    from scripts.train_scorer import load_config
    cfg, resume = load_config([])
    assert "model" in cfg
    assert "label_smoothing" in cfg["model"]
    assert "mixup_alpha" not in cfg["model"]


def test_generate_fonts_config_loads():
    sys.argv = ["generate_fonts.py"]
    from scripts.generate_fonts import load_config
    cfg = load_config([])
    assert "cond_scales" in cfg
    assert "n_samples_per_class" in cfg
    assert "charset" in cfg


def test_cli_override_int():
    from scripts.train_diffusion import load_config
    cfg, _ = load_config(["trainer.max_steps=999"])
    assert cfg["trainer"]["max_steps"] == 999


def test_cli_override_list():
    from scripts.generate_fonts import load_config
    cfg = load_config(["cond_scales=[1.0,2.0]"])
    assert cfg["cond_scales"] == [1.0, 2.0]


def test_cli_override_nested():
    from scripts.train_diffusion import load_config
    cfg, _ = load_config(["model.dim=128"])
    assert cfg["model"]["dim"] == 128


def test_resume_flag():
    from scripts.train_diffusion import load_config
    from pathlib import Path
    cfg, resume = load_config(["--resume", "outputs/exp/last.ckpt"])
    assert resume == Path("outputs/exp/last.ckpt")
