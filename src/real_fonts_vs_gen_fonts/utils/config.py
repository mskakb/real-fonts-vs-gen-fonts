"""Config loading utilities shared across training scripts."""

from __future__ import annotations

import logging
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logging to emit INFO+ records to stdout.

    The project's module loggers use logging.getLogger(__name__) but no handler
    is configured by default, so their log.info() output is silently dropped.
    Call this once at the start of a training script to make those logs visible.
    Idempotent: does nothing if the root logger already has handlers.
    """
    root = logging.getLogger()
    if root.handlers:
        return
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively override base dict with values from override."""
    result = base.copy()
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Apply 'section.key=value' CLI overrides to cfg."""
    for arg in overrides:
        if "=" not in arg:
            raise ValueError(f"Override must be 'key=value' format: {arg}")
        key_path, raw_value = arg.split("=", 1)
        keys = key_path.split(".")
        try:
            value = yaml.safe_load(raw_value)
        except yaml.YAMLError:
            value = raw_value
        node = cfg
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = value
    return cfg


def load_config(config_path: str | Path, overrides: list[str] | None = None) -> dict:
    """Load YAML config and apply CLI overrides."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return apply_overrides(cfg, overrides or [])


def setup_experiment_dir(cfg: dict, extra: dict | None = None) -> Path:
    """実験ディレクトリを作成し、config.yaml と run_info.yaml を保存する。

    comment フィールドが設定されている場合、experiment_name に _{comment} を付加する。
    comment のサニタイズ（記号除去・小文字化・5文字切り捨て）も行う。

    Args:
        cfg:   load_config() で読み込んだ設定辞書。
               "experiment_name", "output_root", "comment"（省略可）を参照する。
        extra: run_info.yaml に追記する追加情報（resume パスなど）。
    Returns:
        作成した出力ディレクトリのパス。
    """
    # Prepend a timestamp so each run gets a unique output dir without the launcher
    # script needing to compute $(date) (settings live in YAML, not the shell).
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    comment = re.sub(r'[^a-zA-Z0-9]', '', str(cfg.get("comment") or "")).lower()[:5]
    base_name = f"{cfg['experiment_name']}_{comment}" if comment else cfg["experiment_name"]
    exp_name = f"{timestamp}_{base_name}"
    output_dir = Path(cfg["output_root"]) / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f, allow_unicode=True, sort_keys=False)

    run_info_extra = {"comment": comment or None}
    if extra:
        run_info_extra.update(extra)
    save_run_info(output_dir, extra=run_info_extra)

    return output_dir


def save_run_info(output_dir: str | Path, extra: dict | None = None) -> None:
    """Save run_info.yaml with argv, timestamp, and git commit to output_dir.

    Call this once per script after the output directory is created.
    extra: additional key-value pairs to include (e.g. {"resume": "/path/to.ckpt"}).
    """
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        git_commit = "unknown"

    info: dict = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "argv": sys.argv,
    }
    if extra:
        info.update(extra)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "run_info.yaml", "w") as f:
        yaml.dump(info, f, allow_unicode=True, sort_keys=False)
