"""複数 Scorer チェックポイント横断で評価・解析スイート一式を一括実行するオーケストレータ。

extract_scorer_features.py / analyze_scorer_features.py / analyze_frequency.py /
analyze_occlusion.py / visualize_occlusion.py / visualize_gradcam.py /
analyze_attention.py を subprocess として呼び出す薄い実行フロー層。各スクリプトの内部
ロジックは一切変更しない。ViT backbone 専用/非対応のステップ（analyze_attention /
visualize_gradcam）はチェックポイントの config.yaml から backbone を判定し自動スキップする。

設定の真実は --config で渡す YAML（例: config/eval/eval_suite.yaml）。
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

import yaml

from real_fonts_vs_gen_fonts.models.vit_backbone import VIT_SPECS
from real_fonts_vs_gen_fonts.utils.config import load_config, save_run_info

SCRIPTS_DIR = Path(__file__).parent

# ステップの実行順序。extract_features が生成する parquet に依存するステップは
# それより後に置く（依存が満たされなければ skip として記録し処理は継続する）。
STEP_ORDER = [
    "extract_features",
    "analyze_features",
    "analyze_frequency",
    "analyze_occlusion",
    "visualize_occlusion",
    "visualize_gradcam",
    "analyze_attention",
]

STEP_OUTPUT_SUBDIR = {
    "extract_features": "features",
    "analyze_features": "feature_analysis",
    "analyze_frequency": "frequency_analysis",
    "analyze_occlusion": "occlusion_batch",
    "visualize_occlusion": "occlusion_per_char",
    "visualize_gradcam": "gradcam_per_char",
    "analyze_attention": "attention_analysis",
}

# step -> (script filename, parquet 特徴量が必要か, backbone 制約)
STEP_SPECS = {
    "extract_features": ("extract_scorer_features.py", False, "any"),
    "analyze_features": ("analyze_scorer_features.py", True, "any"),
    "analyze_frequency": ("analyze_frequency.py", True, "any"),
    "analyze_occlusion": ("analyze_occlusion.py", True, "any"),
    "visualize_occlusion": ("visualize_occlusion.py", False, "any"),
    "visualize_gradcam": ("visualize_gradcam.py", False, "non_vit_only"),
    "analyze_attention": ("analyze_attention.py", False, "vit_only"),
}


def _bool_flag(step_cfg: dict, key: str, true_flag: str, false_flag: str | None = None) -> list[str]:
    """step_cfg[key] (bool) を --flag / --no-flag に変換する。"""
    value = step_cfg.get(key)
    if value is None:
        return []
    if value:
        return [true_flag]
    return [false_flag] if false_flag else []


def _base_ckpt_args(ckpt: dict) -> list[str]:
    args = ["--checkpoint", ckpt["checkpoint"], "--config", ckpt["config"]]
    if not ckpt.get("use_ema", True):
        args.append("--no-ema")
    return args


def build_argv_extract_features(ckpt, step_cfg, out_dir, **ctx) -> list[tuple[list[str], Path]]:
    cmd = [
        sys.executable, str(SCRIPTS_DIR / "extract_scorer_features.py"),
        *_base_ckpt_args(ckpt),
        "--output-dir", str(out_dir),
        "--split", str(step_cfg.get("split", "all")),
    ]
    if step_cfg.get("batch_size") is not None:
        cmd += ["--batch-size", str(step_cfg["batch_size"])]
    return [(cmd, out_dir)]


def build_argv_analyze_features(ckpt, step_cfg, out_dir, features: Path, **ctx) -> list[tuple[list[str], Path]]:
    cmd = [
        sys.executable, str(SCRIPTS_DIR / "analyze_scorer_features.py"),
        "--features", str(features),
        "--output-dir", str(out_dir),
        "--reducer", str(step_cfg.get("reducer", "tsne")),
        "--embedding-layer", str(step_cfg.get("embedding_layer", "layer4")),
        "--stats-split", str(step_cfg.get("stats_split", "all")),
    ]
    return [(cmd, out_dir)]


def build_argv_analyze_frequency(ckpt, step_cfg, out_dir, features: Path, **ctx) -> list[tuple[list[str], Path]]:
    cmd = [
        sys.executable, str(SCRIPTS_DIR / "analyze_frequency.py"),
        "--features", str(features),
        "--output-dir", str(out_dir),
        "--split", str(step_cfg.get("split", "all")),
        "--stats-split", str(step_cfg.get("stats_split", "test")),
        "--fit-split", str(step_cfg.get("fit_split", "train")),
    ]
    return [(cmd, out_dir)]


def build_argv_analyze_occlusion(ckpt, step_cfg, out_dir, features: Path, **ctx) -> list[tuple[list[str], Path]]:
    patch_sizes = step_cfg.get("patch_sizes", [8])
    results = []
    for patch in patch_sizes:
        patch_out_dir = out_dir / f"patch{patch}"
        cmd = [
            sys.executable, str(SCRIPTS_DIR / "analyze_occlusion.py"),
            *_base_ckpt_args(ckpt),
            "--features", str(features),
            "--output-dir", str(patch_out_dir),
            "--split", str(step_cfg.get("split", "test")),
            "--label", str(step_cfg.get("label", "real")),
            "--n-images", str(step_cfg.get("n_images", 1)),
            "--patch", str(patch),
            "--stride", str(patch),
            "--embedding-layer", str(step_cfg.get("embedding_layer", "layer4")),
            "--reducer", str(step_cfg.get("reducer", "pca")),
            "--pop-samples", str(step_cfg.get("pop_samples", 5000)),
            "--min-change", str(step_cfg.get("min_change", 0.02)),
        ]
        if step_cfg.get("all_chars"):
            cmd.append("--all-chars")
        results.append((cmd, patch_out_dir))
    return results


def build_argv_visualize_occlusion(ckpt, step_cfg, out_dir, **ctx) -> list[tuple[list[str], Path]]:
    cmd = [
        sys.executable, str(SCRIPTS_DIR / "visualize_occlusion.py"),
        *_base_ckpt_args(ckpt),
        "--output-dir", str(out_dir),
        "--split", str(step_cfg.get("split", "test")),
        "--n-samples", str(step_cfg.get("n_samples", 25)),
    ]
    if step_cfg.get("all_chars"):
        cmd.append("--all-chars")
    return [(cmd, out_dir)]


def build_argv_visualize_gradcam(ckpt, step_cfg, out_dir, **ctx) -> list[tuple[list[str], Path]]:
    cmd = [
        sys.executable, str(SCRIPTS_DIR / "visualize_gradcam.py"),
        *_base_ckpt_args(ckpt),
        "--output-dir", str(out_dir),
        "--split", str(step_cfg.get("split", "test")),
        "--n-samples", str(step_cfg.get("n_samples", 25)),
    ]
    if step_cfg.get("all_chars"):
        cmd.append("--all-chars")
    return [(cmd, out_dir)]


def build_argv_analyze_attention(ckpt, step_cfg, out_dir, **ctx) -> list[tuple[list[str], Path]]:
    cmd = [
        sys.executable, str(SCRIPTS_DIR / "analyze_attention.py"),
        *_base_ckpt_args(ckpt),
        "--output-dir", str(out_dir),
        "--split", str(step_cfg.get("split", "test")),
        "--label", str(step_cfg.get("label", "real")),
        "--n-images", str(step_cfg.get("n_images", 4)),
    ]
    if step_cfg.get("all_chars"):
        cmd.append("--all-chars")
    return [(cmd, out_dir)]


ARGV_BUILDERS = {
    "extract_features": build_argv_extract_features,
    "analyze_features": build_argv_analyze_features,
    "analyze_frequency": build_argv_analyze_frequency,
    "analyze_occlusion": build_argv_analyze_occlusion,
    "visualize_occlusion": build_argv_visualize_occlusion,
    "visualize_gradcam": build_argv_visualize_gradcam,
    "analyze_attention": build_argv_analyze_attention,
}


def resolve_backbone(config_path: str) -> str:
    cfg = load_config(config_path)
    return cfg["model"]["backbone"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, metavar="FILE",
                         help="eval_suite.yaml へのパス")
    parser.add_argument("--checkpoints", default=None, metavar="name1,name2",
                         help="対象チェックポイントを name で絞り込む（省略時は全件）")
    parser.add_argument("--steps", default=None, metavar="step1,step2",
                         help="対象ステップを絞り込む（省略時は enabled な全ステップ）")
    parser.add_argument("--stop-on-error", action="store_true",
                         help="いずれかのステップが失敗したら即座に中断する")
    parser.add_argument("overrides", nargs="*", metavar="key=value",
                         help="eval_suite.yaml のトップレベルキーへの上書き（例: output_root=...）")
    return parser.parse_args()


def _write_manifest(rows: list[dict], output_root: Path) -> None:
    manifest_path = output_root / "manifest.csv"
    fieldnames = ["checkpoint_name", "step", "status", "skip_reason", "duration_s",
                  "output_dir", "returncode"]
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _record(rows: list[dict], checkpoint_name: str, step: str, status: str,
            skip_reason: str | None, duration_s: float, output_dir: Path,
            returncode: int | None) -> None:
    rows.append({
        "checkpoint_name": checkpoint_name,
        "step": step,
        "status": status,
        "skip_reason": skip_reason or "",
        "duration_s": round(duration_s, 1),
        "output_dir": str(output_dir),
        "returncode": returncode if returncode is not None else "",
    })
    print(f"[{checkpoint_name}/{step}] {status}"
          + (f" ({skip_reason})" if skip_reason else f" ({duration_s:.1f}s, rc={returncode})"))


def main() -> None:
    args = parse_args()
    suite_cfg = load_config(args.config, args.overrides)
    output_root = Path(suite_cfg["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)

    with open(output_root / "config.yaml", "w") as f:
        yaml.dump(suite_cfg, f, allow_unicode=True, sort_keys=False)
    save_run_info(output_root)

    checkpoints = suite_cfg["checkpoints"]
    if args.checkpoints:
        wanted_ckpts = set(args.checkpoints.split(","))
        checkpoints = [c for c in checkpoints if c["name"] in wanted_ckpts]

    steps = [s for s in STEP_ORDER if suite_cfg["steps"].get(s, {}).get("enabled")]
    if args.steps:
        wanted_steps = set(args.steps.split(","))
        steps = [s for s in steps if s in wanted_steps]

    manifest_rows: list[dict] = []
    aborted = False

    for ckpt in checkpoints:
        if aborted:
            break
        ckpt_dir = output_root / ckpt["name"]
        backbone = resolve_backbone(ckpt["config"])
        is_vit = backbone in VIT_SPECS
        features_parquet: Path | None = None

        for step in steps:
            script_name, needs_features, vit_req = STEP_SPECS[step]
            step_cfg = suite_cfg["steps"].get(step, {})
            out_dir = ckpt_dir / STEP_OUTPUT_SUBDIR[step]

            if vit_req == "vit_only" and not is_vit:
                _record(manifest_rows, ckpt["name"], step, "skipped",
                         f"backbone={backbone} is not ViT", 0.0, out_dir, None)
                continue
            if vit_req == "non_vit_only" and is_vit:
                _record(manifest_rows, ckpt["name"], step, "skipped",
                         f"backbone={backbone} is ViT (unsupported)", 0.0, out_dir, None)
                continue
            if needs_features and features_parquet is None:
                _record(manifest_rows, ckpt["name"], step, "skipped",
                         "extract_features did not produce a parquet", 0.0, out_dir, None)
                continue

            step_failed = False
            for cmd, sub_out_dir in ARGV_BUILDERS[step](
                ckpt, step_cfg, out_dir, features=features_parquet
            ):
                t0 = time.monotonic()
                proc = subprocess.run(cmd, check=False)
                duration = time.monotonic() - t0
                status = "ok" if proc.returncode == 0 else "error"
                _record(manifest_rows, ckpt["name"], step, status, None, duration,
                        sub_out_dir, proc.returncode)
                if status == "error":
                    step_failed = True
                    if args.stop_on_error:
                        aborted = True
                        break
            if aborted:
                break

            if step == "extract_features" and not step_failed:
                features_parquet = out_dir / "scorer_features.parquet"

    _write_manifest(manifest_rows, output_root)
    print(f"\nDone. Manifest: {output_root / 'manifest.csv'}")
    if aborted:
        sys.exit(1)


if __name__ == "__main__":
    main()
