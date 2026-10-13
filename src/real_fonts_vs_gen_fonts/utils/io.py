from __future__ import annotations

from pathlib import Path

import pandas as pd


def save_metadata_parquet(records: list[dict], output_dir: Path) -> Path:
    """レコードリストを metadata.parquet として output_dir に保存する。

    generate_fonts.py / guided_sample.py など、画像生成スクリプトで共通して使う。

    Args:
        records:    メタデータの dict リスト。各 dict が1行に対応する。
        output_dir: 保存先ディレクトリ（images/ と同階層）。
    Returns:
        保存した parquet ファイルのパス。
    """
    meta_path = output_dir / "metadata.parquet"
    pd.DataFrame(records).to_parquet(meta_path, index=False)
    return meta_path
