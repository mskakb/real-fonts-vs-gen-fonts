from __future__ import annotations

import logging
import random
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, Literal

log = logging.getLogger(__name__)

import pandas as pd
import yaml
import torch
from lightning import LightningDataModule
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

# Predefined charsets matching font2dataset conventions.
# Each value is an ordered list of codepoints; index in the list = class label.
CHARSETS: dict[str, list[int]] = {
    "uppercase": list(range(ord("A"), ord("Z") + 1)),   # 26 classes
    "lowercase": list(range(ord("a"), ord("z") + 1)),   # 26 classes
    "digits":    list(range(ord("0"), ord("9") + 1)),   # 10 classes
    "ascii":     list(range(0x20, 0x7F)),               # 95 classes
}


def _resolve_codepoints(charset: str | list[int]) -> list[int]:
    """charset 名・文字列・コードポイントリストを正規化して返す。

    対応形式:
      - CHARSETS のキー: "uppercase", "lowercase", "digits", "ascii"
      - 文字列: "A" → [65], "AB" → [65, 66]（デバッグ用の任意サブセット指定）
      - list[int]: コードポイントの直接指定
    """
    if isinstance(charset, list):
        return charset
    if charset in CHARSETS:
        return CHARSETS[charset]
    # 文字列をコードポイントリストとして解釈（例: "AB" → [65, 66]）
    if isinstance(charset, str) and len(charset) >= 1:
        return [ord(c) for c in charset]
    raise ValueError(f"Unknown charset '{charset}'. Available: {list(CHARSETS)}")


def _make_label_map(codepoints: list[int]) -> dict[int, int]:
    """codepoint → class index のマッピングを返す。"""
    return {cp: idx for idx, cp in enumerate(codepoints)}


class FontDataset(Dataset[dict[str, Any]]):
    """フォント画像データセット。

    font2dataset の出力（images/ + metadata.parquet）を読み込む。
    label の意味は mode に依存する:
      - "diffusion": 文字クラスインデックス (charset の順序に従う)
      - "scorer":    本物度ラベル (real=1, generated=0)
    """

    def __init__(
        self,
        metadata: pd.DataFrame,
        images_dir: Path,
        transform: v2.Transform | None = None,
        image_channels: int = 3,
        binary_transform: v2.Transform | None = None,
        sdf_dir: Path | None = None,
        sdf_column: str = "sdf_png_file",
    ) -> None:
        self._meta = metadata.reset_index(drop=True)
        self._images_dir = images_dir
        self._transform = transform
        self._image_channels = image_channels
        self._binary_transform = binary_transform
        # sdf_dir が指定された場合、image は sdf/ の事前計算済み SDF PNG からロードする
        # （binary_transform 用の stroke_map は引き続き images/ の生画像から算出）
        self._sdf_dir = sdf_dir
        self._sdf_column = sdf_column

    def __len__(self) -> int:
        return len(self._meta)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self._meta.iloc[idx]
        pil_mode = "RGB" if self._image_channels == 3 else "L"
        if self._sdf_dir is not None:
            # 事前計算済み SDF PNG をロード（grayscale）。transform は binarize/SDF 計算なしの
            # Resize→float 変換のみを想定。
            src = Image.open(self._sdf_dir / row[self._sdf_column]).convert(pil_mode)
        else:
            src = Image.open(self._images_dir / row["file"]).convert(pil_mode)
        image = self._transform(src) if self._transform is not None else src
        result: dict[str, Any] = {
            "image": image,
            "label": int(row["label"]),
            "char": str(row["char"]),
            "font_family": str(row["font_family"]),
        }
        if self._binary_transform is not None:
            pil_img = Image.open(self._images_dir / row["file"]).convert(pil_mode)
            binary = self._binary_transform(pil_img)  # {0, 1}: 0=ink, 1=background
            stroke_map = 1.0 - binary                 # ink=1, background=0
            result["stroke_density"] = stroke_map.mean()
            result["stroke_map"] = stroke_map
        return result


class _MixedFontDataset(Dataset[dict[str, Any]]):
    """real / generated が混在する scorer 用データセット（分類モード）。

    metadata に _images_dir カラムを持ち、レコードごとに異なるディレクトリを参照する。
    load_sdf=True のとき、各レコードの _sdf_dir / sdf_column から事前計算済み SDF PNG を読む
    （transform 側は binarize/SDF 計算なし、必要なら decode のみ）。
    """

    def __init__(
        self,
        metadata: pd.DataFrame,
        transform: v2.Transform,
        image_channels: int = 3,
        load_sdf: bool = False,
        sdf_column: str = "sdf_png_file",
    ) -> None:
        self._meta = metadata.reset_index(drop=True)
        self._transform = transform
        self._image_channels = image_channels
        self._load_sdf = load_sdf
        self._sdf_column = sdf_column

    def __len__(self) -> int:
        return len(self._meta)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self._meta.iloc[idx]
        pil_mode = "RGB" if self._image_channels == 3 else "L"
        if self._load_sdf:
            path = Path(row["_sdf_dir"]) / row[self._sdf_column]
        else:
            path = Path(row["_images_dir"]) / row["file"]
        image = Image.open(path).convert(pil_mode)
        image = self._transform(image)
        return {
            "image": image,
            "label": int(row["label"]),
            "char": str(row["char"]),
            "font_family": str(row["font_family"]),
            "image_path": str(path),
        }


class RankingDataset(Dataset[dict[str, Any]]):
    """RankNet 学習用ペアデータセット（ランキングモード）。

    real と gen の画像ペアを返す。

    pair_sampling:
        "random":       ランダムな real + ランダムな gen（クラスをまたぐ）
        "same_char":    同じ文字コードポイントの real + gen（条件を揃えた比較）
        "in_batch_hard": データセットは "random" でペアを返す。
                         ハードマイニングは ScorerModule の training_step で実施する。
    """

    def __init__(
        self,
        real_meta: pd.DataFrame,
        gen_meta: pd.DataFrame,
        real_images_dir: Path,
        gen_images_dir: Path,
        transform: v2.Transform,
        pair_sampling: str = "same_char",
        image_channels: int = 3,
        load_sdf: bool = False,
        real_sdf_dir: Path | None = None,
        gen_sdf_dir: Path | None = None,
        sdf_column: str = "sdf_png_file",
    ) -> None:
        self._real = real_meta.reset_index(drop=True)
        self._gen = gen_meta.reset_index(drop=True)
        self._real_dir = real_images_dir
        self._gen_dir = gen_images_dir
        self._transform = transform
        self._pair_sampling = pair_sampling
        self._image_channels = image_channels
        # load_sdf=True のとき 4 枚すべて事前計算済み SDF PNG から読む（real は real_sdf_dir、
        # gen は gen_sdf_dir）。transform 側は binarize/SDF 計算なし、必要なら decode のみ。
        self._load_sdf = load_sdf
        self._real_sdf_dir = real_sdf_dir
        self._gen_sdf_dir = gen_sdf_dir
        self._sdf_column = sdf_column

        if pair_sampling == "same_char":
            # codepoint → gen インデックスのリスト
            self._gen_by_cp: dict[int, list[int]] = {}
            for i, row in self._gen.iterrows():
                self._gen_by_cp.setdefault(int(row["codepoint"]), []).append(int(i))

    def __len__(self) -> int:
        return len(self._real)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        real_row = self._real.iloc[idx]

        if self._pair_sampling == "same_char":
            cp = int(real_row["codepoint"])
            candidates = self._gen_by_cp.get(cp) or list(range(len(self._gen)))
            gen_idx = random.choice(candidates)
            gen_row = self._gen.iloc[gen_idx]
        else:
            # "random" および "in_batch_hard"（後者はステップ側で再ペアリング）
            gen_row = self._gen.iloc[random.randint(0, len(self._gen) - 1)]

        # tie ペア用: real-real / gen-gen の一貫性学習のためクラス全体からランダムに1枚ずつ選ぶ
        real_row2 = self._real.iloc[random.randint(0, len(self._real) - 1)]
        gen_row2 = self._gen.iloc[random.randint(0, len(self._gen) - 1)]

        pil_mode = "RGB" if self._image_channels == 3 else "L"
        if self._load_sdf:
            col = self._sdf_column
            real_img = Image.open(self._real_sdf_dir / real_row[col]).convert(pil_mode)
            gen_img = Image.open(self._gen_sdf_dir / gen_row[col]).convert(pil_mode)
            real_img2 = Image.open(self._real_sdf_dir / real_row2[col]).convert(pil_mode)
            gen_img2 = Image.open(self._gen_sdf_dir / gen_row2[col]).convert(pil_mode)
        else:
            real_img = Image.open(self._real_dir / real_row["file"]).convert(pil_mode)
            gen_img = Image.open(self._gen_dir / gen_row["file"]).convert(pil_mode)
            real_img2 = Image.open(self._real_dir / real_row2["file"]).convert(pil_mode)
            gen_img2 = Image.open(self._gen_dir / gen_row2["file"]).convert(pil_mode)

        return {
            "real_image": self._transform(real_img),
            "gen_image": self._transform(gen_img),
            "real_image2": self._transform(real_img2),
            "gen_image2": self._transform(gen_img2),
            "real_char": str(real_row["char"]),
            "gen_char": str(gen_row["char"]),
        }


def _load_real_meta(
    data_dir: Path,
    mode: str,
    label_map: dict[int, int],
) -> pd.DataFrame:
    meta = pd.read_parquet(data_dir / "metadata.parquet")
    meta = meta[meta["codepoint"].isin(label_map)].copy()
    if mode == "diffusion":
        meta["label"] = meta["codepoint"].map(label_map)
    else:
        meta["label"] = 1  # real
    return meta


def _load_gen_meta(
    data_dir: Path,
    label_map: dict[int, int],
) -> pd.DataFrame:
    meta = pd.read_parquet(data_dir / "metadata.parquet")
    meta = meta[meta["codepoint"].isin(label_map)].copy()
    meta["label"] = 0  # generated
    if "font_family" not in meta.columns:
        meta["font_family"] = "generated"
    return meta


def _log_split(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    label: str,
    output_dir: Path | None = None,
) -> None:
    """スプリット結果をログ出力し、output_dir が指定された場合は split.yaml に保存する。"""
    def _families(df: pd.DataFrame) -> list[str]:
        return sorted(df["font_family"].unique().tolist())

    train_fams, val_fams, test_fams = _families(train), _families(val), _families(test)

    log.info(
        "Split (%s) | train: %d families / %d samples | val: %d families / %d samples | test: %d families / %d samples",
        label,
        len(train_fams), len(train),
        len(val_fams),   len(val),
        len(test_fams),  len(test),
    )

    if output_dir is not None:
        import yaml as _yaml
        split_info = {
            "label": label,
            "train": {"n_families": len(train_fams), "n_samples": len(train), "families": train_fams},
            "val":   {"n_families": len(val_fams),   "n_samples": len(val),   "families": val_fams},
            "test":  {"n_families": len(test_fams),  "n_samples": len(test),  "families": test_fams},
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "split.yaml", "w") as f:
            _yaml.dump(split_info, f, allow_unicode=True, sort_keys=False)


def _split_by_family(
    meta: pd.DataFrame,
    val_split: float,
    test_split: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """font_family 単位で train/val/test に分割する。

    文字レベルでランダム分割すると同一フォントが複数スプリットに入りリークが起きる。
    """
    families = sorted(meta["font_family"].unique().tolist())
    rng = random.Random(seed)
    rng.shuffle(families)

    n = len(families)
    n_test = max(1, int(n * test_split))
    n_val = max(1, int(n * val_split))

    test_fams = set(families[:n_test])
    val_fams = set(families[n_test : n_test + n_val])
    train_fams = set(families[n_test + n_val :])

    return (
        meta[meta["font_family"].isin(train_fams)],
        meta[meta["font_family"].isin(val_fams)],
        meta[meta["font_family"].isin(test_fams)],
    )


def _load_split(
    split_file: Path,
    meta: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """make_split.py が生成した split.yaml からスプリットを復元する。

    Args:
        split_file: split.yaml のパス。
        meta:       全サンプルの metadata DataFrame（codepoint フィルタ済み）。
    Returns:
        (train_df, val_df, test_df) の tuple。
    """
    with open(split_file) as f:
        split = yaml.safe_load(f)

    families = split["families"]
    train_fams = set(families["train"])
    val_fams = set(families["val"])
    test_fams = set(families["test"])

    return (
        meta[meta["font_family"].isin(train_fams)],
        meta[meta["font_family"].isin(val_fams)],
        meta[meta["font_family"].isin(test_fams)],
    )


def _split_gen_by_file(
    gen_meta: pd.DataFrame,
    val_split: float,
    test_split: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """ファイル単位で生成データを train/val/test に分割する。"""
    n_total = len(gen_meta)
    n_test = max(1, int(n_total * test_split))
    n_val = max(1, int(n_total * val_split))
    shuffled = gen_meta.sample(frac=1, random_state=seed).reset_index(drop=True)
    return (
        shuffled.iloc[: n_total - n_val - n_test],
        shuffled.iloc[n_total - n_val - n_test : n_total - n_test],
        shuffled.iloc[n_total - n_test :],
    )


def _load_gen_split(
    split_file: Path,
    gen_meta: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """make_gen_split.py が生成した gen_split.yaml からスプリットを復元する。

    Args:
        split_file: gen_split.yaml のパス。
        gen_meta:   全生成サンプルの metadata DataFrame（codepoint フィルタ済み）。
    Returns:
        (train_df, val_df, test_df) の tuple。
    """
    with open(split_file) as f:
        split = yaml.safe_load(f)

    files = split["files"]
    train_files = set(files["train"])
    val_files = set(files["val"])
    test_files = set(files["test"])

    return (
        gen_meta[gen_meta["file"].isin(train_files)],
        gen_meta[gen_meta["file"].isin(val_files)],
        gen_meta[gen_meta["file"].isin(test_files)],
    )


def _balance(real: pd.DataFrame, gen: pd.DataFrame, seed: int) -> pd.DataFrame:
    """real と generated を 1:1 にアンダーサンプリングして結合する。"""
    n = min(len(real), len(gen))
    real_s = real.sample(n=n, random_state=seed)
    gen_s = gen.sample(n=n, random_state=seed)
    return pd.concat([real_s, gen_s], ignore_index=True)


class FontDataModule(LightningDataModule):
    """フォント画像の LightningDataModule。

    mode="diffusion":
        実フォントのみ。label = charset 内の文字インデックス。
        拡散モデルの学習に使う。

    mode="scorer":
        実フォント (label=1) + 生成フォント (label=0) の単体バッチ。
        分類損失で RealnessScorer を学習する。gen_data_dir が必須。

    mode="ranking":
        実フォントと生成フォントのペアバッチ {"real_image", "gen_image", ...}。
        RankNet 損失で RealnessScorer を学習する。gen_data_dir が必須。

    Args:
        real_data_dir: font2dataset の出力ディレクトリ (images/ + metadata.parquet)。
        gen_data_dir:  生成フォントのディレクトリ (scorer/ranking モード時に必須)。
        mode:          "diffusion", "scorer", または "ranking"。
        pair_sampling: ランキングモードのペアリング戦略。
                       "random": クロスクラスランダム / "same_char": 同一文字クラス /
                       "in_batch_hard": ランダムでデータ取得し training_step でハードマイニング。
        charset:       使用する文字種。CHARSETS のキーまたはコードポイントのリスト。
        image_size:    リサイズ後の正方形サイズ (px)。
        batch_size:    ミニバッチサイズ。
        val_split:     val セットの割合 (font_family 数基準)。
        test_split:    test セットの割合 (font_family 数基準)。
        num_workers:   DataLoader のワーカー数。None で CPU コア数を使用。
        seed:          スプリット再現性のためのシード。
        output_dir:    split.yaml を保存するディレクトリ。None で保存しない。
        binarize_threshold: 画像を二値化するしきい値 (デフォルト 0.5)。None で無効。
                            学習データの前処理としてのみ適用し、誘導サンプリング時には使わない。
        split_file:         make_split.py が生成した split.yaml のパス。
                            指定すると val_split / test_split / seed を無視してファイルからスプリットを復元する。
        gen_split_file:     make_gen_split.py が生成した gen_split.yaml のパス。
                            指定すると生成データのスプリットをファイルから復元する。None でランダム分割。
        sdf_source:         use_sdf=True のときの SDF の入手方法（diffusion / scorer / ranking モードで有効）。
                            "auto": data_dir に sdf/ と sdf_png_file 列があれば事前計算済み
                                    SDF をロード、無ければ on-the-fly 計算（旧データセット互換）。
                                    scorer/ranking では real/gen 双方が precomputed のときのみ
                                    precomputed を使い、片方でも欠ければ両方 on-the-fly に揃える。
                            "precomputed": 常に sdf/ からロード（無ければエラー）。
                            "compute": 常に binary_image_to_sdf で計算。
                            事前計算済み SDF は生成時の sdf_max_dist で固定されている点に注意。
        sdf_decode:         scorer/ranking モードで Scorer に渡す前に SDF をグレースケール decode する方式。
                            None: raw SDF([0,1]) をそのまま入力。"hard": 0.5 二値化。
                            "sigmoid": sdf_sharpness によるソフト二値化。"smoothstep": アンチエイリアス
                            連続グレースケール（sdf_filter_width_px / sdf_max_dist を使用）。
                            diffusion モードの FID decode（_sdf_decode）と同じ sdf.py 関数を再利用する。
        sdf_sharpness:      sdf_decode="sigmoid" のときの sharpness。
        sdf_filter_width_px: sdf_decode="smoothstep" の遷移帯の半幅 (px)。
    """

    def __init__(
        self,
        real_data_dir: str | Path,
        gen_data_dir: str | Path | None = None,
        mode: Literal["diffusion", "scorer", "ranking"] = "diffusion",
        pair_sampling: str = "same_char",
        charset: str | list[int] = "uppercase",
        image_size: int = 64,
        image_channels: int = 3,
        batch_size: int = 32,
        val_split: float = 0.15,
        test_split: float = 0.15,
        num_workers: int | None = None,
        seed: int = 42,
        output_dir: str | Path | None = None,
        binarize_threshold: float = 0.5,
        split_file: str | Path | None = None,
        gen_split_file: str | Path | None = None,
        use_sdf: bool = False,
        sdf_max_dist: float = 10.0,
        sdf_source: Literal["auto", "compute", "precomputed"] = "auto",
        sdf_decode: Literal["hard", "sigmoid", "smoothstep"] | None = None,
        sdf_sharpness: float | None = None,
        sdf_filter_width_px: float = 1.5,
        aug_scale: tuple[float, float] | None = None,
        aug_translate: tuple[float, float] | None = None,
    ) -> None:
        super().__init__()
        if mode in ("scorer", "ranking") and gen_data_dir is None:
            raise ValueError(f"gen_data_dir is required in {mode} mode")

        self.real_data_dir = Path(real_data_dir)
        self.gen_data_dir = Path(gen_data_dir) if gen_data_dir else None
        self.mode = mode
        self.pair_sampling = pair_sampling
        self.image_size = image_size
        self.image_channels = image_channels
        self.batch_size = batch_size
        self.val_split = val_split
        self.test_split = test_split
        self.num_workers = num_workers if num_workers is not None else cpu_count()
        self.seed = seed
        self.output_dir = Path(output_dir) if output_dir else None
        self.binarize_threshold = binarize_threshold
        self.split_file = Path(split_file) if split_file else None
        self.gen_split_file = Path(gen_split_file) if gen_split_file else None
        self.use_sdf = use_sdf
        self.sdf_max_dist = sdf_max_dist
        self.sdf_source = sdf_source
        self.sdf_decode = sdf_decode
        self.sdf_sharpness = sdf_sharpness
        self.sdf_filter_width_px = sdf_filter_width_px
        self.aug_scale = aug_scale
        self.aug_translate = aug_translate

        self._codepoints = _resolve_codepoints(charset)
        self._label_map = _make_label_map(self._codepoints)

        self._train: FontDataset | _MixedFontDataset | None = None
        self._val: FontDataset | _MixedFontDataset | None = None
        self._test: FontDataset | _MixedFontDataset | None = None

        # setup() が stage ごとに複数回呼ばれても設定ダンプは 1 回だけ出すためのガード
        self._config_logged = False

    def _log_config(self) -> None:
        """使用する主要パラメータ・設定を 1 ブロックにまとめてログ出力する。"""
        lines = [
            f"  mode: {self.mode}"
            + (f" (pair_sampling={self.pair_sampling})" if self.mode == "ranking" else ""),
            f"  charset: {len(self._codepoints)} codepoints / num_classes={self.num_classes}",
            f"  image_size: {self.image_size} | image_channels: {self.image_channels}",
            f"  batch_size: {self.batch_size} | num_workers: {self.num_workers}",
            f"  val_split: {self.val_split} | test_split: {self.test_split} | seed: {self.seed}",
            f"  binarize_threshold: {self.binarize_threshold}",
            f"  use_sdf: {self.use_sdf}"
            + (
                f" (sdf_max_dist={self.sdf_max_dist}, sdf_source={self.sdf_source}, "
                f"sdf_decode={self.sdf_decode}, sdf_sharpness={self.sdf_sharpness}, "
                f"sdf_filter_width_px={self.sdf_filter_width_px})"
                if self.use_sdf
                else ""
            ),
            f"  aug_scale: {self.aug_scale} | aug_translate: {self.aug_translate}",
            f"  real_data_dir: {self.real_data_dir}",
            f"  gen_data_dir: {self.gen_data_dir}",
            f"  split_file: {self.split_file} | gen_split_file: {self.gen_split_file}",
            f"  output_dir: {self.output_dir}",
        ]
        log.info("FontDataModule config:\n%s", "\n".join(lines))

    def _binary_transform(self) -> v2.Transform:
        """SDF を含まない binarize までのパイプライン。stroke_density / stroke_map の計算用。"""
        transforms: list = [
            v2.Resize((self.image_size, self.image_size)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ]
        if self.binarize_threshold is not None:
            thr = self.binarize_threshold
            transforms.append(v2.Lambda(lambda x, t=thr: (x > t).float()))
        return v2.Compose(transforms)

    def _build_transform(self, augment: bool = False) -> v2.Transform:
        transforms: list = [
            v2.Resize((self.image_size, self.image_size)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ]
        if augment and (self.aug_scale is not None or self.aug_translate is not None):
            transforms.append(
                v2.RandomAffine(
                    degrees=0,
                    translate=tuple(self.aug_translate) if self.aug_translate is not None else None,
                    scale=tuple(self.aug_scale) if self.aug_scale is not None else None,
                    fill=1.0,
                )
            )
        if self.binarize_threshold is not None:
            thr = self.binarize_threshold
            transforms.append(v2.Lambda(lambda x, t=thr: (x > t).float()))
        if self.use_sdf:
            from real_fonts_vs_gen_fonts.sdf import binary_image_to_sdf
            md = self.sdf_max_dist
            transforms.append(v2.Lambda(lambda x, md=md: binary_image_to_sdf(x, max_dist=md)))
            # binary_image_to_sdf collapses multi-channel input to 1 channel (averages then
            # restores a single channel). Re-expand to image_channels so the 3ch ResNet accepts it
            # (the precomputed-load path already loads RGB and skips this).
            if self.image_channels > 1:
                ch = self.image_channels
                transforms.append(
                    v2.Lambda(lambda x, c=ch: x.repeat(c, 1, 1) if x.shape[0] == 1 else x)
                )
        decode = self._sdf_decode_lambda()
        if decode is not None:
            transforms.append(decode)
        return v2.Compose(transforms)

    def _transform(self) -> v2.Transform:
        return self._build_transform(augment=False)

    def _sdf_decode_lambda(self) -> v2.Lambda | None:
        """sdf_decode 指定時に SDF([0,1]) をグレースケールへ decode する Lambda を返す（None なら raw）。

        diffusion モードの _sdf_decode と同じ sdf.py 関数を再利用する。
        """
        if not self.use_sdf or self.sdf_decode is None:
            return None
        from real_fonts_vs_gen_fonts.sdf import sdf_to_binary, sdf_to_smoothstep
        mode = self.sdf_decode
        if mode == "smoothstep":
            fw, md = self.sdf_filter_width_px, self.sdf_max_dist
            return v2.Lambda(
                lambda x, fw=fw, md=md: sdf_to_smoothstep(x, filter_width_px=fw, max_dist=md)
            )
        if mode == "hard":
            return v2.Lambda(lambda x: sdf_to_binary(x, sharpness=None))
        if mode == "sigmoid":
            sh = self.sdf_sharpness
            return v2.Lambda(lambda x, sh=sh: sdf_to_binary(x, sharpness=sh))
        raise ValueError(
            f"Unknown sdf_decode '{mode}'. Expected None|'hard'|'sigmoid'|'smoothstep'"
        )

    def _build_sdf_load_transform(self, augment: bool = False) -> v2.Transform:
        """事前計算済み SDF PNG をロードするためのパイプライン。

        SDF はすでに [0,1] にエンコード済みなので binarize も SDF 計算も行わない。
        augment 時の RandomAffine の fill は背景値（SDF 規約で背景=0）に合わせる。
        """
        transforms: list = [
            v2.Resize((self.image_size, self.image_size)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ]
        if augment and (self.aug_scale is not None or self.aug_translate is not None):
            transforms.append(
                v2.RandomAffine(
                    degrees=0,
                    translate=tuple(self.aug_translate) if self.aug_translate is not None else None,
                    scale=tuple(self.aug_scale) if self.aug_scale is not None else None,
                    fill=0.0,  # SDF 規約: 背景=0（binary 経路の fill=1.0 と異なる）
                )
            )
        decode = self._sdf_decode_lambda()
        if decode is not None:
            transforms.append(decode)
        return v2.Compose(transforms)

    def _use_precomputed_sdf(self, data_dir: Path, meta: pd.DataFrame) -> bool:
        """data_dir の事前計算済み SDF（sdf/）をロードすべきか判定する。"""
        if not self.use_sdf or self.sdf_source == "compute":
            return False
        sdf_dir = data_dir / "sdf"
        has_precomputed = sdf_dir.exists() and "sdf_png_file" in meta.columns
        if self.sdf_source == "precomputed":
            if not has_precomputed:
                raise FileNotFoundError(
                    f"sdf_source='precomputed' but no precomputed SDF found "
                    f"(sdf/ exists={sdf_dir.exists()}, sdf_png_file in metadata="
                    f"{'sdf_png_file' in meta.columns}) under {data_dir}"
                )
            return True
        return has_precomputed  # "auto"

    def _warn_sdf_max_dist_mismatch(self, data_dir: Path) -> None:
        """data_dir の事前計算済み SDF の生成時 max_dist と self.sdf_max_dist の不一致を警告する。"""
        cfg_path = data_dir / "config.yaml"
        if not cfg_path.exists():
            return
        with open(cfg_path) as f:
            ds_cfg = yaml.safe_load(f) or {}
        ds_max_dist = ds_cfg.get("sdf_max_dist")
        if ds_max_dist is not None and float(ds_max_dist) != float(self.sdf_max_dist):
            log.warning(
                "Precomputed SDF in %s was generated with sdf_max_dist=%s but FontDataModule "
                "sdf_max_dist=%s. Precomputed SDF is fixed at generation time; the SDF values "
                "follow the dataset's max_dist (%s).",
                data_dir, ds_max_dist, self.sdf_max_dist, ds_max_dist,
            )

    def _resolve_scorer_sdf(
        self, real_meta: pd.DataFrame, gen_meta: pd.DataFrame
    ) -> tuple[bool, v2.Transform, v2.Transform]:
        """scorer/ranking モードの SDF 経路を解決し (use_precomputed, train_tf, eval_tf) を返す。

        real/gen 双方が precomputed のときのみ precomputed を使う。auto で片方のみの場合は
        両方 on-the-fly に揃える（real/gen で経路が食い違うと不公平なため）。
        """
        if not self.use_sdf:
            return False, self._build_transform(augment=True), self._transform()

        assert self.gen_data_dir is not None
        real_pre = self._use_precomputed_sdf(self.real_data_dir, real_meta)
        gen_pre = self._use_precomputed_sdf(self.gen_data_dir, gen_meta)

        if real_pre and gen_pre:
            self._warn_sdf_max_dist_mismatch(self.real_data_dir)
            self._warn_sdf_max_dist_mismatch(self.gen_data_dir)
            log.info("Loading precomputed SDF (real + gen) for %s mode", self.mode)
            return (
                True,
                self._build_sdf_load_transform(augment=True),
                self._build_sdf_load_transform(augment=False),
            )

        if real_pre != gen_pre:
            log.warning(
                "use_sdf=True but only one of real/gen has precomputed SDF "
                "(real=%s, gen=%s); falling back to on-the-fly SDF for both.",
                real_pre, gen_pre,
            )
        return False, self._build_transform(augment=True), self._transform()

    def setup(self, stage: str | None = None) -> None:
        if not self._config_logged:
            self._log_config()
            self._config_logged = True

        real_meta = _load_real_meta(self.real_data_dir, self.mode, self._label_map)

        if self.split_file is not None:
            log.info("Loading split from %s", self.split_file)
            real_train, real_val, real_test = _load_split(self.split_file, real_meta)
        else:
            real_train, real_val, real_test = _split_by_family(
                real_meta, self.val_split, self.test_split, self.seed
            )

        _log_split(real_train, real_val, real_test, "real fonts", self.output_dir)

        if self.mode == "diffusion":
            images_dir = self.real_data_dir / "images"
            binary_tf = self._binary_transform()

            if self._use_precomputed_sdf(self.real_data_dir, real_meta):
                self._warn_sdf_max_dist_mismatch(self.real_data_dir)
                sdf_dir = self.real_data_dir / "sdf"
                log.info("Loading precomputed SDF from %s", sdf_dir)
                train_tf, eval_tf = self._build_sdf_load_transform(augment=True), self._build_sdf_load_transform(augment=False)
            else:
                sdf_dir = None
                train_tf, eval_tf = self._build_transform(augment=True), self._transform()

            self._train = FontDataset(real_train, images_dir, train_tf, self.image_channels, binary_transform=binary_tf, sdf_dir=sdf_dir)
            self._val = FontDataset(real_val, images_dir, eval_tf, self.image_channels, binary_transform=binary_tf, sdf_dir=sdf_dir)
            self._test = FontDataset(real_test, images_dir, eval_tf, self.image_channels, binary_transform=binary_tf, sdf_dir=sdf_dir)

        elif self.mode == "scorer":
            assert self.gen_data_dir is not None
            gen_meta = _load_gen_meta(self.gen_data_dir, self._label_map)

            if self.gen_split_file is not None:
                log.info("Loading gen split from %s", self.gen_split_file)
                gen_train, gen_val, gen_test = _load_gen_split(self.gen_split_file, gen_meta)
            else:
                gen_train, gen_val, gen_test = _split_gen_by_file(
                    gen_meta, self.val_split, self.test_split, self.seed
                )
            _log_split(gen_train, gen_val, gen_test, "generated fonts", output_dir=None)

            use_pre, train_tf, eval_tf = self._resolve_scorer_sdf(real_meta, gen_meta)
            real_img = str(self.real_data_dir / "images")
            gen_img = str(self.gen_data_dir / "images")
            real_sdf = str(self.real_data_dir / "sdf")
            gen_sdf = str(self.gen_data_dir / "sdf")

            def _make_mixed(
                real: pd.DataFrame,
                gen: pd.DataFrame,
                transform: v2.Transform,
            ) -> _MixedFontDataset:
                combined = _balance(
                    real.assign(_images_dir=real_img, _sdf_dir=real_sdf),
                    gen.assign(_images_dir=gen_img, _sdf_dir=gen_sdf),
                    self.seed,
                )
                return _MixedFontDataset(combined, transform, self.image_channels, load_sdf=use_pre)

            self._train = _make_mixed(real_train, gen_train, train_tf)
            self._val = _make_mixed(real_val, gen_val, eval_tf)
            self._test = _make_mixed(real_test, gen_test, eval_tf)

        else:  # ranking
            assert self.gen_data_dir is not None
            gen_meta = _load_gen_meta(self.gen_data_dir, self._label_map)

            if self.gen_split_file is not None:
                log.info("Loading gen split from %s", self.gen_split_file)
                gen_train, gen_val, gen_test = _load_gen_split(self.gen_split_file, gen_meta)
            else:
                gen_train, gen_val, gen_test = _split_gen_by_file(
                    gen_meta, self.val_split, self.test_split, self.seed
                )
            _log_split(gen_train, gen_val, gen_test, "generated fonts", output_dir=None)

            use_pre, train_tf, eval_tf = self._resolve_scorer_sdf(real_meta, gen_meta)
            real_dir = self.real_data_dir / "images"
            gen_dir = self.gen_data_dir / "images"
            real_sdf_dir = self.real_data_dir / "sdf"
            gen_sdf_dir = self.gen_data_dir / "sdf"
            ps = self.pair_sampling

            def _make_ranking(
                real: pd.DataFrame,
                gen: pd.DataFrame,
                transform: v2.Transform,
            ) -> RankingDataset:
                return RankingDataset(
                    real, gen, real_dir, gen_dir, transform, ps, self.image_channels,
                    load_sdf=use_pre, real_sdf_dir=real_sdf_dir, gen_sdf_dir=gen_sdf_dir,
                )

            self._train = _make_ranking(real_train, gen_train, train_tf)
            self._val = _make_ranking(real_val, gen_val, eval_tf)
            self._test = _make_ranking(real_test, gen_test, eval_tf)

    def _loader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        assert self._train is not None
        return self._loader(self._train, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        assert self._val is not None
        return self._loader(self._val, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        assert self._test is not None
        return self._loader(self._test, shuffle=False)

    @property
    def num_classes(self) -> int:
        """diffusion モード: charset のクラス数。scorer/ranking モード: 2 (real/gen)。"""
        return len(self._codepoints) if self.mode == "diffusion" else 2
