"""Occlusion sensitivity による Scorer の判断根拠可視化。

入力の一部をパッチで遮蔽し、スコアがどれだけ変化するかを総当たりで測る
(Zeiler & Fergus 2014)。勾配を使わず前向き推論のみで寄与マップを作るため、
backbone 種別（`forward_logits` さえあれば良い）にも空間解像度にも縛られない。
Grad-CAM（layer4 の低解像度に依存）が細ストロークで局在しない問題を回避する。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from real_fonts_vs_gen_fonts.models.scorer import RealnessScorer


def patch_value_map(
    positions: list[tuple[int, int]],
    values: np.ndarray,
    patch_size: int,
    shape: tuple[int, int],
) -> np.ndarray:
    """パッチごとのスカラ値を (H, W) マップに広げる。重なる領域は平均する。

    `scripts/analyze_occlusion.py` の score/delta/necessity-sufficiency マップと
    `scripts/analyze_attention.py` の attribution マップの両方で共有する。
    """
    H, W = shape
    value_map = np.zeros((H, W), dtype=np.float64)
    count_map = np.zeros((H, W), dtype=np.float64)
    for (t, l), v in zip(positions, values):
        value_map[t : t + patch_size, l : l + patch_size] += v
        count_map[t : t + patch_size, l : l + patch_size] += 1.0
    return value_map / np.clip(count_map, 1.0, None)


def cumulative_curve_batches(
    image: torch.Tensor,
    positions: list[tuple[int, int]],
    order: np.ndarray,
    patch_size: int,
    fill: float,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """`order` 順にパッチ領域を累積和集合し、deletion バッチ（原画像から順に fill で消す）と
    insertion バッチ（全面 fill から順に原画像を戻す）を構築する。

    `order` は任意の attribution（occlusion delta 降順に限らない）の降順インデックス配列で
    よく、`scripts/analyze_occlusion.py`（occlusion delta 順）と `scripts/analyze_attention.py`
    （attention 系手法の順）の両方から使う「忠実性評価（deletion/insertion AUC）」の中核処理。

    `positions` のタイル幾何は全面を覆うため（`build_occluded_batch` 参照）、
    `order` が全パッチを含む限り最終マスクは全面 True になり、deletion の終端は
    全面塗り（≈ s_blank）、insertion の終端は原画像そのもの（= s0）に一致する
    （`--min-change` で除外されたパッチは定義上 fill とほぼ同一なので、この一致は近似的に保たれる）。

    Returns:
        (del_batch, ins_batch, area_frac): 累積バッチ (n+1, C, H, W) x2 と、各累積段階で
        塗り替わった面積の割合 (n+1,)（先頭 0 番目は空集合＝deletion=原画像・insertion=全面 fill）。
    """
    C, H, W = image.shape
    n = len(order)
    device = image.device
    del_batch = image.unsqueeze(0).repeat(n + 1, 1, 1, 1).clone()
    ins_batch = torch.full((n + 1, C, H, W), fill, dtype=image.dtype, device=device)
    mask = torch.zeros(H, W, dtype=torch.bool, device=device)
    area_frac = np.zeros(n + 1, dtype=np.float64)
    for k, idx in enumerate(order, start=1):
        t, l = positions[idx]
        mask[t : t + patch_size, l : l + patch_size] = True
        del_batch[k, :, mask] = fill
        ins_batch[k, :, mask] = image[:, mask]
        area_frac[k] = float(mask.float().mean())
    return del_batch, ins_batch, area_frac


def _resolve_baseline(img: torch.Tensor, baseline: str | float) -> float:
    """遮蔽パッチの塗り値を 1 枚の画像 (C, H, W) から決める。

    "background" は外周 1px の中央値。白黒どちらが地でも、raw SDF の遠方値でも
    背景を頑健に拾える（グリフは中央寄せで外周は地のため）。
    """
    if isinstance(baseline, (int, float)):
        return float(baseline)
    if baseline == "mean":
        return float(img.mean())
    if baseline == "min":
        return float(img.min())
    if baseline == "max":
        return float(img.max())
    if baseline == "background":
        border = torch.cat([
            img[:, 0, :].reshape(-1), img[:, -1, :].reshape(-1),
            img[:, :, 0].reshape(-1), img[:, :, -1].reshape(-1),
        ])
        return float(border.median())
    raise ValueError(
        f"Unknown baseline '{baseline}'. Use 'background'/'mean'/'min'/'max' or a float."
    )


@torch.no_grad()
def occlusion_sensitivity(
    scorer: RealnessScorer,
    x: torch.Tensor,
    patch_size: int = 8,
    stride: int = 4,
    baseline: str | float = "background",
    target: str = "logit",
    chunk_size: int = 256,
) -> torch.Tensor:
    """遮蔽感度マップを返す（符号付き、正規化なし）。

    Args:
        scorer:     RealnessScorer。`forward_logits` のみ使用（backbone 非依存）。
        x:          モデル入力 (B, C, H, W)。値域は入力表現に依存（decode 済み [0,1] 等）。
        patch_size: 遮蔽パッチの一辺 (px)。
        stride:     パッチを動かす歩幅 (px)。小さいほど高解像・高コスト。
        baseline:   パッチの塗り値。"background"/"mean"/"min"/"max" または float。
        target:     "logit"（sigmoid 前）または "score"（sigmoid 後）。
        chunk_size: 遮蔽画像を前向きするミニバッチ上限。
    Returns:
        感度マップ (B, H, W)。値 = s0 - s_occ（正 = そこを隠すとスコア低下 = realness を支える領域）。
    """
    scorer.eval()
    device = x.device
    B, _, H, W = x.shape

    def _score(batch: torch.Tensor) -> torch.Tensor:
        logits = scorer.forward_logits(batch)  # (N,)
        return torch.sigmoid(logits) if target == "score" else logits

    tops = list(range(0, max(H - patch_size, 0) + 1, stride))
    lefts = list(range(0, max(W - patch_size, 0) + 1, stride))
    # 端がパッチで覆われない場合は最終位置を足して全面を保証する。
    if tops[-1] != H - patch_size:
        tops.append(H - patch_size)
    if lefts[-1] != W - patch_size:
        lefts.append(W - patch_size)

    maps = torch.zeros(B, H, W, device=device)
    for b in range(B):
        img = x[b : b + 1]  # (1, C, H, W)
        s0 = _score(img).item()
        fill = _resolve_baseline(x[b], baseline)

        positions = [(t, l) for t in tops for l in lefts]
        heat = torch.zeros(H, W, device=device)
        count = torch.zeros(H, W, device=device)

        for start in range(0, len(positions), chunk_size):
            chunk = positions[start : start + chunk_size]
            occ = img.repeat(len(chunk), 1, 1, 1)  # (n, C, H, W)
            for k, (t, l) in enumerate(chunk):
                occ[k, :, t : t + patch_size, l : l + patch_size] = fill
            delta = s0 - _score(occ)  # (n,)
            for k, (t, l) in enumerate(chunk):
                heat[t : t + patch_size, l : l + patch_size] += delta[k]
                count[t : t + patch_size, l : l + patch_size] += 1.0

        maps[b] = heat / count.clamp_min(1.0)

    return maps.cpu()


def make_blank_like(img: torch.Tensor, baseline: str | float = "background") -> torch.Tensor:
    """画像 1 枚 (C, H, W) から、全面を baseline 値で塗った (1, C, H, W) を作る。

    preservation（`mode="keep"`）は 96px のほぼ空白キャンバスに小さな断片が乗った
    分布外入力になるため、素の `s(keep)` は「断片の realness」ではなく「ほぼ空白であること」
    に支配されやすい。この全面塗り画像のスコア `s_blank` を基準に取り、
    `s(keep) - s_blank` で断片自体の寄与を切り出す。deletion/insertion 曲線の始点にも使う。
    """
    fill = _resolve_baseline(img, baseline)
    return torch.full((1, *img.shape), fill, dtype=img.dtype, device=img.device)


def build_occluded_batch(
    img: torch.Tensor,
    patch_size: int = 8,
    stride: int = 4,
    baseline: str | float = "background",
    min_change: float = 0.0,
    mode: str = "occlude",
) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    """全パッチ位置で遮蔽 (or 保持) したバッチと対応 (top, left) 位置リストを返す。

    `occlusion_sensitivity` と同一のパッチ幾何（tops/lefts の range + 端補完）を使うため、
    両者の座標系は一致する。per-patch の絶対スコア・中間特徴が欲しい呼び出し元向けの
    ヘルパ（`occlusion_sensitivity` 自体は平均 Δ マップしか返さないため）。

    `min_change > 0` のとき、元画像のパッチ領域が塗り値 `fill` とほぼ同一な位置
    （＝背景だけを覆い、遮蔽しても画像が実質変わらない no-op パッチ）を除外する。
    しきい値は入力のダイナミックレンジ (max-min) に対する割合で、raw SDF でも
    binary でも表現に依らず効く。ストロークを覆ったパッチだけを残せるため、
    「ストロークが消えたときの挙動」だけを解析したい呼び出し元に有用。

    Args:
        img:        モデル入力 1 枚 (C, H, W)。
        patch_size: 遮蔽パッチの一辺 (px)。
        stride:     パッチを動かす歩幅 (px)。
        baseline:   パッチの塗り値。"background"/"mean"/"min"/"max" または float。
        min_change: no-op パッチを除外するしきい値（入力ダイナミックレンジに対する割合、
                    0 で無効）。パッチ領域内の |img - fill| の最大値が
                    `min_change * (img.max() - img.min())` 未満なら除外する。
                    "keep" モードでもこの判定は元画像基準（= deletion と同一の除外集合）で
                    行うため、両モードの `positions` は常に一致する（per-patch 突き合わせ用）。
        mode:       "occlude"（既定、パッチ領域を fill で塗る = deletion）または
                    "keep"（パッチ領域だけ元画像を残し他は全面 fill = preservation）。
    Returns:
        (occ, positions): occ は変換後の画像バッチ (N, C, H, W)、positions は各画像に対応する
        (top, left) のリスト（順序が一致）。
    """
    if mode not in ("occlude", "keep"):
        raise ValueError(f"Unknown mode '{mode}'. Use 'occlude' or 'keep'.")

    _, H, W = img.shape
    tops = list(range(0, max(H - patch_size, 0) + 1, stride))
    lefts = list(range(0, max(W - patch_size, 0) + 1, stride))
    if tops[-1] != H - patch_size:
        tops.append(H - patch_size)
    if lefts[-1] != W - patch_size:
        lefts.append(W - patch_size)

    positions = [(t, l) for t in tops for l in lefts]
    fill = _resolve_baseline(img, baseline)

    if min_change > 0.0:
        dyn_range = float(img.max() - img.min())
        if dyn_range > 0.0:
            thresh = min_change * dyn_range
            kept = [
                (t, l) for (t, l) in positions
                if float((img[:, t : t + patch_size, l : l + patch_size] - fill).abs().max())
                >= thresh
            ]
            if kept:  # 全除外（空バッチ）は避け、その場合はフィルタ無効化
                positions = kept

    if mode == "occlude":
        occ = img.unsqueeze(0).repeat(len(positions), 1, 1, 1)  # (N, C, H, W)
        for k, (t, l) in enumerate(positions):
            occ[k, :, t : t + patch_size, l : l + patch_size] = fill
    else:
        occ = torch.full((len(positions), *img.shape), fill, dtype=img.dtype, device=img.device)
        for k, (t, l) in enumerate(positions):
            occ[k, :, t : t + patch_size, l : l + patch_size] = \
                img[:, t : t + patch_size, l : l + patch_size]
    return occ, positions


def overlay_occlusion(
    image: np.ndarray,
    heat: np.ndarray,
    alpha: float = 0.6,
    target_size: tuple[int, int] | None = None,
    vmax: float | None = None,
) -> np.ndarray:
    """符号付き感度マップを発散カラーマップで画像にオーバーレイする。

    Args:
        image:       元画像 (H, W, 3)、値域 [0, 1]。
        heat:        感度マップ (H', W')、符号付き（正規化不要）。
        alpha:       ヒートマップの不透明度。
        target_size: リサイズ先 (H, W)。None で image と同じ。
        vmax:        対称正規化の上限。None なら max|heat|（画像ごと自動）。
    Returns:
        オーバーレイ画像 (H, W, 3)、値域 [0, 1]。赤 = realness 支持 / 青 = 抑制。
    """
    import matplotlib.cm as cm

    h, w = image.shape[:2]
    size = target_size or (h, w)

    heat_t = torch.tensor(heat, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    heat_resized = F.interpolate(heat_t, size=size, mode="bilinear", align_corners=False)
    heat_np = heat_resized.squeeze(0).squeeze(0).numpy()

    scale = vmax if vmax is not None else float(np.abs(heat_np).max())
    norm = np.clip(heat_np / (scale + 1e-8), -1.0, 1.0)
    # [-1, 1] -> [0, 1] を RdBu_r に通す（0 -> 白, 正 -> 赤, 負 -> 青）。
    heatmap = cm.RdBu_r((norm + 1.0) / 2.0)[..., :3]  # (H, W, 3)
    return np.clip(alpha * heatmap + (1 - alpha) * image, 0, 1)
