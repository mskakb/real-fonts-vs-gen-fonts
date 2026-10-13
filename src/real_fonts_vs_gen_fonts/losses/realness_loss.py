from __future__ import annotations

import torch
import torch.nn.functional as F

LABEL_SMOOTHING: float = 0.1


def smooth_labels(
    labels: torch.Tensor,
    smoothing: float = LABEL_SMOOTHING,
) -> torch.Tensor:
    """ハードラベルをソフトラベルに変換する。

    1 → (1 - smoothing),  0 → smoothing

    Args:
        labels:    (B,) float {0.0, 1.0}
        smoothing: ラベル平滑化の強度。0 でハードラベルと同等。
    Returns:
        soft_labels: (B,) ∈ [smoothing, 1 - smoothing]
    """
    return labels * (1.0 - smoothing) + (1.0 - labels) * smoothing


def realness_loss(
    logits: torch.Tensor,
    soft_labels: torch.Tensor,
) -> torch.Tensor:
    """ソフトラベルに対する BCE 損失を返す（分類モード）。

    Args:
        logits:      (B,) スコアラーの forward_logits() 出力。
        soft_labels: (B,) smooth_labels() のソフトラベル出力。
    Returns:
        スカラー損失。
    """
    return F.binary_cross_entropy_with_logits(logits, soft_labels)


def ranknet_loss(
    score_real: torch.Tensor,
    score_gen: torch.Tensor,
    p_ij: float = 1.0,
) -> torch.Tensor:
    """RankNet ペアワイズランキング損失。

    Burges et al., "Learning to Rank Using Gradient Descent", ICML 2015, Eq. (3):

        C_ij = -P̄_ij * o_ij + log(1 + exp(o_ij))
        where o_ij = score_real - score_gen

    P̄_ij は real が gen より上位である既知確率。
    p_ij=1.0（real > gen が確定）の場合:
        C_ij = log(1 + exp(-o_ij)) = softplus(-o_ij)

    log(1 + exp(o_ij)) の数値安定な計算に F.softplus を使用する。

    Args:
        score_real: 実フォントのロジット (B,)。forward_logits() の出力。
        score_gen:  生成フォントのロジット (B,)。forward_logits() の出力。
        p_ij:       real が gen より上位である既知確率 P̄_ij ∈ [0, 1]。通常 1.0。
    Returns:
        スカラー損失。
    """
    o_ij = score_real - score_gen
    return (-p_ij * o_ij + F.softplus(o_ij)).mean()
