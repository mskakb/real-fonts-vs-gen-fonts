"""ViT Scorer の判断根拠マップ群（attention ベース + 対照手法）。

attention weight は feature importance の忠実な説明ではないことが繰り返し示されている
（Jain & Wallace 2019 / Serrano & Smith 2019 / Wiegreffe & Pinter 2019）。加えて ViT には
*register* 現象（Darcet et al. 2024）があり、情報の少ない背景トークンが高ノルムの外れ値になり
attention を吸うことがある。本モジュールは「attention をそのまま説明として使う」のではなく、
複数の attribution 手法 + 対照（勾配ベース・ノルムベース）を並べ、`scripts/analyze_attention.py`
で occlusion sensitivity（因果的な正解に最も近い基準）と突き合わせて**忠実性を検証する**ための
材料を揃える。

すべての関数は `RealnessScorer`（ViT backbone、`patch_size` で `image_size` を割り切れる構成）を
受け取り、CLS トークンを除いた `(B, N_patches)` を返す（呼び出し側で `(gh, gw)` に reshape する）。
"""

from __future__ import annotations

import torch

from real_fonts_vs_gen_fonts.models.scorer import RealnessScorer
from real_fonts_vs_gen_fonts.models.vit_backbone import (
    capture_attention,
    stage_block_index,
    stage_tokens,
)


def _require_vit(scorer: RealnessScorer) -> None:
    if not scorer._is_vit:
        raise ValueError(
            "attention_maps functions require a ViT backbone "
            "(RealnessScorer._is_vit == False for this scorer)."
        )


@torch.no_grad()
def cls_attention(scorer: RealnessScorer, x: torch.Tensor, layer: int = -1) -> torch.Tensor:
    """最終（既定）または指定ブロックの CLS→patch attention（ヘッド平均）。

    最も素朴な attribution。忠実性の下限候補として扱う（attention weight は必ずしも
    feature importance を反映しない、というモジュール docstring の懸念が最も直接的に
    当てはまる手法）。

    Args:
        scorer: ViT backbone の RealnessScorer。
        x:      入力画像 (B, C, H, W)。
        layer:  対象ブロックの 0-based index。負値は末尾から数える（既定 -1 = 最終層）。
    Returns:
        (B, N_patches)、正規化なし（attention weight は既に非負・行和1）。
    """
    _require_vit(scorer)
    net = scorer.backbone
    scorer.eval()
    with capture_attention(net) as store:
        net(x)
    num_layers = len(store)
    idx = layer if layer >= 0 else num_layers + layer
    attn = store[idx].mean(dim=1)  # (B, N, N), average over heads
    return attn[:, 0, 1:]  # CLS row, drop CLS column


@torch.no_grad()
def attention_rollout(scorer: RealnessScorer, x: torch.Tensor) -> torch.Tensor:
    """Attention rollout（Abnar & Zuidema 2020）。

    各層の attention（ヘッド平均）に `0.5*A + 0.5*I` を適用して行正規化し、層方向に積を取る
    ことで、複数層を跨いだ「情報がどこから来たか」を近似する。単層の attention より層間の
    合成を考慮する分、忠実性が上がると期待される。

    Returns:
        (B, N_patches)。
    """
    _require_vit(scorer)
    net = scorer.backbone
    scorer.eval()
    with capture_attention(net) as store:
        net(x)
    num_layers = len(store)
    B, _, N, _ = store[0].shape
    device = x.device
    eye = torch.eye(N, device=device).unsqueeze(0).expand(B, N, N)
    rollout = eye
    for layer_idx in range(num_layers):
        A = store[layer_idx].mean(dim=1)  # (B, N, N)
        A_hat = 0.5 * A + 0.5 * eye
        A_hat = A_hat / A_hat.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        rollout = torch.bmm(A_hat, rollout)
    return rollout[:, 0, 1:]


def chefer_relevance(
    scorer: RealnessScorer, x: torch.Tensor, target: str = "logit"
) -> torch.Tensor:
    """Chefer et al. (2021) の勾配加重 relevance rollout。

    各層で `E_heads[(grad_A ⊙ A)^+]` を計算し、`R <- R + contrib @ R` で層方向に合成する
    （"Transformer Interpretability Beyond Attention Visualization" の単一モダリティ版）。
    attention 単体より判別スコアへの勾配を織り込む分、忠実性が高いと期待される最有力候補。

    Args:
        target: 逆伝播する量。"logit"（既定、未飽和で高感度）または "score"（sigmoid 後）。
    Returns:
        (B, N_patches)。勾配を要するため `torch.no_grad()` の外で呼ぶこと。
    """
    _require_vit(scorer)
    net = scorer.backbone
    scorer.eval()
    with capture_attention(net) as store:
        logits = scorer.forward_logits(x)
        for w in store.values():
            w.retain_grad()
        scorer.zero_grad()
        val = logits if target == "logit" else torch.sigmoid(logits)
        val.sum().backward()

    num_layers = len(store)
    B, _, N, _ = store[0].shape
    device = x.device
    R = torch.eye(N, device=device).unsqueeze(0).expand(B, N, N).clone()
    for layer_idx in sorted(store):
        A = store[layer_idx].detach()
        grad = store[layer_idx].grad
        if grad is None:
            continue
        contrib = (grad * A).clamp(min=0).mean(dim=1)  # average over heads, (B, N, N)
        R = R + torch.bmm(contrib, R)
    return R[:, 0, 1:].detach()


def token_grad_cam(
    scorer: RealnessScorer, x: torch.Tensor, layer: str = "layer4", target: str = "logit"
) -> torch.Tensor:
    """トークン版 Grad-CAM（既存 `models/grad_cam.py` の CNN 版と同じ重み付けをトークンに適用）。

    指定ブロックのトークン出力 `(B, N, D)` を活性化とみなし、判別スコアに対する勾配を
    トークン方向に GAP してチャネル重みにする（CNN Grad-CAM の空間 GAP → トークン平均 GAP に
    対応）。attention に一切依存しないため、attention ベースの手法群との**対照**として使う
    （attention と乖離するなら、attention は本当に判断根拠を反映していない可能性が高い）。

    Returns:
        (B, N_patches)、非負・サンプルごと [0, 1] に正規化済み。CNN 版 Grad-CAM と同様、
        重み付き和が全トークンで非正になると ReLU で全 0 に潰れうる（未学習/弱学習モデルで
        発生しやすい既知の退化。バグではなく手法の性質——`vit_backbone.capture_attention`
        の勾配疎通は別途 `chefer_relevance` で検証済み）。
    """
    _require_vit(scorer)
    net = scorer.backbone
    idx = stage_block_index(len(net.encoder.layers), layer)
    block = list(net.encoder.layers)[idx]

    captured: dict[str, torch.Tensor] = {}

    def _hook(module: torch.nn.Module, inp: tuple, out: torch.Tensor) -> None:
        out.retain_grad()
        captured["tokens"] = out

    handle = block.register_forward_hook(_hook)
    scorer.eval()
    try:
        logits = scorer.forward_logits(x)
        scorer.zero_grad()
        val = logits if target == "logit" else torch.sigmoid(logits)
        val.sum().backward()
    finally:
        handle.remove()

    tokens = captured["tokens"].detach()  # (B, N, D)
    grad = captured["tokens"].grad  # (B, N, D)
    weights = grad.mean(dim=1, keepdim=True)  # GAP over tokens, (B, 1, D)
    cam = torch.relu((weights * tokens).sum(dim=-1))  # (B, N)
    cam = cam[:, 1:]  # drop CLS

    lo = cam.min(dim=1, keepdim=True).values
    hi = cam.max(dim=1, keepdim=True).values
    return (cam - lo) / (hi - lo + 1e-8)


@torch.no_grad()
def token_norm_map(scorer: RealnessScorer, x: torch.Tensor, layer: str = "layer4") -> torch.Tensor:
    """パッチトークンの L2 ノルム（register 診断用）。

    Darcet et al. (2024) の *register* 現象では、情報量の乏しい背景トークンが異常に高い
    ノルムを持つ外れ値になり、attention を独占して他のトークンからの重みを奪う。この関数を
    常時出力しておくことで、attention 系マップの解釈が有効かを目視で確認できる
    （背景に高ノルムの外れ値が集中していれば、attention ベースの説明は疑ってかかる必要がある）。

    Returns:
        (B, N_patches)。
    """
    _require_vit(scorer)
    tokens = stage_tokens(scorer.backbone, x)[layer]  # (B, N, D)
    return tokens[:, 1:, :].norm(dim=-1)
