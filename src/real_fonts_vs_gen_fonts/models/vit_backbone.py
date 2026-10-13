"""torchvision VisionTransformer をスクラッチ学習向けに構築する薄いラッパー。

`RealnessScorer` の ResNet 系 backbone に対する ViT 選択肢。事前学習重みは使わない
（config はいずれも `pretrained: false`）ため、torchvision の 224px 固定・ImageNet 事前学習
重みという通常の制約は問題にならない。`image_size`/`patch_size` を自由に選べるので、
`patch_size=8` かつ `image_size=96` で 12x12=144 トークンにし、`analyze_occlusion.py` の
`--patch 8 --stride 8` と幾何を一致させることが主な狙い（`attention_maps.py`/
`scripts/analyze_attention.py` 参照）。
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.nn as nn
from torchvision.models import VisionTransformer


@dataclass(frozen=True)
class ViTSpec:
    """torchvision `VisionTransformer` のハイパーパラメータ 1 セット。"""

    patch_size: int
    num_layers: int
    num_heads: int
    hidden_dim: int
    mlp_dim: int
    dropout: float = 0.0
    attention_dropout: float = 0.0


VIT_SPECS: dict[str, ViTSpec] = {
    "vit_tiny_p8": ViTSpec(patch_size=8, num_layers=6, num_heads=3, hidden_dim=192, mlp_dim=768),
    "vit_small_p8": ViTSpec(patch_size=8, num_layers=8, num_heads=6, hidden_dim=384, mlp_dim=1536),
    "vit_small_p16": ViTSpec(patch_size=16, num_layers=8, num_heads=6, hidden_dim=384, mlp_dim=1536),
}


def build_vit(name: str, image_size: int) -> VisionTransformer:
    """`VIT_SPECS[name]` から `VisionTransformer` を構築し、`heads` を `Identity` にする。

    `net.hidden_dim`（`RealnessScorer.feature_dim` に相当）を持つ状態、`net(x)` が
    CLS トークン `(B, hidden_dim)` を返す状態（ResNet の `fc=Identity` と同じ役割）で返す。

    Args:
        name:       `VIT_SPECS` のキー。
        image_size: 入力解像度（px）。`patch_size` で割り切れる必要がある
                    （割り切れないと `VisionTransformer` 側が内部で切り捨てて
                    パッチグリッドと occlusion のパッチ幾何が暗黙にずれるため、
                    ここで明示的に弾く）。
    """
    if name not in VIT_SPECS:
        raise ValueError(f"Unknown ViT spec '{name}'. Supported: {sorted(VIT_SPECS)}")
    spec = VIT_SPECS[name]
    if image_size % spec.patch_size != 0:
        raise ValueError(
            f"image_size={image_size} is not divisible by patch_size={spec.patch_size} "
            f"for ViT spec '{name}'."
        )
    net = VisionTransformer(
        image_size=image_size,
        patch_size=spec.patch_size,
        num_layers=spec.num_layers,
        num_heads=spec.num_heads,
        hidden_dim=spec.hidden_dim,
        mlp_dim=spec.mlp_dim,
        dropout=spec.dropout,
        attention_dropout=spec.attention_dropout,
        num_classes=spec.hidden_dim,
    )
    net.heads = nn.Identity()
    return net


def grid_size(name: str, image_size: int) -> int:
    """1 辺あたりのトークン数（パッチグリッドが正方形である前提、CLS は含まない）。"""
    spec = VIT_SPECS[name]
    return image_size // spec.patch_size


def _stage_layer_indices(num_layers: int) -> list[int]:
    """`num_layers` 層のうち、4 段（layer1〜4 相当）に対応するブロックの 0-based index。

    深さ d の ⌈d/4⌉, ⌈d/2⌉, ⌈3d/4⌉, d 番目のブロック出力を ResNet の layer1〜4 に見立てる
    （`RealnessScorer.extract_intermediate` の parquet 契約 `l1_`〜`l4_` を ViT でも維持し、
    `analyze_scorer_features.py` 等の下流を無改造で動かすための互換戦略）。単調増加を保証し、
    衝突（浅い num_layers で同じ index が複数回選ばれる）した場合は 1 つずつ繰り下げる。
    """
    idxs = [max(0, round(num_layers * i / 4) - 1) for i in range(1, 5)]
    for k in range(1, len(idxs)):
        if idxs[k] <= idxs[k - 1]:
            idxs[k] = min(idxs[k - 1] + 1, num_layers - 1)
    return idxs


def stage_block_index(num_layers: int, name: str) -> int:
    """"layer1"〜"layer4" を `net.encoder.layers` の 0-based ブロック index に変換する。

    `stage_tokens` と同じ `_stage_layer_indices` を使うため対応関係が一致する
    （`attention_maps.token_grad_cam` が Grad-CAM 対象ブロックを選ぶのに使う）。
    """
    idxs = _stage_layer_indices(num_layers)
    mapping = {f"layer{i + 1}": idx for i, idx in enumerate(idxs)}
    if name not in mapping:
        raise ValueError(f"Unknown stage '{name}'. Use one of {sorted(mapping)}")
    return mapping[name]


def stage_tokens(net: VisionTransformer, x: torch.Tensor) -> dict[str, torch.Tensor]:
    """4 段に対応するブロック出力（CLS 込みのトークン列）を返す。

    Args:
        net: `build_vit` で構築した `VisionTransformer`。
        x:   入力画像 (B, C, H, W)。
    Returns:
        dict with keys "layer1"〜"layer4"（各 (B, 1+N_patches, hidden_dim)）。
        `hidden_dim` は ViT の全層で共通（ResNet のように段ごとに増えない）。
    """
    blocks = list(net.encoder.layers)
    idxs = _stage_layer_indices(len(blocks))
    captured: dict[int, torch.Tensor] = {}
    handles = []
    for i in idxs:
        def _hook(module: nn.Module, inp: tuple, out: torch.Tensor, i: int = i) -> None:
            captured[i] = out

        handles.append(blocks[i].register_forward_hook(_hook))
    try:
        net(x)
    finally:
        for h in handles:
            h.remove()
    return {f"layer{k + 1}": captured[i] for k, i in enumerate(idxs)}


@contextmanager
def capture_attention(net: VisionTransformer) -> Iterator[dict[int, torch.Tensor]]:
    """`with` ブロック内の `net(x)` 呼び出しで、各ブロックの attention weight を集める。

    torchvision の `EncoderBlock.forward` は
    `x, _ = self.self_attention(x, x, x, need_weights=False)` で attention weight を
    捨てるため、各ブロックの `self_attention.forward`（`nn.MultiheadAttention.forward`）を
    一時的にラップし `need_weights=True, average_attn_weights=False` を強制して
    `(B, num_heads, N, N)`（N = 1 + パッチ数、CLS 込み）を層ごとに収集する。`with` を抜けると
    元の `forward` に戻す。

    Usage:
        with capture_attention(net) as attn:
            net(x)
        attn[layer_idx]  # (B, heads, N, N)

    WHY a manual re-implementation instead of `need_weights=True`: `nn.MultiheadAttention`
    computes its actual output via a fused SDPA kernel and only computes the *returned*
    `attn_output_weights` through a separate, parallel path for reporting purposes — that
    tensor has a `grad_fn` but is numerically disconnected from the graph that produces the
    model's output, so `.grad` on it is always `None` after backward (verified empirically:
    `chefer_relevance`/`token_grad_cam` returned all-zero maps until this was found). Manually
    recomputing scaled dot-product attention with explicit matmul + softmax keeps the weights
    on the actual forward path, so gradients flow into them.
    """
    store: dict[int, torch.Tensor] = {}
    blocks = list(net.encoder.layers)
    originals: list[tuple[nn.Module, object]] = []
    for idx, block in enumerate(blocks):
        mha = block.self_attention
        orig_forward = mha.forward

        def _wrapper(query, key, value, mha=mha, idx=idx, **kwargs):
            B, N, D = query.shape
            H = mha.num_heads
            head_dim = D // H
            qkv = nn.functional.linear(query, mha.in_proj_weight, mha.in_proj_bias)
            q, k, v = qkv.chunk(3, dim=-1)
            q, k, v = (t.view(B, N, H, head_dim).transpose(1, 2) for t in (q, k, v))
            scores = (q @ k.transpose(-2, -1)) / (head_dim**0.5)
            attn = scores.softmax(dim=-1)  # (B, H, N, N)
            out = (attn @ v).transpose(1, 2).reshape(B, N, D)
            out = mha.out_proj(out)
            store[idx] = attn
            return out, attn

        mha.forward = _wrapper
        originals.append((mha, orig_forward))
    try:
        yield store
    finally:
        for module, orig_forward in originals:
            module.forward = orig_forward
