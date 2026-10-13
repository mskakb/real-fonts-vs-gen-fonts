from __future__ import annotations

import torch
import pytest

from real_fonts_vs_gen_fonts.losses.realness_loss import (
    LABEL_SMOOTHING,
    realness_loss,
    smooth_labels,
)
from real_fonts_vs_gen_fonts.metrics.classification import make_scorer_metrics
from real_fonts_vs_gen_fonts.models.attention_maps import (
    attention_rollout,
    chefer_relevance,
    cls_attention,
    token_grad_cam,
    token_norm_map,
)
from real_fonts_vs_gen_fonts.models.grad_cam import GradCAM
from real_fonts_vs_gen_fonts.models.scorer import RealnessScorer
from real_fonts_vs_gen_fonts.models.scorer_module import ScorerModule

VIT_IMAGE_SIZE = 64  # image_batch フィクスチャに合わせる（vit_tiny_p8: patch=8, grid=8x8=64）
VIT_GRID = VIT_IMAGE_SIZE // 8


# ---------- RealnessScorer ----------

class TestRealnessScorer:
    @pytest.fixture()
    def scorer(self) -> RealnessScorer:
        return RealnessScorer(pretrained=False)

    def test_forward_shape(self, scorer, image_batch):
        out = scorer(image_batch)
        assert out.shape == (4,)

    def test_forward_range(self, scorer, image_batch):
        out = scorer(image_batch)
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_forward_logits_shape(self, scorer, image_batch):
        out = scorer.forward_logits(image_batch)
        assert out.shape == (4,)

    def test_extract_features_shape(self, scorer, image_batch):
        out = scorer.extract_features(image_batch)
        assert out.shape == (4, scorer.feature_dim)
        assert scorer.feature_dim == 512  # resnet18

    def test_logits_sigmoid_consistency(self, scorer, image_batch):
        # eval() で Dropout を無効化してから比較する
        # （training モードでは Dropout が各 forward で独立に適用されて値がずれる）
        scorer.eval()
        with torch.no_grad():
            logits = scorer.forward_logits(image_batch)
            scores = scorer(image_batch)
        assert torch.allclose(torch.sigmoid(logits), scores, atol=1e-6)

    def test_gradient_flows(self, scorer, image_batch):
        scorer.eval()
        x = image_batch.requires_grad_(True)
        score = scorer(x)
        score.sum().backward()
        assert x.grad is not None


# ---------- RealnessScorer (ViT backbone) ----------

class TestRealnessScorerViT:
    @pytest.fixture()
    def vit_scorer(self) -> RealnessScorer:
        return RealnessScorer(backbone="vit_tiny_p8", pretrained=False, image_size=VIT_IMAGE_SIZE)

    def test_forward_shape(self, vit_scorer, image_batch):
        out = vit_scorer(image_batch)
        assert out.shape == (4,)
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_image_size_not_divisible_by_patch_raises(self):
        with pytest.raises(ValueError):
            RealnessScorer(backbone="vit_tiny_p8", pretrained=False, image_size=65)

    def test_unknown_backbone_raises(self):
        with pytest.raises(ValueError):
            RealnessScorer(backbone="not_a_real_backbone", pretrained=False)

    def test_extract_intermediate_four_stages(self, vit_scorer, image_batch):
        inter = vit_scorer.extract_intermediate(image_batch)
        assert set(inter) == {"layer1", "layer2", "layer3", "layer4"}
        for v in inter.values():
            assert v.shape == (4, vit_scorer.feature_dim)

    def test_cls_attention_shape(self, vit_scorer, image_batch):
        attn = cls_attention(vit_scorer, image_batch)
        assert attn.shape == (4, VIT_GRID * VIT_GRID)
        assert (attn >= 0).all()

    def test_attention_rollout_shape(self, vit_scorer, image_batch):
        rollout = attention_rollout(vit_scorer, image_batch)
        assert rollout.shape == (4, VIT_GRID * VIT_GRID)
        assert (rollout >= 0).all()

    def test_chefer_relevance_shape(self, vit_scorer, image_batch):
        rel = chefer_relevance(vit_scorer, image_batch)
        assert rel.shape == (4, VIT_GRID * VIT_GRID)

    def test_token_grad_cam_shape_and_range(self, vit_scorer, image_batch):
        cam = token_grad_cam(vit_scorer, image_batch, layer="layer4")
        assert cam.shape == (4, VIT_GRID * VIT_GRID)
        assert cam.min() >= 0.0 and cam.max() <= 1.0

    def test_token_norm_map_shape(self, vit_scorer, image_batch):
        norms = token_norm_map(vit_scorer, image_batch, layer="layer4")
        assert norms.shape == (4, VIT_GRID * VIT_GRID)
        assert (norms >= 0).all()

    def test_grad_cam_raises_for_vit(self, vit_scorer):
        with pytest.raises(ValueError):
            GradCAM(vit_scorer)


# ---------- Loss functions ----------

class TestLoss:
    def test_smooth_labels_ones(self):
        labels = torch.ones(4)
        soft = smooth_labels(labels)
        assert torch.allclose(soft, torch.full((4,), 1.0 - LABEL_SMOOTHING))

    def test_smooth_labels_zeros(self):
        labels = torch.zeros(4)
        soft = smooth_labels(labels)
        assert torch.allclose(soft, torch.full((4,), LABEL_SMOOTHING))

    def test_smooth_labels_custom_smoothing(self):
        labels = torch.ones(4)
        soft = smooth_labels(labels, smoothing=0.2)
        assert torch.allclose(soft, torch.full((4,), 0.8))

    def test_realness_loss_scalar(self):
        logits = torch.randn(4)
        soft = smooth_labels(torch.tensor([1., 0., 1., 0.]))
        loss = realness_loss(logits, soft)
        assert loss.shape == torch.Size([])
        assert loss.item() > 0


# ---------- Metrics ----------

class TestMetrics:
    def test_metric_keys(self):
        mc = make_scorer_metrics("val/")
        assert "val/accuracy" in mc
        assert "val/auroc" in mc
        assert "val/ece" in mc

    def test_metric_compute(self, binary_labels):
        mc = make_scorer_metrics("val/")
        scores = torch.sigmoid(torch.randn(4))
        mc.update(scores, binary_labels)
        result = mc.compute()
        assert all(0.0 <= v.item() <= 1.0 for v in result.values())


# ---------- ScorerModule ----------

class TestScorerModule:
    @pytest.fixture()
    def module(self) -> ScorerModule:
        return ScorerModule(pretrained=False, ranking=False, lr_total_steps=100)

    def test_hparams_saved(self, module):
        assert "label_smoothing" in module.hparams
        assert "lr_total_steps" in module.hparams
        assert "mixup_alpha" not in module.hparams  # MixUp は削除済み

    def test_training_step(self, module, image_batch, binary_labels):
        batch = {"image": image_batch, "label": binary_labels}
        loss = module.training_step(batch, 0)
        assert loss.shape == torch.Size([])
        assert loss.item() > 0

    def test_validation_step(self, module, image_batch, binary_labels):
        batch = {"image": image_batch, "label": binary_labels}
        module.validation_step(batch, 0)
        # メトリクスが蓄積されているか確認
        assert module.val_metrics["val/auroc"]._update_count > 0


class TestScorerModuleViT:
    def test_vit_module_construction(self):
        module = ScorerModule(
            backbone="vit_tiny_p8", pretrained=False, ranking=False,
            image_size=VIT_IMAGE_SIZE, lr_total_steps=100,
        )
        assert module.scorer._is_vit

    def test_no_decay_norm_bias_splits_param_groups(self):
        module = ScorerModule(
            backbone="vit_tiny_p8", pretrained=False, ranking=False,
            image_size=VIT_IMAGE_SIZE, lr_total_steps=100, no_decay_norm_bias=True,
        )
        opt_cfg = module.configure_optimizers()
        optimizer = opt_cfg["optimizer"]
        assert len(optimizer.param_groups) == 4
        no_decay_groups = [g for g in optimizer.param_groups if g["weight_decay"] == 0.0]
        assert len(no_decay_groups) == 2
        assert all(len(g["params"]) > 0 for g in optimizer.param_groups)
