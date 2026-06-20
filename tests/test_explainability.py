"""Unit tests for shared.explainability."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from torch import nn

from shared.explainability import (
    AttentionRollout,
    GradCAMExplainer,
    get_explainer,
    overlay_heatmap,
    reshape_transform,
)

# ── Helpers ──────────────────────────────────────────────


class _FakeViTBlock(nn.Module):
    """Minimal ViT block with attention that outputs (B,H,N,N)."""

    def __init__(self, seq_len: int = 17) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(16)
        self.attn = nn.Identity()
        self._seq_len = seq_len

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return x


class _FakeViT(nn.Module):
    """Minimal ViT with blocks attribute for testing."""

    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_FakeViTBlock()])
        self.head = nn.Linear(16, num_classes)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.head(torch.randn(x.size(0), 16))


class _FakeResNet(nn.Module):
    """Minimal ResNet with layer4 attribute for testing."""

    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()
        self.layer4 = nn.ModuleList([nn.Conv2d(3, 16, 1)])
        self.fc = nn.Linear(16, num_classes)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.fc(torch.randn(x.size(0), 16))


# ── reshape_transform ───────────────────────────────────


class TestReshapeTransform:
    """Tests for the ViT reshape_transform function."""

    def test_output_shape(self) -> None:
        b, n, c = 2, 257, 384
        tensor = torch.randn(b, n, c)
        out = reshape_transform(tensor, height=16, width=16)
        assert out.shape == (2, 384, 16, 16)

    def test_cls_token_removed(self) -> None:
        b, n, c = 1, 5, 8
        tensor = torch.randn(b, n, c)
        out = reshape_transform(tensor, height=2, width=2)
        assert out.shape == (1, 8, 2, 2)

    def test_batch_independence(self) -> None:
        b, n, c = 4, 65, 32
        tensor = torch.randn(b, n, c)
        out = reshape_transform(tensor, height=8, width=8)
        assert out.shape[0] == 4


# ── GradCAMExplainer ────────────────────────────────────


class TestGradCAMExplainer:
    """Tests for the GradCAMExplainer wrapper."""

    @patch('shared.explainability.GradCAM.__call__')
    def test_generate_returns_2d_array(
        self,
        mock_cam: MagicMock,
    ) -> None:
        mock_cam.return_value = np.random.rand(1, 4, 4)
        model = _FakeViT()
        explainer = GradCAMExplainer(
            model=model,
            target_layers=[model.blocks[0].norm1],
            reshape_transform_fn=lambda t: reshape_transform(t, 4, 4),
        )
        inp = torch.randn(1, 3, 56, 56)
        heatmap = explainer.generate(inp)

        assert isinstance(heatmap, np.ndarray)
        assert heatmap.ndim == 2

    @patch('shared.explainability.GradCAM.__call__')
    def test_generate_with_target_class(
        self,
        mock_cam: MagicMock,
    ) -> None:
        mock_cam.return_value = np.random.rand(1, 4, 4)
        model = _FakeViT(num_classes=5)
        explainer = GradCAMExplainer(
            model=model,
            target_layers=[model.blocks[0].norm1],
            reshape_transform_fn=lambda t: reshape_transform(t, 4, 4),
        )
        inp = torch.randn(1, 3, 56, 56)
        heatmap = explainer.generate(inp, target_class=2)

        assert isinstance(heatmap, np.ndarray)
        mock_cam.assert_called_once()

    def test_cleanup_on_del(self) -> None:
        model = _FakeViT()
        explainer = GradCAMExplainer(
            model=model,
            target_layers=[model.blocks[0].norm1],
        )
        assert hasattr(explainer, '_cam')
        del explainer


# ── AttentionRollout ─────────────────────────────────────


class TestAttentionRollout:
    """Tests for the AttentionRollout class."""

    def test_raises_without_blocks(self) -> None:
        model = nn.Linear(10, 10)
        with pytest.raises(AttributeError, match='blocks'):
            AttentionRollout(model)

    def test_hooks_registered(self) -> None:
        model = _FakeViT()
        rollout = AttentionRollout(model)
        assert len(rollout._hooks) == len(list(model.blocks))
        rollout.remove_hooks()

    def test_remove_hooks_clears(self) -> None:
        model = _FakeViT()
        rollout = AttentionRollout(model)
        rollout.remove_hooks()
        assert len(rollout._hooks) == 0

    def test_generate_raises_no_attentions(self) -> None:
        model = _FakeViT()
        rollout = AttentionRollout(model)
        rollout.remove_hooks()
        inp = torch.randn(1, 3, 224, 224)

        with pytest.raises(
            RuntimeError,
            match='No attention',
        ):
            rollout.generate(inp)

    def test_generate_with_mock_attention(self) -> None:
        model = _FakeViT()
        rollout = AttentionRollout(model)
        rollout.remove_hooks()

        seq_len = 17
        fake_attn = torch.rand(1, 4, seq_len, seq_len)
        fake_attn = fake_attn / fake_attn.sum(dim=-1, keepdim=True)

        original_forward = model.forward

        def _inject_attn(
            x: torch.Tensor,
        ) -> torch.Tensor:
            rollout._attentions.append(fake_attn)
            return original_forward(x)

        with patch.object(model, 'forward', side_effect=_inject_attn):
            inp = torch.randn(1, 3, 224, 224)
            result = rollout.generate(inp)

        assert isinstance(result, np.ndarray)
        assert result.ndim == 2
        assert result.min() >= 0.0
        assert result.max() <= 1.0 + 1e-6


# ── overlay_heatmap ──────────────────────────────────────


class TestOverlayHeatmap:
    """Tests for the overlay_heatmap utility."""

    def test_output_shape_matches_input(self) -> None:
        image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        heatmap = np.random.rand(14, 14).astype(np.float32)
        result = overlay_heatmap(image, heatmap)

        assert result.shape == (224, 224, 3)
        assert result.dtype == np.uint8

    def test_alpha_zero_preserves_original(self) -> None:
        image = np.full((100, 100, 3), 128, dtype=np.uint8)
        heatmap = np.ones((10, 10), dtype=np.float32)
        result = overlay_heatmap(image, heatmap, alpha=0.0)

        assert result.shape == image.shape

    def test_heatmap_resized_to_image(self) -> None:
        image = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        heatmap = np.random.rand(8, 8).astype(np.float32)
        result = overlay_heatmap(image, heatmap)

        assert result.shape == (300, 400, 3)


# ── get_explainer ────────────────────────────────────────


class TestGetExplainer:
    """Tests for the get_explainer factory function."""

    def test_dinov2_backend(self) -> None:
        model = _FakeViT()
        explainer = get_explainer(model, backend='dinov2')
        assert isinstance(explainer, GradCAMExplainer)

    def test_resnet50_backend(self) -> None:
        model = _FakeResNet()
        explainer = get_explainer(model, backend='resnet50')
        assert isinstance(explainer, GradCAMExplainer)

    def test_unsupported_backend_raises(self) -> None:
        model = nn.Linear(10, 10)
        with pytest.raises(
            ValueError,
            match='Unsupported backend',
        ):
            get_explainer(
                model,
                backend='mobilenet',  # type: ignore[arg-type]
            )

    def test_default_is_dinov2(self) -> None:
        model = _FakeViT()
        explainer = get_explainer(model)
        assert isinstance(explainer, GradCAMExplainer)
