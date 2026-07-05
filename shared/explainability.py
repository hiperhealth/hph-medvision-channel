from __future__ import annotations

from typing import Any, Callable, Literal, cast

import cv2
import numpy as np
import torch

from numpy.typing import NDArray
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import (
    ClassifierOutputTarget,
)
from torch import nn


def reshape_transform(
    tensor: torch.Tensor,
    height: int = 16,
    width: int = 16,
) -> torch.Tensor:
    """Reshape ViT token sequence back to a spatial 2D grid.

    Grad-CAM needs a 2D spatial feature map to compute
    heatmaps. Vision Transformers output a flat sequence of
    tokens. This function removes the CLS token and reshapes
    the remaining patch tokens into a grid.

    Parameters
    ----------
    tensor : torch.Tensor
        Output from a ViT block, shape ``(B, N, C)`` where
        ``N = 1 + H*W`` (1 CLS token + patch tokens).
    height : int
        Number of patch rows.  For DINOv2 ViT-S/14 with
        224x224 input: ``224 / 14 = 16``.
    width : int
        Number of patch columns.  Same calculation as height.

    Returns
    -------
    torch.Tensor
        Spatial feature map of shape ``(B, C, H, W)``.
    """
    # Index 0 is the CLS token which has no spatial position.
    result = tensor[:, 1:, :]

    result = result.reshape(
        tensor.size(0),
        height,
        width,
        tensor.size(2),
    )

    result = result.transpose(2, 3).transpose(1, 2)

    return result


class GradCAMExplainer:
    """Wrapper around pytorch-grad-cam for clinical explainability.

    Generates heatmaps highlighting the regions of an image
    that most strongly influenced a model's prediction.

    Parameters
    ----------
    model : nn.Module
        The PyTorch model to explain (CNN or ViT).
    target_layers : list[nn.Module]
        Layers to compute gradients from. For ResNet, typically
        the last conv layer. For ViT, typically the LayerNorm
        before the final block.
    reshape_transform_fn : Callable | None
        Function to reshape flat sequences to spatial grids.
        Required for ViTs, ignored for CNNs.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layers: list[nn.Module],
        reshape_transform_fn: Callable[[torch.Tensor], torch.Tensor]
        | None = None,
    ) -> None:
        self._model = model
        self._target_layers = target_layers

        self._cam = GradCAM(
            model=model,
            target_layers=target_layers,
            reshape_transform=reshape_transform_fn,
        )

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: int | None = None,
    ) -> NDArray[Any]:
        """Generate a Grad-CAM heatmap for the input.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Preprocessed image tensor, shape ``(1, 3, H, W)``.
        target_class : int | None
            Class index to explain.  If ``None``, uses the
            model's highest-scoring (argmax) prediction.

        Returns
        -------
        np.ndarray
            Grayscale heatmap of shape ``(H, W)`` with float
            values in ``[0, 1]``.  Higher values indicate more
            important regions for the prediction.
        """
        targets: list[ClassifierOutputTarget] | None = None
        if target_class is not None:
            targets = [ClassifierOutputTarget(target_class)]

        grayscale_cam = self._cam(
            input_tensor=input_tensor,
            targets=targets,
        )

        # np.asarray to satisfy mypy — library returns untyped.
        result: NDArray[Any] = np.asarray(grayscale_cam[0])
        return result

    def __del__(self) -> None:
        """Remove hooks when the explainer is garbage collected."""
        if hasattr(self, '_cam'):
            del self._cam


class AttentionRollout:
    """Attention Rollout for Vision Transformers.

    Produces a spatial attention map by recursively multiplying
    attention matrices across all transformer layers.  This is
    a gradient-free method — useful as a fallback when
    Grad-CAM cannot be used (e.g., frozen models, no-grad
    inference pipelines).

    Parameters
    ----------
    model : nn.Module
        A ViT model with ``model.blocks`` containing
        transformer encoder blocks, each having an ``attn``
        sub-module.
    head_fusion : str
        How to fuse multi-head attention matrices.
        ``'mean'`` averages across heads (most common).
        ``'max'`` takes the maximum.
        ``'min'`` takes the minimum.

    Raises
    ------
    AttributeError
        If the model does not have a ``blocks`` attribute
        (i.e., it is not a ViT).
    """

    def __init__(
        self,
        model: nn.Module,
        head_fusion: Literal['mean', 'max', 'min'] = 'mean',
    ) -> None:
        self._model = model
        self._head_fusion = head_fusion

        if not hasattr(model, 'blocks'):
            msg = (
                'AttentionRollout requires a ViT model '
                'with a `blocks` attribute'
            )
            raise AttributeError(msg)

        self._attentions: list[torch.Tensor] = []
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []
        for block in model.blocks:  # type: ignore[union-attr]
            hook = block.attn.register_forward_hook(self._save_attention)
            self._hooks.append(hook)

    def _save_attention(
        self,
        _module: nn.Module,
        _input: tuple[torch.Tensor, ...],
        output: tuple[torch.Tensor, ...] | torch.Tensor,
    ) -> None:
        """Hook callback: save attention output."""
        if isinstance(output, tuple):
            self._attentions.append(output[0].detach().cpu())
        else:
            self._attentions.append(output.detach().cpu())

    def generate(
        self,
        input_tensor: torch.Tensor,
    ) -> NDArray[Any]:
        """Generate an attention rollout map.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Preprocessed image, shape ``(1, 3, H, W)``.

        Returns
        -------
        np.ndarray
            Attention map of shape ``(H_patches, W_patches)``
            with float values in ``[0, 1]``.
        """
        self._attentions.clear()

        with torch.no_grad():
            self._model(input_tensor)

        if not self._attentions:
            msg = (
                'No attention weights captured. Verify '
                'the model attention modules produce '
                'compatible output.'
            )
            raise RuntimeError(msg)

        first_attn = self._attentions[0]
        seq_len = first_attn.size(-1)
        result = torch.eye(seq_len)

        for attention in self._attentions:
            if attention.ndim == 4:
                if self._head_fusion == 'mean':
                    fused = attention.mean(dim=1)[0]
                elif self._head_fusion == 'max':
                    fused = attention.amax(dim=1)[0]
                else:
                    fused = attention.amin(dim=1)[0]
            else:
                fused = attention[0]

            # 0.5*A + 0.5*I simulates the residual connection
            # path (x + attn(x)) present in every ViT block.
            fused_hat = 0.5 * fused + 0.5 * torch.eye(fused.size(-1))
            fused_hat = fused_hat / fused_hat.sum(dim=-1, keepdim=True)
            result = fused_hat @ result

        # CLS token row (index 0), excluding CLS-to-CLS.
        mask = result[0, 1:]

        num_patches = mask.size(0)
        h = w = int(num_patches**0.5)
        mask_np: NDArray[Any] = np.asarray(mask.reshape(h, w).numpy())

        mask_min = float(mask_np.min())
        mask_max = float(mask_np.max())
        mask_np = (mask_np - mask_min) / (mask_max - mask_min + 1e-8)

        return mask_np

    def remove_hooks(self) -> None:
        """Remove all registered forward hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def __del__(self) -> None:
        """Clean up hooks on garbage collection."""
        if hasattr(self, '_hooks'):
            self.remove_hooks()


def overlay_heatmap(
    image: NDArray[Any],
    heatmap: NDArray[Any],
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
) -> NDArray[Any]:
    """Overlay a Grad-CAM heatmap on the original image.

    Parameters
    ----------
    image : np.ndarray
        Original image in RGB format, shape ``(H, W, 3)``,
        dtype ``uint8``.
    heatmap : np.ndarray
        Grayscale heatmap from ``GradCAMExplainer.generate()``,
        shape ``(H_cam, W_cam)`` with float values in
        ``[0, 1]``.  Will be resized to match ``image``.
    alpha : float
        Blending factor.  ``0.0`` = original image only,
        ``1.0`` = heatmap only.  ``0.5`` is a good default
        for clinical visualization.
    colormap : int
        OpenCV colormap constant.  ``cv2.COLORMAP_JET`` is
        the standard for Grad-CAM (blue=cold, red=hot).

    Returns
    -------
    np.ndarray
        Blended image in RGB, shape ``(H, W, 3)``, dtype
        ``uint8``.  Ready to save as PNG.
    """
    from pytorch_grad_cam.utils.image import show_cam_on_image

    h, w = image.shape[:2]

    heatmap_resized = cv2.resize(
        heatmap.astype(np.float32),
        (w, h),
        interpolation=cv2.INTER_LINEAR,
    )

    img_float = image.astype(np.float32) / 255.0

    overlay_img: NDArray[Any] = show_cam_on_image(
        img_float,
        heatmap_resized,
        use_rgb=True,
        image_weight=1.0 - alpha,
        colormap=colormap,
    )

    return overlay_img


def get_explainer(
    model: nn.Module,
    backend: Literal['dinov2', 'resnet50'] = 'dinov2',
) -> GradCAMExplainer:
    """Create a GradCAMExplainer for a given model backend.

    Automatically selects the correct target layer and
    reshape transform based on the model architecture.

    Parameters
    ----------
    model : nn.Module
        The model to explain.
    backend : str
        ``'dinov2'`` for DINOv2 ViT-S/14 or ``'resnet50'``
        for ResNet-50.

    Returns
    -------
    GradCAMExplainer
        Ready-to-use explainer instance.

    Raises
    ------
    ValueError
        If ``backend`` is not a supported architecture.
    """
    if backend == 'dinov2':
        blocks = getattr(model, 'blocks')
        target_layers: list[nn.Module] = [
            cast(nn.Module, blocks[-1].norm1),
        ]
        return GradCAMExplainer(
            model=model,
            target_layers=target_layers,
            reshape_transform_fn=reshape_transform,
        )

    if backend == 'resnet50':
        layer4 = getattr(model, 'layer4')
        target_layers = [
            cast(nn.Module, layer4[-1]),
        ]
        return GradCAMExplainer(
            model=model,
            target_layers=target_layers,
        )

    msg = f"Unsupported backend: '{backend}'"
    raise ValueError(msg)
