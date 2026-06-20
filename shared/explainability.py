from __future__ import annotations

from typing import Callable, Literal

import cv2
import numpy as np
import torch

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
    ) -> np.ndarray:
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
        result: np.ndarray = np.asarray(grayscale_cam[0])
        return result

    def __del__(self) -> None:
        """Remove hooks when the explainer is garbage collected."""
        if hasattr(self, '_cam'):
            del self._cam
