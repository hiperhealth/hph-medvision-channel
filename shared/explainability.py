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
