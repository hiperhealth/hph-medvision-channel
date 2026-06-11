"""Image preprocessing and quality validation for MedVision skills."""

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import cv2
import numpy as np
import torch

from monai.transforms import (  # type: ignore[attr-defined]
    Compose,
    EnsureChannelFirst,
    NormalizeIntensity,
    Resize,
    ScaleIntensity,
)
from PIL import Image as PILImage
from PIL.ExifTags import Base as ExifBase


@dataclass(frozen=True)
class QualityThresholds:
    """Configurable thresholds for image quality validation.

    Parameters
    ----------
    min_resolution : int
        Minimum width AND height in pixels.  224 matches the
        standard input size for DINOv2 and ResNet-50.
    max_aspect_ratio : float
        Maximum longest-side / shortest-side ratio.
    min_laplacian_var : float
        Minimum Laplacian variance for blur detection.
    allowed_formats : frozenset[str]
        Accepted file extensions (lowercase, with dot).
    """

    min_resolution: int = 224
    max_aspect_ratio: float = 3.0
    min_laplacian_var: float = 100.0
    allowed_formats: frozenset[str] = frozenset(
        {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'}
    )


DEFAULT_THRESHOLDS = QualityThresholds()


class ImageQualityError(Exception):
    """Raised when an image fails quality validation checks.

    Parameters
    ----------
    reason : str
        Human-readable description of the failure.
    detail : dict[str, object]
        Machine-readable context with keys ``"check"``,
        ``"actual"``, and ``"threshold"`` for programmatic
        error handling.

    Attributes
    ----------
    reason : str
    detail : dict[str, object]
    """

    def __init__(
        self,
        reason: str,
        detail: dict[str, object] | None = None,
    ) -> None:
        self.reason = reason
        self.detail = detail or {}
        super().__init__(f'Image quality error: {reason}')


def correct_exif_orientation(
    image: np.ndarray,
    path: Path,
) -> np.ndarray:
    """Rotate *image* to match its EXIF orientation tag.

    Phone cameras store photos with the raw sensor orientation
    and an EXIF tag indicating the correct display rotation.
    OpenCV's ``imread`` ignores this tag, so without correction
    the model would see a rotated image.

    Parameters
    ----------
    image : np.ndarray
        BGR image array from ``cv2.imread``.
    path : Path
        File path used to read the EXIF header via PIL.

    Returns
    -------
    np.ndarray
        Correctly oriented image, or the original if no EXIF
        orientation is present.
    """
    try:
        with PILImage.open(path) as pil_img:
            exif_data = pil_img.getexif()
    except Exception:
        return image

    orientation = exif_data.get(ExifBase.Orientation)
    if orientation is None:
        return image

    _ROTATION_MAP = {
        3: cv2.ROTATE_180,
        6: cv2.ROTATE_90_CLOCKWISE,
        8: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }

    rotation = _ROTATION_MAP.get(orientation)
    if rotation is not None:
        return cv2.rotate(image, rotation)

    return image


def validate_image(
    image_path: str | Path,
    thresholds: QualityThresholds | None = None,
) -> np.ndarray:
    """Validate a clinical image against quality thresholds.

    Checks are applied in order: file existence, format, decode
    and resolution, aspect ratio, blur detection.  The image is
    EXIF-corrected before dimension checks.

    Parameters
    ----------
    image_path : str | Path
        Path to the image file.
    thresholds : QualityThresholds | None
        Custom thresholds; defaults to ``DEFAULT_THRESHOLDS``.

    Returns
    -------
    np.ndarray
        Decoded, EXIF-corrected BGR image.

    Raises
    ------
    FileNotFoundError
        If *image_path* does not exist.
    ImageQualityError
        If any quality check fails.
    """
    path = Path(image_path)
    cfg = thresholds or DEFAULT_THRESHOLDS

    if not path.is_file():
        raise FileNotFoundError(f'Image not found: {path}')

    suffix = path.suffix.lower()
    if suffix not in cfg.allowed_formats:
        raise ImageQualityError(
            reason=f'unsupported format {suffix!r}',
            detail={
                'check': 'format',
                'actual': suffix,
                'allowed': sorted(cfg.allowed_formats),
            },
        )

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ImageQualityError(
            reason='file could not be decoded as an image',
            detail={'check': 'decode', 'path': str(path)},
        )

    image = correct_exif_orientation(image, path)

    height, width = image.shape[:2]
    if width < cfg.min_resolution or height < cfg.min_resolution:
        raise ImageQualityError(
            reason=(
                f'resolution {width}x{height} below minimum '
                f'{cfg.min_resolution}px'
            ),
            detail={
                'check': 'resolution',
                'actual': (width, height),
                'threshold': cfg.min_resolution,
            },
        )

    aspect = max(width, height) / max(min(width, height), 1)
    if aspect > cfg.max_aspect_ratio:
        raise ImageQualityError(
            reason=(
                f'aspect ratio {aspect:.2f} exceeds maximum '
                f'{cfg.max_aspect_ratio}'
            ),
            detail={
                'check': 'aspect_ratio',
                'actual': aspect,
                'threshold': cfg.max_aspect_ratio,
            },
        )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if laplacian_var < cfg.min_laplacian_var:
        raise ImageQualityError(
            reason=(
                f'image is too blurry (laplacian variance '
                f'{laplacian_var:.1f} < {cfg.min_laplacian_var})'
            ),
            detail={
                'check': 'blur',
                'actual': laplacian_var,
                'threshold': cfg.min_laplacian_var,
            },
        )

    return image


_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def build_inference_transforms(
    target_size: int = 224,
) -> Compose:
    """Build a MONAI transform pipeline for inference.

    Converts a decoded NumPy image (HxWxC, uint8) into a
    normalised tensor (CxHxW, float32) suitable for DINOv2
    or ResNet-50.

    Parameters
    ----------
    target_size : int
        Spatial dimensions for the output tensor.

    Returns
    -------
    monai.transforms.Compose
        Callable: ``np.ndarray`` → ``torch.Tensor``.
    """
    return Compose(
        [
            EnsureChannelFirst(channel_dim=-1),
            Resize(spatial_size=(target_size, target_size)),
            ScaleIntensity(),
            NormalizeIntensity(
                subtrahend=_IMAGENET_MEAN,
                divisor=_IMAGENET_STD,
                channel_wise=True,
            ),
        ]
    )


def _bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert a BGR image to contiguous RGB array.

    Parameters
    ----------
    image : np.ndarray
        BGR array from ``cv2.imread``.

    Returns
    -------
    np.ndarray
        Contiguous RGB array.
    """
    return np.ascontiguousarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


class ImagePreprocessor:
    """End-to-end image preprocessing for MedVision inference.

    Combines quality validation and MONAI transforms into a single
    callable.  This is the public API that skills import.

    Parameters
    ----------
    target_size : int
        Spatial size for the model input tensor.
    thresholds : QualityThresholds | None
        Custom quality thresholds; uses module defaults if ``None``.

    Examples
    --------
    >>> preprocessor = ImagePreprocessor()
    >>> tensor = preprocessor.preprocess('/uploads/skin_photo.jpg')
    >>> tensor.shape
    torch.Size([1, 3, 224, 224])
    """

    def __init__(
        self,
        target_size: int = 224,
        thresholds: QualityThresholds | None = None,
    ) -> None:
        self._target_size = target_size
        self._thresholds = thresholds or DEFAULT_THRESHOLDS
        self._transforms = build_inference_transforms(target_size)

    def validate(self, image_path: str | Path) -> np.ndarray:
        """Validate image quality without running transforms.

        Parameters
        ----------
        image_path : str | Path
            Path to the image file.

        Returns
        -------
        np.ndarray
            Decoded, EXIF-corrected BGR image.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ImageQualityError
            If the image fails any quality check.
        """
        return validate_image(image_path, self._thresholds)

    def preprocess(self, image_path: str | Path) -> torch.Tensor:
        """Validate and preprocess an image for model inference.

        Parameters
        ----------
        image_path : str | Path
            Path to the image file.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(1, 3, target_size, target_size)``.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ImageQualityError
            If the image fails any quality check.
        """
        bgr_image = self.validate(image_path)
        rgb_array = _bgr_to_rgb(bgr_image)
        tensor = self._transforms(rgb_array)

        if not isinstance(tensor, torch.Tensor):
            tensor = torch.as_tensor(tensor)

        return cast(torch.Tensor, tensor.unsqueeze(0))
