"""Image preprocessing and quality validation for MedVision skills."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

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
