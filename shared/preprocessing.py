"""Image preprocessing and quality validation for MedVision skills."""

from dataclasses import dataclass


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
