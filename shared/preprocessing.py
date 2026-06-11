"""Image preprocessing and quality validation for MedVision skills."""


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
