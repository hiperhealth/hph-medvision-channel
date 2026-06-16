"""Model weight registry for MedVision channel.

Manages downloading, caching, integrity verification, and
version-tracking of pre-trained model weights (DINOv2,
ResNet-50, etc.).
"""

from __future__ import annotations


class ModelIntegrityError(Exception):
    """Raised when a model file fails SHA-256 verification.

    Parameters
    ----------
    model_id : str
        Identifier of the model that failed verification.
    expected : str
        The SHA-256 hex digest we expected.
    actual : str
        The SHA-256 hex digest computed from the file.

    Attributes
    ----------
    model_id : str
    expected : str
    actual : str
    """

    def __init__(
        self,
        model_id: str,
        expected: str,
        actual: str,
    ) -> None:
        self.model_id = model_id
        self.expected = expected
        self.actual = actual
        super().__init__(
            f'Integrity check failed for {model_id!r}: '
            f'expected {expected[:16]}…, '
            f'got {actual[:16]}…'
        )
