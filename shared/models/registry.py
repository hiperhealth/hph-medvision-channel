"""Model weight registry for MedVision channel.

Manages downloading, caching, integrity verification, and
version-tracking of pre-trained model weights (DINOv2,
ResNet-50, etc.).
"""

from __future__ import annotations

import json

from dataclasses import dataclass
from pathlib import Path

_DEFAULT_CACHE_DIR = Path.home() / '.hiperhealth' / 'models'
_MANIFEST_FILENAME = 'manifest.json'
_CHUNK_SIZE = 8192  # bytes per read during download/hash


@dataclass(frozen=True)
class ModelEntry:
    """Single entry in the model registry manifest.

    Parameters
    ----------
    model_id : str
        Unique identifier (e.g., ``'dinov2_vits14_skin'``).
    url : str
        Remote URL to download weights from.
    sha256 : str
        Expected SHA-256 hex digest of the file.
    filename : str
        Local filename under the cache directory.
    downloaded_at : str | None
        ISO-8601 timestamp of when the file was cached.
    """

    model_id: str
    url: str
    sha256: str
    filename: str
    downloaded_at: str | None = None


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


class ModelRegistry:
    """Download, cache, and verify pre-trained model weights.

    Maintains a JSON manifest tracking every model that has
    been downloaded.  Files live under a single cache
    directory and are verified with SHA-256.

    Parameters
    ----------
    cache_dir : Path | None
        Override the default ``~/.hiperhealth/models/``
        cache.  Useful for testing with ``tmp_path``.
    """

    def __init__(
        self,
        cache_dir: Path | None = None,
    ) -> None:
        self._cache_dir = cache_dir or _DEFAULT_CACHE_DIR
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._manifest_path = self._cache_dir / _MANIFEST_FILENAME

    def _load_manifest(self) -> dict[str, object]:
        """Read the JSON manifest from disk.

        Returns
        -------
        dict[str, object]
            Manifest contents, or empty dict if the file
            does not exist yet.
        """
        if not self._manifest_path.exists():
            return {}
        with self._manifest_path.open('r') as fh:
            return json.load(fh)  # type: ignore[no-any-return]

    def _save_manifest(
        self,
        manifest: dict[str, object],
    ) -> None:
        """Write the JSON manifest to disk atomically.

        Writes to a temporary ``.tmp`` file first, then
        renames to the final path.  ``rename()`` is atomic
        on POSIX when source and target are on the same
        filesystem (guaranteed here since both are in
        ``_cache_dir``).

        Parameters
        ----------
        manifest : dict[str, object]
            Full manifest to persist.
        """
        tmp = self._manifest_path.with_suffix('.tmp')
        with tmp.open('w') as fh:
            json.dump(manifest, fh, indent=2)
        tmp.rename(self._manifest_path)
