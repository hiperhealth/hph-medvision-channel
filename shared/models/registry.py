"""Model weight registry for MedVision channel.

Manages downloading, caching, integrity verification, and
version-tracking of pre-trained model weights (DINOv2,
ResNet-50, etc.).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import shutil
import urllib.request

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from filelock import FileLock

_DEFAULT_CACHE_DIR = Path.home() / '.hiperhealth' / 'models'
_MANIFEST_FILENAME = 'manifest.json'
_MANIFEST_LOCK = '.manifest.lock'
_CHUNK_SIZE = 8192  # bytes per read during download/hash
_SAFE_ID_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]*$')


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
        self._manifest_lock = FileLock(
            self._cache_dir / _MANIFEST_LOCK,
        )

    @staticmethod
    def _validate_model_id(model_id: str) -> None:
        """Reject model IDs that could escape the cache dir.

        Only alphanumeric characters, dots, hyphens, and
        underscores are allowed.  Path separators (``/``,
        ``\\``) and ``..`` sequences are forbidden to
        prevent path-traversal attacks.

        Raises
        ------
        ValueError
            If *model_id* contains unsafe characters.
        """
        if not model_id or not _SAFE_ID_RE.match(model_id) or '..' in model_id:
            raise ValueError(
                f'Invalid model_id {model_id!r}: '
                'must be non-empty, contain only '
                '[A-Za-z0-9._-], and not include '
                '".." sequences.'
            )

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
        with self._manifest_path.open(
            'r',
            encoding='utf-8',
        ) as fh:
            return json.load(fh)  # type: ignore[no-any-return]

    def _save_manifest(
        self,
        manifest: dict[str, object],
    ) -> None:
        """Write the JSON manifest to disk atomically.

        Writes to a temporary ``.tmp`` file first, then
        uses ``replace()`` (``os.replace``) to atomically
        overwrite the final path.  Unlike ``rename()``,
        ``replace()`` works cross-platform even when the
        destination already exists.

        Parameters
        ----------
        manifest : dict[str, object]
            Full manifest to persist.
        """
        tmp = self._manifest_path.with_suffix('.tmp')
        with tmp.open('w', encoding='utf-8') as fh:
            json.dump(manifest, fh, indent=2)
        tmp.replace(self._manifest_path)

    @staticmethod
    def _compute_sha256(path: Path) -> str:
        """Compute the SHA-256 hex digest of a file.

        Reads in chunks to handle arbitrarily large model
        weight files without loading them into memory.

        Parameters
        ----------
        path : Path
            File to hash.

        Returns
        -------
        str
            Lowercase hex digest (64 characters).
        """
        hasher = hashlib.sha256()
        with path.open('rb') as fh:
            while chunk := fh.read(_CHUNK_SIZE):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _download(self, url: str, dest: Path) -> None:
        """Download a file from *url* to *dest*.

        Uses a temporary ``.part`` file and atomic rename
        to prevent partial downloads from being used.
        The temp file is always in the same directory as
        *dest* so ``rename()`` is atomic on POSIX.

        Parameters
        ----------
        url : str
            Remote URL to fetch.
        dest : Path
            Final local path for the downloaded file.
        """
        tmp = dest.with_suffix('.part')
        try:
            req = urllib.request.Request(
                url,
                headers={'User-Agent': 'hph-medvision/0.1'},
            )
            with urllib.request.urlopen(req) as resp:
                with tmp.open('wb') as fh:
                    shutil.copyfileobj(resp, fh)
            tmp.replace(dest)
        finally:
            if tmp.exists():
                tmp.unlink()

    def get_weights(
        self,
        model_id: str,
        url: str,
        expected_sha256: str,
        *,
        verify_cached: bool = False,
    ) -> Path:
        """Retrieve model weights, downloading if necessary.

        A per-model file lock prevents concurrent processes
        from downloading the same file simultaneously.

        Parameters
        ----------
        model_id : str
            Unique model identifier.
        url : str
            Download URL for the weights file.
        expected_sha256 : str
            Expected SHA-256 hex digest.
        verify_cached : bool
            If ``True``, re-compute SHA-256 even on a cache
            hit.  Defaults to ``False`` for speed; set to
            ``True`` in strict clinical mode.

        Returns
        -------
        Path
            Absolute path to the verified weight file.

        Raises
        ------
        ModelIntegrityError
            If the file's hash does not match
            *expected_sha256*.
        """
        self._validate_model_id(model_id)
        filename = f'{model_id}.pth'
        dest = self._cache_dir / filename
        lock = FileLock(
            self._cache_dir / f'.{model_id}.lock',
        )

        with lock:
            # 1. Cache hit
            if dest.exists():
                if not verify_cached:
                    return dest
                actual = self._compute_sha256(dest)
                if actual == expected_sha256:
                    return dest
                # Hash mismatch -> delete stale file
                dest.unlink()

            # 2. Cache miss -> download and verify
            self._download(url, dest)
            actual = self._compute_sha256(dest)
            if actual != expected_sha256:
                dest.unlink()
                raise ModelIntegrityError(
                    model_id,
                    expected_sha256,
                    actual,
                )

            # 3. Update manifest under global lock
            with self._manifest_lock:
                manifest = self._load_manifest()
                entry = ModelEntry(
                    model_id=model_id,
                    url=url,
                    sha256=expected_sha256,
                    filename=filename,
                    downloaded_at=datetime.now(
                        tz=timezone.utc,
                    ).isoformat(),
                )
                manifest[model_id] = dataclasses.asdict(
                    entry,
                )
                self._save_manifest(manifest)

            return dest

    def list_models(self) -> list[str]:
        """Return model IDs present in the manifest.

        Returns
        -------
        list[str]
            Sorted list of model IDs that have been
            successfully downloaded and verified.
        """
        return sorted(self._load_manifest().keys())
