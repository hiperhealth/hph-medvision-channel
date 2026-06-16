"""Unit tests for shared.models.registry."""

from __future__ import annotations

import hashlib
import json

from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from shared.models.registry import (
    ModelEntry,
    ModelIntegrityError,
    ModelRegistry,
)

# ── Helpers ──────────────────────────────────────────────


def _sha256(data: bytes) -> str:
    """Return the SHA-256 hex digest of *data*."""
    return hashlib.sha256(data).hexdigest()


_FAKE_WEIGHTS = b'\x89MODEL_WEIGHTS_PAYLOAD'
_FAKE_SHA = _sha256(_FAKE_WEIGHTS)
_FAKE_URL = 'https://example.com/model.pth'
_MODEL_ID = 'test_model'


def _mock_urlopen(
    data: bytes = _FAKE_WEIGHTS,
) -> MagicMock:
    """Build a mock that replaces ``urllib.request.urlopen``.

    Returns a context-manager mock whose ``read()`` yields
    *data* via a ``BytesIO`` wrapper, matching the streaming
    interface that ``shutil.copyfileobj`` expects.
    """
    resp = MagicMock()
    buf = BytesIO(data)
    resp.__enter__ = lambda s: buf
    resp.__exit__ = MagicMock(return_value=False)
    return resp


# ── ModelEntry ───────────────────────────────────────────


class TestModelEntry:
    """Tests for the ModelEntry frozen dataclass."""

    def test_fields(self) -> None:
        entry = ModelEntry(
            model_id='m1',
            url='https://x.com/m1.pth',
            sha256='abc123',
            filename='m1.pth',
        )
        assert entry.model_id == 'm1'
        assert entry.downloaded_at is None

    def test_frozen(self) -> None:
        entry = ModelEntry(
            model_id='m1',
            url='u',
            sha256='s',
            filename='f',
        )
        with pytest.raises(AttributeError):
            entry.model_id = 'changed'  # type: ignore[misc]


# ── ModelIntegrityError ──────────────────────────────────


class TestModelIntegrityError:
    """Tests for the ModelIntegrityError exception."""

    def test_attributes(self) -> None:
        err = ModelIntegrityError('m1', 'aaa', 'bbb')
        assert err.model_id == 'm1'
        assert err.expected == 'aaa'
        assert err.actual == 'bbb'

    def test_message_truncates_hashes(self) -> None:
        long_hash = 'a' * 64
        err = ModelIntegrityError('m1', long_hash, 'b' * 64)
        assert long_hash[:16] in str(err)


# ── ModelRegistry ────────────────────────────────────────


class TestModelRegistry:
    """Tests for the ModelRegistry class."""

    def test_creates_cache_dir(self, tmp_path: Path) -> None:
        cache = tmp_path / 'sub' / 'models'
        ModelRegistry(cache_dir=cache)
        assert cache.is_dir()

    def test_list_models_empty(
        self,
        tmp_path: Path,
    ) -> None:
        reg = ModelRegistry(cache_dir=tmp_path)
        assert reg.list_models() == []

    @patch('shared.models.registry.urllib.request.urlopen')
    def test_get_weights_downloads_and_caches(
        self,
        mock_urlopen: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_urlopen.return_value = _mock_urlopen()

        reg = ModelRegistry(cache_dir=tmp_path)
        path = reg.get_weights(
            _MODEL_ID,
            _FAKE_URL,
            _FAKE_SHA,
        )

        assert path.exists()
        assert path.name == f'{_MODEL_ID}.pth'
        assert path.read_bytes() == _FAKE_WEIGHTS
        mock_urlopen.assert_called_once()

    @patch('shared.models.registry.urllib.request.urlopen')
    def test_cache_hit_skips_download(
        self,
        mock_urlopen: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_urlopen.return_value = _mock_urlopen()

        reg = ModelRegistry(cache_dir=tmp_path)
        reg.get_weights(_MODEL_ID, _FAKE_URL, _FAKE_SHA)

        mock_urlopen.reset_mock()
        path = reg.get_weights(
            _MODEL_ID,
            _FAKE_URL,
            _FAKE_SHA,
        )

        assert path.exists()
        mock_urlopen.assert_not_called()

    @patch('shared.models.registry.urllib.request.urlopen')
    def test_cache_hit_with_verify_cached(
        self,
        mock_urlopen: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_urlopen.return_value = _mock_urlopen()

        reg = ModelRegistry(cache_dir=tmp_path)
        reg.get_weights(_MODEL_ID, _FAKE_URL, _FAKE_SHA)

        mock_urlopen.reset_mock()
        path = reg.get_weights(
            _MODEL_ID,
            _FAKE_URL,
            _FAKE_SHA,
            verify_cached=True,
        )

        assert path.exists()
        mock_urlopen.assert_not_called()

    @patch('shared.models.registry.urllib.request.urlopen')
    def test_integrity_error_on_hash_mismatch(
        self,
        mock_urlopen: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_urlopen.return_value = _mock_urlopen()

        reg = ModelRegistry(cache_dir=tmp_path)
        with pytest.raises(ModelIntegrityError) as exc_info:
            reg.get_weights(
                _MODEL_ID,
                _FAKE_URL,
                'wrong_hash',
            )

        assert exc_info.value.model_id == _MODEL_ID
        assert exc_info.value.expected == 'wrong_hash'
        # File should be deleted after failed verification
        dest = tmp_path / f'{_MODEL_ID}.pth'
        assert not dest.exists()

    @patch('shared.models.registry.urllib.request.urlopen')
    def test_stale_cache_redownloads(
        self,
        mock_urlopen: MagicMock,
        tmp_path: Path,
    ) -> None:
        # Pre-populate cache with stale data
        stale = tmp_path / f'{_MODEL_ID}.pth'
        stale.write_bytes(b'stale_content')

        mock_urlopen.return_value = _mock_urlopen()

        reg = ModelRegistry(cache_dir=tmp_path)
        path = reg.get_weights(
            _MODEL_ID,
            _FAKE_URL,
            _FAKE_SHA,
            verify_cached=True,
        )

        assert path.read_bytes() == _FAKE_WEIGHTS
        mock_urlopen.assert_called_once()

    @patch('shared.models.registry.urllib.request.urlopen')
    def test_manifest_persists_across_instances(
        self,
        mock_urlopen: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_urlopen.return_value = _mock_urlopen()

        reg1 = ModelRegistry(cache_dir=tmp_path)
        reg1.get_weights(_MODEL_ID, _FAKE_URL, _FAKE_SHA)

        reg2 = ModelRegistry(cache_dir=tmp_path)
        assert _MODEL_ID in reg2.list_models()

    @patch('shared.models.registry.urllib.request.urlopen')
    def test_manifest_contains_entry_fields(
        self,
        mock_urlopen: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_urlopen.return_value = _mock_urlopen()

        reg = ModelRegistry(cache_dir=tmp_path)
        reg.get_weights(_MODEL_ID, _FAKE_URL, _FAKE_SHA)

        manifest_path = tmp_path / 'manifest.json'
        assert manifest_path.exists()

        with manifest_path.open() as fh:
            manifest = json.load(fh)

        entry = manifest[_MODEL_ID]
        assert entry['model_id'] == _MODEL_ID
        assert entry['url'] == _FAKE_URL
        assert entry['sha256'] == _FAKE_SHA
        assert entry['filename'] == f'{_MODEL_ID}.pth'
        assert entry['downloaded_at'] is not None

    @patch('shared.models.registry.urllib.request.urlopen')
    def test_no_part_file_after_success(
        self,
        mock_urlopen: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_urlopen.return_value = _mock_urlopen()

        reg = ModelRegistry(cache_dir=tmp_path)
        reg.get_weights(_MODEL_ID, _FAKE_URL, _FAKE_SHA)

        part = tmp_path / f'{_MODEL_ID}.part'
        assert not part.exists()

    @patch('shared.models.registry.urllib.request.urlopen')
    def test_no_part_file_after_failure(
        self,
        mock_urlopen: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_urlopen.return_value = _mock_urlopen()

        reg = ModelRegistry(cache_dir=tmp_path)
        with pytest.raises(ModelIntegrityError):
            reg.get_weights(
                _MODEL_ID,
                _FAKE_URL,
                'bad_hash',
            )

        part = tmp_path / f'{_MODEL_ID}.part'
        assert not part.exists()

    @patch('shared.models.registry.FileLock')
    @patch('shared.models.registry.urllib.request.urlopen')
    def test_filelock_used_per_model(
        self,
        mock_urlopen: MagicMock,
        mock_filelock: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_urlopen.return_value = _mock_urlopen()

        reg = ModelRegistry(cache_dir=tmp_path)
        reg.get_weights(_MODEL_ID, _FAKE_URL, _FAKE_SHA)

        mock_filelock.assert_called_once_with(
            tmp_path / f'.{_MODEL_ID}.lock',
        )
