"""Shared pytest fixtures for MedVision tests."""

from pathlib import Path

import cv2
import numpy as np
import pytest


@pytest.fixture()
def valid_image(tmp_path: Path) -> Path:
    """256x256 RGB JPEG with sharp edges (high Laplacian variance)."""
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (200, 200), (255, 255, 255), -1)
    cv2.line(img, (0, 0), (255, 255), (128, 128, 128), 2)
    path = tmp_path / 'valid.jpg'
    cv2.imwrite(str(path), img)
    return path


@pytest.fixture()
def small_image(tmp_path: Path) -> Path:
    """50x50 image — below the 224px minimum resolution."""
    img = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
    path = tmp_path / 'small.jpg'
    cv2.imwrite(str(path), img)
    return path


@pytest.fixture()
def blurry_image(tmp_path: Path) -> Path:
    """256x256 image with heavy Gaussian blur (low Laplacian)."""
    img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    img = cv2.GaussianBlur(img, (31, 31), sigmaX=15)
    path = tmp_path / 'blurry.jpg'
    cv2.imwrite(str(path), img)
    return path


@pytest.fixture()
def narrow_image(tmp_path: Path) -> Path:
    """224x800 image — exceeds the 3.0 max aspect ratio."""
    img = np.random.randint(0, 255, (224, 800, 3), dtype=np.uint8)
    path = tmp_path / 'narrow.jpg'
    cv2.imwrite(str(path), img)
    return path


@pytest.fixture()
def rgba_png(tmp_path: Path) -> Path:
    """256x256 RGBA PNG image."""
    img = np.random.randint(0, 255, (256, 256, 4), dtype=np.uint8)
    path = tmp_path / 'rgba.png'
    cv2.imwrite(str(path), img)
    return path


@pytest.fixture()
def non_image_file(tmp_path: Path) -> Path:
    """Plain text file with unsupported extension."""
    path = tmp_path / 'notes.txt'
    path.write_text('not an image')
    return path


@pytest.fixture()
def corrupt_image(tmp_path: Path) -> Path:
    """File with .jpg extension but random bytes inside."""
    path = tmp_path / 'corrupt.jpg'
    path.write_bytes(b'\x00\x01\x02\x03\x04\x05')
    return path
