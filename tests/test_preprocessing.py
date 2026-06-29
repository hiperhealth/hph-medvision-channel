"""Unit tests for shared.preprocessing module."""

from pathlib import Path

import numpy as np
import pytest
import torch

from shared.preprocessing import (
    DEFAULT_THRESHOLDS,
    ImagePreprocessor,
    ImageQualityError,
    QualityThresholds,
    build_inference_transforms,
    correct_exif_orientation,
    validate_image,
)


class TestQualityThresholds:
    """Tests for QualityThresholds dataclass."""

    def test_defaults(self) -> None:
        t = QualityThresholds()
        assert t.min_resolution == 224
        assert t.max_aspect_ratio == 3.0
        assert t.min_laplacian_var == 100.0
        assert '.jpg' in t.allowed_formats

    def test_frozen(self) -> None:
        t = QualityThresholds()
        with pytest.raises(AttributeError):
            t.min_resolution = 100  # type: ignore[misc]

    def test_custom_values(self) -> None:
        t = QualityThresholds(min_resolution=100, max_aspect_ratio=5.0)
        assert t.min_resolution == 100
        assert t.max_aspect_ratio == 5.0

    def test_default_thresholds_is_singleton(self) -> None:
        assert DEFAULT_THRESHOLDS == QualityThresholds()


class TestImageQualityError:
    """Tests for ImageQualityError exception."""

    def test_reason_stored(self) -> None:
        err = ImageQualityError(reason='too small')
        assert err.reason == 'too small'
        assert 'too small' in str(err)

    def test_detail_stored(self) -> None:
        detail = {'check': 'resolution', 'actual': (50, 50)}
        err = ImageQualityError(reason='too small', detail=detail)
        assert err.detail == detail

    def test_detail_defaults_to_empty_dict(self) -> None:
        err = ImageQualityError(reason='test')
        assert err.detail == {}


class TestValidateImage:
    """Tests for validate_image function."""

    def test_valid_image(self, valid_image: Path) -> None:
        result = validate_image(valid_image)
        assert isinstance(result, np.ndarray)
        assert result.ndim == 3

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            validate_image(tmp_path / 'nonexistent.jpg')

    def test_below_min_resolution(self, small_image: Path) -> None:
        with pytest.raises(ImageQualityError) as exc_info:
            validate_image(small_image)
        assert exc_info.value.detail['check'] == 'resolution'

    def test_bad_aspect_ratio(self, narrow_image: Path) -> None:
        with pytest.raises(ImageQualityError) as exc_info:
            validate_image(narrow_image)
        assert exc_info.value.detail['check'] == 'aspect_ratio'

    def test_blurry_image(self, blurry_image: Path) -> None:
        with pytest.raises(ImageQualityError) as exc_info:
            validate_image(blurry_image)
        assert exc_info.value.detail['check'] == 'blur'

    def test_unsupported_format(self, non_image_file: Path) -> None:
        with pytest.raises(ImageQualityError) as exc_info:
            validate_image(non_image_file)
        assert exc_info.value.detail['check'] == 'format'

    def test_corrupt_file(self, corrupt_image: Path) -> None:
        with pytest.raises(ImageQualityError) as exc_info:
            validate_image(corrupt_image)
        assert exc_info.value.detail['check'] == 'decode'

    def test_custom_thresholds(self, valid_image: Path) -> None:
        lenient = QualityThresholds(min_laplacian_var=0.0)
        result = validate_image(valid_image, thresholds=lenient)
        assert isinstance(result, np.ndarray)

    def test_accepts_string_path(self, valid_image: Path) -> None:
        result = validate_image(str(valid_image))
        assert isinstance(result, np.ndarray)

    def test_rgba_handled(self, rgba_png: Path) -> None:
        result = validate_image(rgba_png)
        assert result.shape[2] == 3


class TestExifOrientationCorrection:
    """Tests for correct_exif_orientation function."""

    def test_no_exif_returns_unchanged(self, valid_image: Path) -> None:
        import cv2

        original = cv2.imread(str(valid_image), cv2.IMREAD_COLOR)
        result = correct_exif_orientation(original, valid_image)
        np.testing.assert_array_equal(result, original)

    def test_non_image_path_returns_unchanged(self, tmp_path: Path) -> None:
        fake_image = np.zeros((100, 100, 3), dtype=np.uint8)
        fake_path = tmp_path / 'nonexistent.jpg'
        result = correct_exif_orientation(fake_image, fake_path)
        np.testing.assert_array_equal(result, fake_image)


class TestBuildInferenceTransforms:
    """Tests for build_inference_transforms function."""

    def test_output_shape_default(self) -> None:
        transforms = build_inference_transforms()
        img = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        result = transforms(img)
        assert result.shape == (3, 224, 224)

    def test_output_shape_custom_size(self) -> None:
        transforms = build_inference_transforms(target_size=128)
        img = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        result = transforms(img)
        assert result.shape == (3, 128, 128)

    def test_output_is_float(self) -> None:
        transforms = build_inference_transforms()
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        result = transforms(img)
        assert result.dtype in (torch.float32, np.float32)


class TestImagePreprocessor:
    """Tests for ImagePreprocessor class."""

    def test_returns_tensor(self, valid_image: Path) -> None:
        preprocessor = ImagePreprocessor()
        result = preprocessor.preprocess(valid_image)
        assert isinstance(result, torch.Tensor)

    def test_output_shape(self, valid_image: Path) -> None:
        preprocessor = ImagePreprocessor()
        result = preprocessor.preprocess(valid_image)
        assert result.shape == (1, 3, 224, 224)

    def test_custom_target_size(self, valid_image: Path) -> None:
        preprocessor = ImagePreprocessor(target_size=128)
        result = preprocessor.preprocess(valid_image)
        assert result.shape == (1, 3, 128, 128)

    def test_validation_failure_propagates(self, small_image: Path) -> None:
        preprocessor = ImagePreprocessor()
        with pytest.raises(ImageQualityError):
            preprocessor.preprocess(small_image)

    def test_validate_returns_ndarray(self, valid_image: Path) -> None:
        preprocessor = ImagePreprocessor()
        result = preprocessor.validate(valid_image)
        assert isinstance(result, np.ndarray)

    def test_file_not_found(self, tmp_path: Path) -> None:
        preprocessor = ImagePreprocessor()
        with pytest.raises(FileNotFoundError):
            preprocessor.preprocess(tmp_path / 'missing.jpg')
