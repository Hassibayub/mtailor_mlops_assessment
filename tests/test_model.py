import os
import sys
from typing import Dict, List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time

import pytest

from model import ImagePreprocessor, ONNXModel


@pytest.fixture
def model_path() -> str:
    """
    Fixture providing path to ONNX model file.

    Returns:
        str: Path to the ONNX model file
    """
    return "model/model.onnx"


@pytest.fixture
def test_images() -> List[str]:
    """
    Fixture providing list of test image paths.

    Returns:
        List[str]: List of image filenames for testing
    """
    return ["n01440764_tench.jpeg", "n01667114_mud_turtle.jpeg"]


@pytest.fixture
def expected_classes() -> Dict[str, int]:
    """
    Fixture providing mapping of image filenames to expected class IDs.

    Returns:
        Dict[str, int]: Mapping of image filenames to their expected class IDs
    """
    return {"n01440764_tench.jpeg": 0, "n01667114_mud_turtle.jpeg": 35}


def test_model_loading(model_path: str) -> None:
    """
    Test if ONNX model loads successfully.

    Args:
        model_path: Path to the ONNX model file

    Raises:
        AssertionError: If model fails to load
    """
    model = ONNXModel(model_path)
    assert model is not None


def test_preprocessing(test_images: List[str]) -> None:
    """
    Test image preprocessing pipeline functionality.

    Verifies that images are correctly preprocessed to the expected shape
    and format required by the model.

    Args:
        test_images: List of test image paths

    Raises:
        AssertionError: If processed image shape doesn't match expected dimensions
    """
    preprocessor = ImagePreprocessor()
    for image_path in test_images:
        processed = preprocessor.preprocess(image_path)

        assert processed.shape == (1, 3, 224, 224)


def test_model_predictions(model_path: str, test_images: List[str], expected_classes: Dict[str, int]) -> None:
    """
    Test ONNX model predictions accuracy and performance.

    Verifies that:
    1. Model predicts correct classes for test images
    2. Inference time is within acceptable threshold
    3. Confidence scores are valid

    Args:
        model_path: Path to the ONNX model file
        test_images: List of test image paths
        expected_classes: Dictionary mapping image names to expected class IDs

    Raises:
        AssertionError: If predictions are incorrect or performance thresholds are not met
    """
    model = ONNXModel(model_path)

    for image_path in test_images:
        assert os.path.exists(image_path), f"Image file not found: {image_path}"
        assert os.path.exists(model_path), f"ONNX model not found: {model_path}"

        model.predict(image_path)

        # Timed run
        start_time = time.time()
        pred_class, confidence = model.predict(image_path)
        inference_time = time.time() - start_time

        # Verify predictions
        image_name = os.path.basename(image_path)
        expected_class = expected_classes[image_name]
        assert pred_class == expected_class, f"Wrong prediction! Expected {expected_class}, got {pred_class}"

        # Verify performance
        assert inference_time < 3.0, "Inference time exceeds threshold"
        assert confidence > 0.0, "Confidence score should be positive"
