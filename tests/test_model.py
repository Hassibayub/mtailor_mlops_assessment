import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time

import numpy as np
import pytest

from model import ImagePreprocessor, ONNXModel


@pytest.fixture
def model_path():
    return "model/model.onnx"


@pytest.fixture
def test_images():
    return ["n01440764_tench.jpeg", "n01667114_mud_turtle.jpeg"]


@pytest.fixture
def expected_classes():
    return {"n01440764_tench.jpeg": 0, "n01667114_mud_turtle.jpeg": 35}


def test_model_loading(model_path):
    """Test if model loads correctly"""
    model = ONNXModel(model_path)
    assert model is not None


def test_preprocessing(test_images):
    """Test image preprocessing pipeline"""
    preprocessor = ImagePreprocessor()
    for image_path in test_images:
        processed = preprocessor.preprocess(image_path)

        assert processed.shape == (1, 3, 224, 224)


def test_model_predictions(model_path, test_images, expected_classes):
    """Test ONNX model predictions and performance"""
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
