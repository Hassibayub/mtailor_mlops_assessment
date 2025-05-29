import os
import time
from model import ONNXModel


def test_model_predictions(image_path: str, onnx_model: str):
    """Load and test the ONNX model with timing and validation"""
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        if not os.path.exists(onnx_model):
            raise FileNotFoundError(f"ONNX model not found: {onnx_model}")

        model = ONNXModel(onnx_model)

        start_time = time.time()
        
        pred_class, confidence = model.predict(image_path)
        inference_time = time.time() - start_time

        expected_classes = {
            "n01440764_tench.jpeg": 0,
            "n01667114_mud_turtle.jpeg": 35
        }

        # Get expected class for this image
        image_name = os.path.basename(image_path)
        expected_class = expected_classes.get(image_name)

        print(f"\n\n=== Results for {image_path} ===")
        print(f"Prediction: Class {pred_class} (confidence: {confidence:.4f})")
        print(f"Inference time: {inference_time:.3f} seconds")
        
        if expected_class is not None:
            print(f"Expected class: {expected_class}")
            print(f"Prediction {'correct' if pred_class == expected_class else 'incorrect'}")
        
        if inference_time > 3.0:
            print("Warning: Inference time exceeds 3 seconds threshold")

    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        raise 


def main():
    test_images = [
        "n01440764_tench.jpeg",  # should predict class 0
        "n01667114_mud_turtle.jpeg",  # should predict class 35
    ]

    for image in test_images:
        test_model_predictions(
            image_path=image,
            onnx_model="model/model.onnx"
        )


if __name__ == "__main__":
    main()
