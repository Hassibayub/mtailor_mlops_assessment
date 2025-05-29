import os

import numpy as np
import onnxruntime
import torch
import torch.onnx
from PIL import Image

from pytorch_model import BasicBlock, Classifier


def convert_to_onnx(model_path: str, output_path: str):
    """
    Convert PyTorch model to ONNX format
    Args:
        model_path: Path to PyTorch model weights
        output_path: Path to save ONNX model
    """
    # Initialize model
    model = Classifier(BasicBlock, [2, 2, 2, 2])
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # dummy input (batch_size=1, channels=3, height=224, width=224)
    dummy_input = torch.randn(1, 3, 224, 224)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},  # variable length axes
            "output": {0: "batch_size"},
        },
    )
    print(f"Model converted and saved to {output_path}")


def verify_onnx_model(onnx_path: str, test_image_path: str):
    """
    Verify ONNX model produces same output as PyTorch model
    Args:
        onnx_path: Path to ONNX model
        test_image_path: Path to test image
    """

    # Load PyTorch model to compare the results
    pytorch_model = Classifier(BasicBlock, [2, 2, 2, 2])
    pytorch_model.load_state_dict(torch.load("model/pytorch_model_weights.pth"))
    pytorch_model.eval()

    img = Image.open(test_image_path)
    input_tensor = pytorch_model.preprocess_numpy(img).unsqueeze(0)

    # PyTorch prediction
    with torch.no_grad():
        pytorch_output = pytorch_model(input_tensor)
        pytorch_prediction = torch.argmax(pytorch_output).item()

    # ONNX Runtime prediction
    ort_session = onnxruntime.InferenceSession(onnx_path)
    ort_inputs = {ort_session.get_inputs()[0].name: input_tensor.numpy()}
    ort_output = ort_session.run(None, ort_inputs)
    onnx_prediction = np.argmax(ort_output[0])

    print(f"PyTorch prediction: {pytorch_prediction}")
    print(f"ONNX prediction: {onnx_prediction}")
    print(f"Predictions match: {pytorch_prediction == onnx_prediction}")


if __name__ == "__main__":
    model_path = "model/pytorch_model_weights.pth"
    onnx_path = "model/model.onnx"
    test_image_path = "n01667114_mud_turtle.JPEG"

    convert_to_onnx(model_path, onnx_path)

    verify_onnx_model(onnx_path, test_image_path)
