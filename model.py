import onnxruntime
import numpy as np
from PIL import Image
from typing import Tuple


class ImagePreprocessor:
    def __init__(self):
        self.input_size = (224, 224)
        self.mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)

    def preprocess(self, image_path: str) -> np.ndarray:
        """
        Preprocess image for model inference.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Preprocessed image as numpy array in NCHW format
        """
        # Load and convert to RGB
        image = Image.open(image_path).convert('RGB')
        
        # Resize using bilinear interpolation
        image = image.resize(self.input_size, Image.BILINEAR)
        
        # Convert to numpy and normalize (float32)
        image_array = np.array(image, dtype=np.float32) / 255.0
        
        # Apply ImageNet normalization (float32)
        mean = self.mean.astype(np.float32)
        std = self.std.astype(np.float32)
        normalized = (image_array - mean) / std
        
        # Add batch dimension and transpose to NCHW format
        return np.expand_dims(normalized.transpose(2, 0, 1), 0).astype(np.float32)


class ONNXModel:
    """Handles ONNX model loading and inference."""
    
    def __init__(self, model_path: str):
        """
        Initialize ONNX model.
        
        Args:
            model_path: Path to ONNX model file
        """
        self.session = onnxruntime.InferenceSession(model_path)
        self.preprocessor = ImagePreprocessor()

    def predict(self, image_path: str) -> Tuple[int, float]:
        """
        Run inference on input image.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Tuple of (predicted_class_id, confidence_score)
        """
        # Preprocess image
        input_tensor = self.preprocessor.preprocess(image_path)
        
        # Get input name from model
        input_name = self.session.get_inputs()[0].name
        
        # Run inference
        outputs = self.session.run(None, {input_name: input_tensor})
        
        # Get predicted class and confidence
        probabilities = outputs[0][0]
        predicted_class = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_class])
        
        return predicted_class, confidence
