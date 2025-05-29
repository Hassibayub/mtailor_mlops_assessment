import os
import time
from typing import Any, Dict

from fastapi import FastAPI, File, HTTPException, UploadFile

from model import ONNXModel

app = FastAPI(title="MTailor Model API")

model = None
try:
    model = ONNXModel("model/model.onnx")
except Exception as e:
    print(f"Error loading model: {e}")


@app.get("/")
async def root() -> Dict[str, str]:
    """
    Root endpoint to check if the API is running.

    Returns:
        Dict[str, str]: A dictionary containing a status message
            message: Status message indicating API is running
    """
    return {"message": "MTailor Model API is running"}


@app.get("/health")
async def health() -> str:
    """
    Health check endpoint for monitoring API health.

    Returns:
        str: "OK" if the service is healthy
    """
    return "OK"


@app.get("/ready")
async def ready() -> str:
    """
    Readiness check endpoint to verify if service is ready to accept requests.

    Returns:
        str: "OK" if the service is ready to accept requests
    """
    return "OK"


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Predict image class using ONNX model.

    Args:
        file (UploadFile): The uploaded image file to classify

    Returns:
        Dict[str, Any]: Prediction results containing:
            filename: Name of the uploaded file
            prediction: Dict containing class_id and confidence score
            inference_time: Time taken for prediction in seconds
            status: Status of the prediction request

    Raises:
        HTTPException: If model is not loaded or prediction fails
    """
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # Save uploaded file temporarily
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # Make prediction
        start_time = time.time()
        pred_class, confidence = model.predict(temp_path)
        inference_time = time.time() - start_time

        # Clean up
        os.remove(temp_path)

        return {
            "filename": file.filename,
            "prediction": {"class_id": pred_class, "confidence": round(float(confidence), 3)},
            "inference_time": inference_time,
            "status": "success",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
