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
    Root endpoint to check if the API is running
    """
    return {"message": "MTailor Model API is running"}


@app.get("/health")
def health():
    return "OK"

@app.get("/ready")
def ready():
    return "OK"

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Predict image class using ONNX model
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
            "prediction": {"class_id": pred_class, "confidence": round(float(confidence),3)},
            "inference_time": inference_time,
            "status": "success",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
