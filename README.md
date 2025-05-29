# MTailor Model API

## Introduction

This project provides a FastAPI-based web service for image classification using an ONNX model. The service loads a pre-trained deep learning model and exposes endpoints for making predictions on uploaded images. The model is a ResNet-like architecture converted from PyTorch to ONNX format for optimized inference.

## Setup

### Prerequisites

- Python 3.8+
- uv (Python package manager)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/hassibayub/mtailor_mlops_assessment.git
   cd mtailor_mlops_assessment
   ```

2. Install dependencies using uv:
   ```bash
   # Install uv if you don't have it
   pip install uv
   
   # Create a virtual environment and install dependencies
   uv sync
   ```

3. Make sure the model file exists at `model/model.onnx`. If you need to convert a PyTorch model to ONNX format, use the provided script:
   ```bash
   python convert_to_onnx.py
   ```

## Running the API Locally

Start the FastAPI server:

```bash
python main.py
```

The API will be available at http://localhost:8080. You can access the interactive API documentation at http://localhost:8080/docs.

## Docker

### Building the Docker Image

Build a Docker image for the application:

```bash
docker build -t mtailor-model-api .
```

### Running the Container

Run the Docker container:

```bash
docker run -p 8080:8080 mtailor-model-api
```

## API Endpoints

### Root Endpoint (/)

- **Method**: GET
- **Description**: Check if the API is running
- **Response**:
  ```json
  {
    "message": "MTailor Model API is running"
  }
  ```

### Health Check (/health)

- **Method**: GET
- **Description**: Health check endpoint
- **Response**: "OK"

### Readiness Check (/ready)

- **Method**: GET
- **Description**: Readiness check endpoint
- **Response**: "OK"

### Predict (/predict)

- **Method**: POST
- **Description**: Predict the class of an uploaded image
- **Request**: 
  - Form data with a file parameter named "file"
- **Response**:
  ```json
  {
    "filename": "example.jpg",
    "prediction": {
      "class_id": 123,
      "confidence": 0.985
    },
    "inference_time": 0.125,
    "status": "success"
  }
  ```

## End-to-End Testing

The project includes an end-to-end test script (`e2e_test.py`) that tests the deployed API:

```bash
python e2e_test.py
```

This script tests the `/predict` endpoint by sending a test image to the deployed API and verifies that the response contains the expected fields.

## Cerebrium Deployment

This project is set up for deployment on Cerebrium, a MLOps platform for easy model deployment.

### Deployment Details

- **Project ID**: p-9d0ef0c4
- **API Endpoint**: https://api.cortex.cerebrium.ai/v4/p-9d0ef0c4/app/predict
- **Authentication**: Bearer token required in Authorization header

### Testing the Deployed Model

You can test the deployed model using the `e2e_test.py` script, which sends a test image to the Cerebrium endpoint and verifies the response.

### Making API Calls to the Deployed Model

Example API call using curl:

```bash
curl -X POST \
  https://api.cortex.cerebrium.ai/v4/p-9d0ef0c4/app/predict \
  -H 'Authorization: Bearer YOUR_TOKEN_HERE' \
  -F 'file=@path/to/your/image.jpg'
```

Replace `YOUR_TOKEN_HERE` with your actual Cerebrium API token and `path/to/your/image.jpg` with the path to the image you want to classify.