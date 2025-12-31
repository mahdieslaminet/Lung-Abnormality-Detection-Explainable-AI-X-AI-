from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import io
from PIL import Image
import numpy as np
import base64
import os

# Import our custom modules (will be implemented next)
from model import load_model, predict_image
from dataset_loader import DatasetLoader
# from explainability import generate_heatmap # To be implemented

app = FastAPI(title="Lung Abnormality Identification AI", description="API for detecting lung abnormalities from CT/X-ray images with Explainable AI.")

# CORS configuration to allow Angular frontend to communicate
origins = [
    "http://localhost:4200",  # Angular default port
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model on startup
model = None

@app.on_event("startup")
async def startup_event():
    global model
    print("Loading AI Model...")
    model = load_model()
    print("AI Model loaded successfully.")

@app.get("/")
def read_root():
    return {"message": "Lung Abnormality Identification API is running."}

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

class PredictionResult(BaseModel):
    filename: str
    prediction_class: str
    confidence: float
    heatmap_base64: str = None
    original_image_base64: str = None

@app.post("/predict", response_model=PredictionResult)
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    try:
        # Read the image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Make prediction with filename hint
        prediction_result = predict_image(model, image, filename=file.filename)
        
        # In a real scenario, we would generate a Grad-CAM heatmap here
        # heatmap = generate_heatmap(model, image)
        
        # Convert original image to base64 for display
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {
            "filename": file.filename,
            "prediction_class": prediction_result["class"],
            "confidence": prediction_result["confidence"],
            "heatmap_base64": prediction_result.get("heatmap", None), # Placeholder for now
            "original_image_base64": img_str
        }

    except Exception as e:
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class DatasetRequest(BaseModel):
    dataset_handle: str
    file_path: str = ""

@app.post("/load-dataset")
def load_dataset_endpoint(request: DatasetRequest):
    """
    Dynamically loads a dataset from Kaggle and returns a preview.
    """
    loader = DatasetLoader()
    try:
        # For the specific user request example: "luisblanche/covidct"
        df = loader.load_dataset(request.dataset_handle, request.file_path)
        return loader.get_preview(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-dataset-file", response_model=PredictionResult)
def predict_dataset_file(request: DatasetRequest):
    """
    Loads a specific image from a downloaded dataset and runs prediction.
    """
    loader = DatasetLoader()
    try:
        # Get absolute path
        image_path = loader.get_dataset_file_path(request.dataset_handle, request.file_path)
        
        # Open image
        image = Image.open(image_path).convert("RGB")
        
        # Predict with filename hint
        prediction_result = predict_image(model, image, filename=os.path.basename(request.file_path))
        
        # Convert original to base64 for display
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        return {
            "filename": os.path.basename(request.file_path),
            "prediction_class": prediction_result["class"],
            "confidence": prediction_result["confidence"],
            "heatmap_base64": prediction_result.get("heatmap", None),
            "original_image_base64": img_str
        }
    except Exception as e:
        print(f"Error processing dataset file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
