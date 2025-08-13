"""
FastAPI app to serve /predict endpoint
Run: uvicorn src.api.main:app --reload --port 8000
"""
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io
from src.model.model import predict_image, load_model

app = FastAPI(title="SmartWasteVision API")
# load model on startup
model, classes, device = load_model()

@app.get("/health")
def health():
    return {"status":"ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        return JSONResponse({"error":"invalid image","details":str(e)}, status_code=400)
    res = predict_image(img, model=model, classes=classes, device=device)
    return res
