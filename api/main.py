from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
import sys
sys.path.append("..")
from utils.inference import ViTClassifier

app = FastAPI()
model = ViTClassifier()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    pred, conf, _ = model.predict(image)
    return {"prediction": pred, "confidence": conf}