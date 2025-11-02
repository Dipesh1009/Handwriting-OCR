
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import os

from mltu.configs import BaseModelConfigs
from inferenceModel import ImageToWordModel

app = FastAPI()

# Load the model
configs = BaseModelConfigs.load("Models/04_sentence_recognition/202510280421/configs.yaml")
model = ImageToWordModel(model_path=configs.model_path, char_list=configs.vocab)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    Endpoint to predict text from an image.

    Args:
        file (UploadFile): The image file to be processed.

    Returns:
        JSONResponse: A JSON response containing the predicted text.
    """
    try:
        # Read the image file
        contents = await file.read()
        nparr = np.fromstring(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Make a prediction
        prediction_text = model.predict(image)

        return JSONResponse(content={"prediction": prediction_text})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/")
async def root():
    return {"message": "Handwriting Recognition API"}
