import os
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE_DIR / "model" / "model.joblib"))

# Defaults to the SageMaker JupyterLab proxy used in this exercise.
SAGEMAKER_PROXY = os.getenv("SAGEMAKER_PROXY", "true").lower() == "true"
ROOT_PATH = "/jupyterlab/default/proxy/8000" if SAGEMAKER_PROXY else ""

app = FastAPI(title="House Price Prediction API", root_path=ROOT_PATH)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}. "
        "Copy model.joblib into model/ or set MODEL_PATH."
    )

model = joblib.load(MODEL_PATH)


class HouseData(BaseModel):
    area: float
    bedrooms: int
    bathrooms: int
    age: int


@app.get("/")
def home():
    return {"message": "House Price Prediction API is running"}


@app.post("/predict")
def predict(data: HouseData):
    house = pd.DataFrame(
        [[data.area, data.bedrooms, data.bathrooms, data.age]],
        columns=["area", "bedrooms", "bathrooms", "age"],
    )
    prediction = model.predict(house)
    return {"predicted_price": float(prediction[0])}
