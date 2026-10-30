import os
from pathlib import Path

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE_DIR / "model" / "model.joblib"))
model = joblib.load(MODEL_PATH)


def predict_house_price(area, bedrooms, bathrooms, age):
    house = pd.DataFrame(
        [[area, bedrooms, bathrooms, age]],
        columns=["area", "bedrooms", "bathrooms", "age"],
    )
    return float(model.predict(house)[0])


if __name__ == "__main__":
    result = predict_house_price(800, 2, 2, 10)
    print("Predicted House Price:", result)
