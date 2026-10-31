from pathlib import Path
import joblib
import numpy as np

def predict_price(area: float) -> float:
    path = Path(__file__).resolve().parent / "model.joblib"
    if not path.exists():
        raise FileNotFoundError("Run `python train.py` first to create model.joblib.")
    model = joblib.load(path)
    return float(model.predict(np.array([[area]], dtype=float))[0])

if __name__ == "__main__":
    print("Predicted price:", predict_price(800))
