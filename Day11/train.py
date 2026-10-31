from pathlib import Path
import joblib
import numpy as np
from sklearn.linear_model import LinearRegression

def train_model():
    X = np.array([[500], [700], [900], [1100], [1300]], dtype=float)
    y = np.array([1500000, 2100000, 2700000, 3300000, 3900000], dtype=float)
    model = LinearRegression()
    model.fit(X, y)
    output = Path(__file__).resolve().parent / "model.joblib"
    joblib.dump(model, output)
    return output

if __name__ == "__main__":
    print("Training completed. Model saved to:", train_model())
