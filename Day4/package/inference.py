import os
import joblib


def model_fn(model_dir):
    """
    Load the trained model.
    """
    model_path = os.path.join(model_dir, "model.joblib")
    model = joblib.load(model_path)

    return model


def predict_fn(input_data, model):
    """
    Generate predictions.
    """
    predictions = model.predict(input_data)

    return predictions