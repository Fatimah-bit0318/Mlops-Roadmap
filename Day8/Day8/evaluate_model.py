import csv
import json
import math
import os
import tarfile

import xgboost as xgb

MODEL_DIR = "/opt/ml/processing/model"
TEST_FILE = "/opt/ml/processing/test/test.csv"
OUTPUT_DIR = "/opt/ml/processing/evaluation"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    archives = [
        name for name in os.listdir(MODEL_DIR)
        if name.endswith(".tar.gz")
    ]
    if not archives:
        raise FileNotFoundError(f"No model.tar.gz found in {MODEL_DIR}")

    with tarfile.open(os.path.join(MODEL_DIR, archives[0]), "r:gz") as archive:
        archive.extractall(MODEL_DIR, filter="data")

    model_path = os.path.join(MODEL_DIR, "xgboost-model")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"XGBoost model file not found: {model_path}")

    model = xgb.Booster()
    model.load_model(model_path)

    actual = []
    features = []
    # Expected XGBoost CSV format: target first, then feature columns; no header.
    with open(TEST_FILE, newline="") as file:
        for row in csv.reader(file):
            if row:
                values = [float(value) for value in row]
                actual.append(values[0])
                features.append(values[1:])

    if not actual:
        raise ValueError("The test dataset is empty.")

    predictions = model.predict(xgb.DMatrix(features))
    rmse = math.sqrt(
        sum((target - float(prediction)) ** 2
            for target, prediction in zip(actual, predictions)) / len(actual)
    )

    report = {
        "regression_metrics": {"rmse": rmse},
        "test_rows": len(actual),
        "model_evaluated": "trained XGBoost model",
    }
    output_path = os.path.join(OUTPUT_DIR, "evaluation.json")
    with open(output_path, "w") as file:
        json.dump(report, file, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
