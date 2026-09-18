import os
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier


def main():

    # Local dataset
    input_path = "customer_data.csv"

    # Read dataset
    df = pd.read_csv(input_path)

    # Features
    X = df[["age", "salary", "support_calls"]]

    # Target
    y = df["churn"]

    # Create Random Forest model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    # Train model
    model.fit(X, y)

    print("Model training completed!")

    # Create local model directory
    model_dir = "model"
    os.makedirs(model_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(model_dir, "model.joblib")
    joblib.dump(model, model_path)

    print("Model saved successfully!")
    print("Saved at:", model_path)


if __name__ == "__main__":
    main()