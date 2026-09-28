import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


def main():
    # Get training data location from SageMaker when available.
    # If the environment variable is not set, use the current directory.
    input_dir = os.environ.get("SM_CHANNEL_TRAINING", ".")

    input_path = os.path.join(input_dir, "house_data.csv")

    # Load dataset
    df = pd.read_csv(input_path)

    # Features
    X = df[[
        "area_sqft",
        "bedrooms",
        "age_years",
        "location_score"
    ]]

    # Target
    y = df["price"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Create model
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    # Train
    model.fit(X_train, y_train)

    print("Model training completed!")

    # Predict
    predictions = model.predict(X_test)

    # Evaluate
    mse = mean_squared_error(y_test, predictions)
    rmse = mse ** 0.5

    print("MSE:", mse)
    print("RMSE:", rmse)

    # Save model artifact
    model_dir = "model"
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "model.joblib")
    joblib.dump(model, model_path)

    print("Model saved successfully!")
    print("Saved at:", model_path)


if __name__ == "__main__":
    main()
