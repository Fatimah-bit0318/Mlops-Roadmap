import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


def main():

    # Local dataset
    input_path = "house_data.csv"

    # Read dataset
    df = pd.read_csv(input_path)

    # Features
    X = df[["area_sqft", "bedrooms", "age_years", "location_score"]]

    # Target
    y = df["price"]

    # Split data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Create Random Forest regression model
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    # Train model
    model.fit(X_train, y_train)

    print("Model training completed!")

    # Make predictions
    predictions = model.predict(X_test)

    # Evaluate model
    mse = mean_squared_error(y_test, predictions)

    print("Test MSE:", mse)

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