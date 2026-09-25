import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


# SageMaker paths
input_path = r"C:\Users\fatim\OneDrive\Desktop\Mlops\Day4\house_data.csv"
model_dir = "model"


# Load dataset
df = pd.read_csv(input_path)

print("Dataset loaded:")
print(df.head())


# Features and target
X = df[["area", "bedrooms", "bathrooms", "age"]]
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


# Evaluate
predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions)

print("Mean Squared Error:", mse)


# Save model
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, "model.joblib")

joblib.dump(model, model_path)

print("Model saved to:", model_path)