from pathlib import Path
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).parent
training = pd.read_csv(ROOT / "data" / "training_data.csv")

X_train = training[["income", "transaction_amount"]]
y_train = training["fraud"]

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

model_path = ROOT / "model.joblib"
joblib.dump(model, model_path)
print(f"Training completed. Model saved to: {model_path}")
