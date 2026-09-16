import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# 1. Create a small customer dataset
data = {
    "age": [25, 35, 29, 42, 31, 45, 28, 38, 30, 50],
    "salary": [40000, 60000, 50000, 80000, 55000, 90000, 45000, 70000, 52000, 95000],
    "support_calls": [2, 8, 1, 10, 3, 9, 2, 7, 4, 11],
    "churn": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
}


# 2. Convert the data into a DataFrame
df = pd.DataFrame(data)

print("Dataset:")
print(df)


# 3. Select the features
X = df[["age", "salary", "support_calls"]]


# 4. Select the target/label
y = df["churn"]


# 5. Split the data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# 6. Create the Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# 7. Train the model
model.fit(X_train, y_train)


# 8. Make predictions
predictions = model.predict(X_test)


# 9. Calculate accuracy
accuracy = accuracy_score(y_test, predictions)


# 10. Display results
print("\nPredictions:")
print(predictions)

print("\nActual values:")
print(y_test.values)

print("\nAccuracy:")
print(accuracy)
