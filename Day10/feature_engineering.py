import pandas as pd

df = pd.read_csv("customer_data.csv")
df["date_of_birth"] = pd.to_datetime(df["date_of_birth"])
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Fixed reference date for reproducible age calculations
today = pd.Timestamp("2026-10-30")
df["age"] = (
    today.year - df["date_of_birth"].dt.year
    - (
        (today.month < df["date_of_birth"].dt.month)
        | ((today.month == df["date_of_birth"].dt.month)
           & (today.day < df["date_of_birth"].dt.day))
    )
)

df["hour"] = df["timestamp"].dt.hour
df["weekday"] = df["timestamp"].dt.day_name()
df["month"] = df["timestamp"].dt.month

feature_df = df[
    ["customer_id", "age", "hour", "weekday", "month", "purchase_amount"]
].copy()

feature_df.to_csv("engineered_features.csv", index=False)

print("Feature engineering completed!")
print(feature_df)
print("\nDataset shape:", feature_df.shape)
print("\nMissing values:")
print(feature_df.isnull().sum())
