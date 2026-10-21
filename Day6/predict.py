
import joblib
import pandas as pd

# Load the trained model
model = joblib.load("../Day3/model/model.joblib")

houses = pd.DataFrame(
    [
        [1500, 3, 2, 5],
        [2000, 4, 5, 4],
        [1000, 2, 10, 3]
    ],
    columns=[
        "area_sqft",
        "bedrooms",
        "age_years",
        "location_score"
    ]
)

predictions = model.predict(houses)

for i, price in enumerate(predictions, start=1):
    print(f"House {i} predicted price: {price:,.2f}")
