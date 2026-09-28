Day 5 — Model Artifacts, Evaluation and Model Registry

Objective

Learn how to evaluate a trained machine-learning model, save the trained model as an artifact, store the artifact in Amazon S3, and register/version the model using Amazon SageMaker Model Registry.

Dataset

The project uses house_data.csv with these columns:

area_sqft

bedrooms

age_years

location_score

price

Features:

X = area_sqft, bedrooms, age_years, location_score

Target:

y = price

What I did

1. Loaded and prepared the dataset

The dataset was loaded with pandas. The four house attributes were selected as features and price was selected as the target.

2. Split the data

I used train_test_split() with:

test_size=0.2
random_state=42

With 20 records, this produced approximately 16 training records and 4 test records.

3. Trained a Random Forest Regressor

I trained:

RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

4. Evaluated the model

The model generated predictions on the test set.

The evaluation results from the hands-on were:

MSE  = 309401250000.0
RMSE = 556238.4830268398

I also compared actual and predicted prices.

Example results:

Actual Price

Predicted Price

4,500,000

4,720,000

12,000,000

11,010,000

6,500,000

6,856,000

5,500,000

5,787,000

5. Saved the model artifact

The trained model was saved using Joblib:

model.joblib

The model artifact was created at:

/home/sagemaker-user/day5_artifact/model.joblib

6. Packaged the model

The model was packaged into:

model.tar.gz

The archive was verified and contained:

model.joblib

7. Uploaded the artifact to Amazon S3

The final S3 location was:

s3://day04/day5/model.tar.gz

8. Created a SageMaker Model Registry Model Group

The Model Group created was:

Day5-HousePrice-Models

9. Registered Model Version 1

The model was registered as:

Day5-HousePrice-Models
└── Version 1

The registered model package ARN was:

arn:aws:sagemaker:us-east-1:834176180969:model-package/Day5-HousePrice-Models/1

10. Managed model approval

The model initially had:

PendingManualApproval

It was then updated to:

Approved

Final state:

Model Group: Day5-HousePrice-Models
Version: 1
Status: Completed
Approval: Approved
Artifact: s3://day04/day5/model.tar.gz

Day 5 Architecture

house_data.csv
      |
      v
Train/Test Split
      |
      v
Random Forest Regressor
      |
      +------> Predictions
      |            |
      |            v
      |       MSE / RMSE
      |
      v
model.joblib
      |
      v
model.tar.gz
      |
      v
Amazon S3
      |
      v
SageMaker Model Registry
      |
      v
Day5-HousePrice-Models
      |
      v
Version 1
      |
      v
Approved

Important Concepts Learned

Model artifact

A saved representation of the trained model, such as model.joblib or a packaged model.tar.gz.

S3

Used to store the model artifact so it is available as a persistent AWS object.

Model Registry

Used to organize and manage registered model versions and associated metadata, including the artifact location, inference container information, and approval status.

Model Group

A logical group containing different versions of a model.

Model Approval

A model version can have an approval state such as PendingManualApproval or Approved.

What was NOT done on Day 5

This Day 5 exercise did not create a new SageMaker training job or deploy the registered model to a new endpoint. Endpoint deployment/invocation was covered in the earlier Day 4 work.

Result

Day 5 was completed by training and evaluating the house-price model, creating the model artifact, storing it in S3, registering Version 1 in SageMaker Model Registry, and approving the registered model.