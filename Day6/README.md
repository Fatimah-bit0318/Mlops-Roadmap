# Day 6 – Local Inference and SageMaker Endpoints

## Objective
Understand how a trained machine learning model makes predictions and how Amazon SageMaker endpoints serve predictions through AWS.

## Tasks Completed

### 1. Local Model Inference
- Loaded the trained house-price prediction model from `Day3/model/model.joblib`.
- Created `Day6/predict.py` to make predictions using Python.
- Passed house details to the model using the following features:
  - `area_sqft`
  - `bedrooms`
  - `age_years`
  - `location_score`

**Sample input:** `[1500, 3, 2, 5]`

**Output:**
```text
Predicted house price: 9,319,000.00
```

### 2. Understanding SageMaker Endpoints
Learned that a SageMaker endpoint hosts a trained model on AWS and allows applications to send prediction requests and receive results.

### 3. SageMaker Components
- **Model:** Defines the trained model artifact and inference container.
- **Endpoint Configuration:** Defines the hosting setup, including instance type and instance count.
- **Endpoint:** Hosts the model and serves inference requests.

Typical deployment flow:

`Model → Endpoint Configuration → Endpoint → InService`

### 4. Local vs. Hosted Inference
- **Local inference:** Runs the model on a local computer using Python.
- **Hosted inference:** Runs the model on AWS infrastructure and accepts requests from applications.

Also studied the purpose of the SageMaker Runtime `InvokeEndpoint` API for sending inference requests to a deployed endpoint.

### 5. AWS Cost Awareness
- No new SageMaker endpoint or AWS resources were created on Day 6.
- A live endpoint invocation was not tested because no active endpoint was available.
- Local inference was completed successfully without creating a new hosting resource.

## Key Learnings
- Inference means using a trained model to predict outcomes for new data.
- A SageMaker endpoint makes a model accessible for hosted predictions.
- Training and inference are separate processes.
- A provisioned real-time endpoint can incur hosting charges even when it receives no requests.

## Tools and Technologies
- Python
- Pandas
- Joblib
- Scikit-learn
- Amazon SageMaker
- Amazon S3
- AWS SageMaker Runtime (conceptual study)

## Status
**Completed:** Local inference and SageMaker endpoint concepts.
