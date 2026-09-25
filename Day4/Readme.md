## Day 4 – AWS SageMaker Model Deployment & Inference

### Objective
Deploy the trained machine learning model on AWS SageMaker and perform real-time inference using a SageMaker endpoint.

### Work Completed

- Prepared the trained `model.joblib` from Day 3
- Created `inference.py` for model inference
- Packaged `inference.py` and `model.joblib` into `model.tar.gz`
- Stored the model artifact in Amazon S3
- Created a SageMaker Model using ModelBuilder
- Configured the Scikit-learn inference container
- Created a SageMaker endpoint configuration
- Deployed the model as a real-time SageMaker endpoint
- Monitored endpoint deployment using CloudWatch logs
- Troubleshot inference and endpoint health-check issues
- Tested the deployed endpoint using the SageMaker Runtime API
- Successfully received a prediction from the deployed model

### Architecture

Day 3 Trained Model
        ↓
model.joblib
        ↓
inference.py + model.joblib
        ↓
model.tar.gz
        ↓
Amazon S3
        ↓
SageMaker Model
        ↓
Endpoint Configuration
        ↓
SageMaker Endpoint
        ↓
SageMaker Runtime
        ↓
Prediction

### Model Artifact

The final `model.tar.gz` contained:

- `inference.py`
- `model.joblib`

### Endpoint

Endpoint status:

`InService`

### Inference Test

Input:

```text
[1500, 3, 2, 5]