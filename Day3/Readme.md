# Day 3 — SageMaker Training Jobs

## Objective

Learn how to configure and prepare an ML training job using Amazon SageMaker.

## What I Learned

- SageMaker Training Jobs
- SageMaker SDK
- SageMaker ModelTrainer
- Training images
- Source code configuration
- Training data configuration
- Compute configuration
- Output configuration
- S3 input and output
- IAM execution roles
- AWS Service Quotas

## Project

House Price Prediction

### Algorithm

Random Forest Regression

### Input

House-price dataset stored in Amazon S3.

### Training Script

`train.py`

The training script:

1. Reads the training data
2. Separates features and target
3. Splits the dataset
4. Trains a Random Forest Regressor
5. Evaluates the model
6. Saves the trained model

## SageMaker Training Architecture

S3
↓
Training Data
↓
SageMaker Training Job
↓
Training Container
↓
train.py
↓
Random Forest Model
↓
Model Artifact
↓
S3

## Configuration

- Framework: Scikit-learn
- Training image: SageMaker Scikit-learn image
- Compute: `ml.c5.xlarge`
- Input: S3
- Output: S3
- Training script: `train.py`

## Troubleshooting

### S3 AccessDenied

The SageMaker execution role initially did not have permission to perform:

`s3:GetObject`

The IAM permissions were updated to allow SageMaker to access the required S3 object.

### Python Version Issue

The initially requested Python version `py312` was not supported by the selected SageMaker image configuration. The supported Python option was used instead.

### Source Directory Issue

The initial `source_dir` pointed to a directory that did not exist. The configuration was corrected to use the actual local source directory containing `train.py`.

### Service Quota Issue

The final training job could not be launched because the account had a quota of `0` for:

`ml.c5.xlarge for spot training job usage`

The training configuration itself was successfully created.

## Key Learning

SageMaker allows ML training to be configured as a managed AWS training job using S3, IAM, training containers, compute resources and model output configuration.