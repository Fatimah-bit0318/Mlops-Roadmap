# Day 7 – SageMaker Inference Types

## Objective
Understand the different inference options available in Amazon SageMaker and when to use each one.

## Topics Covered

### 1. Real-Time Inference
- Returns predictions to applications that need immediate responses.
- Suitable for websites, fraud detection, and recommendation systems.
- Uses a persistent SageMaker endpoint.

### 2. Serverless Inference
- Suitable for intermittent or unpredictable traffic.
- AWS manages the underlying compute capacity.
- Cold starts may occur.

### 3. Asynchronous Inference
- Suitable for large inputs or longer-running prediction requests.
- Requests are queued and processed asynchronously.
- Results can be stored in Amazon S3.

### 4. Batch Transform
- Used to generate predictions for large datasets.
- Reads input data from Amazon S3 and writes prediction results to S3.
- Does not require a permanent inference endpoint.

## Comparison

| Inference Type | Main Use Case |
|---|---|
| Real-time | Immediate predictions |
| Serverless | Intermittent traffic |
| Asynchronous | Large or long-running requests |
| Batch Transform | Bulk offline predictions |

## Key Learnings
- Inference uses an already-trained model to generate predictions.
- The appropriate method depends on response time, input size, traffic, and workload requirements.
- Batch Transform is useful for processing many records without maintaining a permanent endpoint.

## Practical Status
Completed conceptual learning and scenario-based practice. No new AWS resources were created, and no live inference jobs were executed.