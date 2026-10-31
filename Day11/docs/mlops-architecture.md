# Day 11 – MLOps architecture notes

## CI – Continuous Integration
Changes to training code, pipeline code, inference code, Dockerfile or tests trigger lint/syntax checks, unit tests, Docker builds, security scanning and pipeline validation.

## CD – Continuous Delivery / Deployment
```text
Model Registry → Model Approval → Deployment Pipeline → SageMaker Endpoint
```
The endpoint hosts the approved model for predictions.

## CT – Continuous Training
Possible triggers:
- Scheduled training
- New dataset
- Data/model drift
- Performance degradation
- Manual trigger

```text
New Data / Schedule / Drift / Manual Trigger
                    ↓
             Training Pipeline
                    ↓
                 Evaluate
                    ↓
             Model Registry
                    ↓
                Approval
                    ↓
                Deployment
```

## AWS service roles
| Service | Role |
|---|---|
| GitHub Actions | Runs CI checks and can coordinate deployment |
| Terraform | Defines infrastructure as code |
| SageMaker Pipelines | Orchestrates processing, training and evaluation |
| SageMaker Model Registry | Versions model packages and tracks approval |
| SageMaker Endpoint | Hosts a model for online inference |
| EventBridge | Routes scheduled or event-based triggers |
| Lambda | Runs a small function in response to an event |
| Step Functions | Coordinates multi-step workflows and branching |

EventBridge, Lambda and Step Functions are not all required in every architecture.

## Event-driven examples
New S3 dataset → EventBridge → Lambda (or supported direct target) → Start SageMaker Pipeline.

Model approved → EventBridge → deployment workflow → update/deploy endpoint.

## Cost note
This project is a local CI exercise and architecture study. It does not provision AWS resources or deploy an endpoint. Obtain approval and review pricing before creating SageMaker endpoints, pipelines or other billable AWS resources.
