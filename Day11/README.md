# Day 11 – CI/CD/CT for MLOps

## Goal
Learn and practise CI (Continuous Integration), CD (Continuous Delivery/Deployment), and CT (Continuous Training).

## Files
- `.github/workflows/day11-ci.yml` – GitHub Actions CI workflow.
- `train.py` – trains a small sample regression model.
- `inference.py` – loads the saved model and predicts.
- `tests/test_ml.py` – unit tests.
- `requirements.txt` – Python dependencies.
- `Dockerfile` – packages the example.
- `docs/mlops-architecture.md` – CD, CT and AWS service notes.

## Run locally
From inside `Day11`:
```bash
python -m pip install -r requirements.txt
python train.py
python inference.py
python -m pytest -q
```

## Enable GitHub Actions
The workflow must be at the repository root path `.github/workflows/day11-ci.yml`.
Copy `Day11/.github/workflows/day11-ci.yml` to `Mlops-Roadmap/.github/workflows/day11-ci.yml`. Keep the rest of the project inside `Day11/`.
Then commit and push to `main`, open GitHub → Actions → **Day 11 MLOps CI**, and inspect the run.

The workflow checks syntax, runs tests, trains the example model, verifies its artifact, and builds a Docker image.

## What is covered
- **Implemented:** a small CI workflow with GitHub Actions.
- **Studied/documented:** CD from Model Registry → approval → deployment pipeline → SageMaker Endpoint.
- **Studied/documented:** CT triggers such as schedule, new data, drift, performance degradation and manual trigger; event-driven architecture using EventBridge, Lambda and Step Functions.

This project does not create SageMaker Pipelines, Model Registry resources, endpoints, EventBridge rules, Lambda functions or Step Functions state machines. No AWS resources are provisioned. GitHub Actions usage may count against your GitHub plan's included minutes.
