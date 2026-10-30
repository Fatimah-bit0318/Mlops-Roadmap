# Day 9 – Model Serving with FastAPI

## Objective
Serve a previously trained house-price prediction model through a REST API using FastAPI, and test it with Python requests and Swagger UI.

## Project structure
```text
Day9/
├── app.py
├── predict.py
├── requirements.txt
├── sample_request.json
├── .gitignore
├── README.md
└── model/
    └── model.joblib   # Add the extracted model here; do not commit unless permitted.
```

## Model details
The exercise uses a previously trained `RandomForestRegressor` saved as `model.joblib`. It expects four features in this order: `area`, `bedrooms`, `bathrooms`, `age`.

The artifact was downloaded from `s3://day04/model.tar.gz` and extracted. Copy `model.joblib` into the `model/` folder, or set the `MODEL_PATH` environment variable to its location.

**Compatibility warning:** the model was saved with scikit-learn 1.6.1, while the SageMaker environment had 1.7.2. Scikit-learn warns that cross-version loading may cause errors or invalid results. For repeatable or production use, serve with the training version or retrain and save with the serving version.

## Install dependencies
```bash
pip install -r requirements.txt
```

## Run the API
For local use:
```bash
SAGEMAKER_PROXY=false python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

For SageMaker JupyterLab, use:
```bash
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```
The app defaults to the SageMaker proxy root path `/jupyterlab/default/proxy/8000`. If port 8000 is already occupied, stop the existing Uvicorn process before starting another one.

## Test with Swagger UI
- Local: `http://127.0.0.1:8000/docs`
- SageMaker: open your SageMaker Studio host with `/jupyterlab/default/proxy/8000/docs`.

Expand **POST /predict**, choose **Try it out**, enter:
```json
{
  "area": 800,
  "bedrooms": 2,
  "bathrooms": 2,
  "age": 10
}
```
Then choose **Execute**.

The successful test in the learning environment returned:
```json
{
  "predicted_price": 3547000.0
}
```
The exact result depends on the model artifact and compatible runtime.

## Test with Python requests
```python
import requests

response = requests.post(
    "http://127.0.0.1:8000/predict",
    json={"area": 800, "bedrooms": 2, "bathrooms": 2, "age": 10},
    timeout=30,
)
print("Status:", response.status_code)
print("Response:", response.json())
```

## API endpoints
| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Health check |
| POST | `/predict` | Validates house details and returns a prediction |
| GET | `/docs` | Interactive Swagger UI |

## What I learned
- Model serving: making a trained ML model available for predictions.
- FastAPI: creating REST API endpoints.
- Pydantic: validating incoming request data.
- Joblib: loading a serialized scikit-learn model.
- Pandas: passing named feature columns to the model.
- Uvicorn: running the FastAPI application.
- Swagger UI: testing endpoints interactively.
- HTTP 200: indicates a successful request.

## Scope
The API was tested locally inside SageMaker Studio. This exercise did not deploy a public production API or create a SageMaker real-time endpoint.
