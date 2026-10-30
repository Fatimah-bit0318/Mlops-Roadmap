# Day 10 – Feature Engineering + SageMaker Feature Store

## Goal
Transform raw customer data into reusable machine-learning features and understand how they map to SageMaker Feature Store.

## Files
- `customer_data.csv` – sample raw customer data.
- `feature_engineering.ipynb` – step-by-step notebook.
- `feature_engineering.py` – equivalent Python script.
- `engineered_features.csv` – generated after running the notebook or script.

## Run
1. Open this folder in SageMaker JupyterLab.
2. Open `feature_engineering.ipynb` and run all cells in order.
3. Confirm that `engineered_features.csv` is created.
4. Alternatively, run `python feature_engineering.py` from a terminal.

## Features created
- `age` from `date_of_birth`
- `hour`, `weekday`, and `month` from `timestamp`
- `purchase_amount` retained as a numeric feature

The example uses 30 October 2026 as a fixed reference date so age calculations are reproducible.

## Feature Store concept mapping
| Concept | Example |
|---|---|
| Feature Group | `CustomerFeatures` (conceptual only) |
| Record Identifier | `customer_id` |
| Features | `age`, `hour`, `weekday`, `month`, `purchase_amount` |
| Record | One customer's complete feature row |
| Event Time | Source `timestamp` (conceptual mapping) |

## Online vs Offline Store
- **Online Store:** low-latency feature retrieval for real-time inference.
- **Offline Store:** historical feature data for training and analysis, persisted in Amazon S3.

## Cost note
This is a local data-preparation exercise. It does not create a SageMaker Feature Group, Online Store, Offline Store, S3 resources, or other chargeable AWS resources.
