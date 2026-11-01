from pathlib import Path
import numpy as np
import pandas as pd

RANDOM_SEED = 42
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

rng = np.random.default_rng(RANDOM_SEED)

def make_labels(income, transaction_amount, changed_relationship=False):
    # Baseline relationship: larger transactions and lower income increase fraud risk.
    if changed_relationship:
        # Simulated concept drift: the relationship changes in production.
        score = (income > 85000).astype(int) + (transaction_amount < 1200).astype(int)
    else:
        score = (transaction_amount > 3000).astype(int) + (income < 50000).astype(int)
    return (score >= 1).astype(int)

# Training/baseline data.
n_train = 1200
train_income = rng.normal(70000, 15000, n_train).clip(20000, 150000)
train_amount = rng.normal(2200, 900, n_train).clip(100, 7000)
train_labels = make_labels(train_income, train_amount)

training = pd.DataFrame({
    "income": train_income.round(2),
    "transaction_amount": train_amount.round(2),
    "fraud": train_labels
})
training.to_csv(DATA_DIR / "training_data.csv", index=False)

# Baseline evaluation data: same distribution and same relationship.
n_eval = 400
eval_income = rng.normal(70000, 15000, n_eval).clip(20000, 150000)
eval_amount = rng.normal(2200, 900, n_eval).clip(100, 7000)
baseline_eval = pd.DataFrame({
    "income": eval_income.round(2),
    "transaction_amount": eval_amount.round(2),
    "fraud": make_labels(eval_income, eval_amount)
})
baseline_eval.to_csv(DATA_DIR / "baseline_evaluation_data.csv", index=False)

# Data drift data: input distributions shift, while baseline label logic stays the same.
n_prod = 400
drift_income = rng.normal(105000, 18000, n_prod).clip(25000, 190000)
drift_amount = rng.normal(3600, 1100, n_prod).clip(100, 9000)
data_drift = pd.DataFrame({
    "income": drift_income.round(2),
    "transaction_amount": drift_amount.round(2),
    "fraud": make_labels(drift_income, drift_amount)
})
data_drift.to_csv(DATA_DIR / "data_drift_data.csv", index=False)

# Concept drift data: input distribution stays similar to baseline,
# but the relationship between features and the target changes.
concept_income = rng.normal(70000, 15000, n_prod).clip(20000, 150000)
concept_amount = rng.normal(2200, 900, n_prod).clip(100, 7000)
concept_drift = pd.DataFrame({
    "income": concept_income.round(2),
    "transaction_amount": concept_amount.round(2),
    "fraud": make_labels(concept_income, concept_amount, changed_relationship=True)
})
concept_drift.to_csv(DATA_DIR / "concept_drift_data.csv", index=False)

print("Sample monitoring datasets created in the data/ folder.")
print("Files: training_data.csv, baseline_evaluation_data.csv,")
print("       data_drift_data.csv, concept_drift_data.csv")
