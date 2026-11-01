from pathlib import Path
import joblib
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score

ROOT = Path(__file__).parent
DATA = ROOT / "data"
FEATURES = ["income", "transaction_amount"]

model_path = ROOT / "model.joblib"
if not model_path.exists():
    raise FileNotFoundError("model.joblib not found. Run: python train.py")

model = joblib.load(model_path)

def compare_feature_means(reference, current, label):
    print(f"\n{label}")
    for column in FEATURES:
        ref_mean = reference[column].mean()
        cur_mean = current[column].mean()
        change_pct = ((cur_mean - ref_mean) / ref_mean) * 100 if ref_mean else 0
        print(
            f"{column}: baseline mean={ref_mean:,.2f}, "
            f"current mean={cur_mean:,.2f}, change={change_pct:+.1f}%"
        )

def prediction_rate(data):
    predictions = model.predict(data[FEATURES])
    return predictions.mean(), predictions

training = pd.read_csv(DATA / "training_data.csv")
baseline_eval = pd.read_csv(DATA / "baseline_evaluation_data.csv")
data_drift = pd.read_csv(DATA / "data_drift_data.csv")
concept_drift = pd.read_csv(DATA / "concept_drift_data.csv")

print("=" * 60)
print("DAY 12 - LOCAL ML MONITORING REPORT")
print("=" * 60)

# 1. Data drift: compare feature distributions/means.
compare_feature_means(training, data_drift, "1) DATA DRIFT CHECK")
print("Interpretation: noticeable input changes may indicate data drift.")

# 2. Prediction drift: compare positive prediction rates.
baseline_rate, baseline_predictions = prediction_rate(baseline_eval)
drift_rate, drift_predictions = prediction_rate(data_drift)
print("\n2) PREDICTION DRIFT CHECK")
print(f"Baseline positive prediction rate: {baseline_rate:.1%}")
print(f"Current positive prediction rate:  {drift_rate:.1%}")
print(f"Difference: {(drift_rate - baseline_rate) * 100:+.1f} percentage points")

# 3. Concept drift / performance degradation:
# Inputs are sampled from a similar distribution, but target relationship changes.
_, baseline_eval_predictions = prediction_rate(baseline_eval)
_, concept_predictions = prediction_rate(concept_drift)

baseline_f1 = f1_score(baseline_eval["fraud"], baseline_eval_predictions, zero_division=0)
concept_f1 = f1_score(concept_drift["fraud"], concept_predictions, zero_division=0)
concept_accuracy = accuracy_score(concept_drift["fraud"], concept_predictions)

print("\n3) MODEL PERFORMANCE CHECK")
print(f"Baseline evaluation F1: {baseline_f1:.3f}")
print(f"Current concept-drift F1: {concept_f1:.3f}")
print(f"Current concept-drift accuracy: {concept_accuracy:.3f}")
print("Ground-truth fraud labels are used here to measure actual model quality.")

print("\n4) MONITORING DECISION")
if concept_f1 < baseline_f1:
    print("Performance decreased on concept-drift data: investigate and consider retraining.")
else:
    print("No F1 decrease detected in this synthetic run.")
print("This is a learning simulation, not a production alarm or automatic retraining system.")
