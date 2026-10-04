"""
Tries the saved model on a few hand-made applicants.
The script does NOT know the "right" answer - the model + threshold decide.
Run:  python test_predictions.py
"""

import json
from pathlib import Path

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
model = joblib.load(BASE_DIR / "models" / "best_model.pkl")
config = json.loads((BASE_DIR / "models" / "model_config.json").read_text())

FEATURES = config["numeric_features"] + config["categorical_features"]
THRESHOLD = config["decision_threshold"]

applicants = {
    "Applicant 1 - Strong (high CIBIL, high income, modest loan, many assets)": {
        "no_of_dependents": 1, "education": "Graduate", "self_employed": "No",
        "income_annum": 9_000_000, "loan_amount": 10_000_000, "loan_term": 10,
        "cibil_score": 820,
        "residential_assets_value": 15_000_000, "commercial_assets_value": 10_000_000,
        "luxury_assets_value": 20_000_000, "bank_asset_value": 8_000_000,
    },
    "Applicant 2 - Weak (low CIBIL, low income, huge loan, almost no assets)": {
        "no_of_dependents": 5, "education": "Not Graduate", "self_employed": "No",
        "income_annum": 300_000, "loan_amount": 25_000_000, "loan_term": 18,
        "cibil_score": 330,
        "residential_assets_value": 0, "commercial_assets_value": 0,
        "luxury_assets_value": 100_000, "bank_asset_value": 20_000,
    },
    "Applicant 3 - Moderate (middle-range profile)": {
        "no_of_dependents": 2, "education": "Graduate", "self_employed": "Yes",
        "income_annum": 5_000_000, "loan_amount": 15_000_000, "loan_term": 12,
        "cibil_score": 600,
        "residential_assets_value": 6_000_000, "commercial_assets_value": 3_000_000,
        "luxury_assets_value": 8_000_000, "bank_asset_value": 3_000_000,
    },
}

print(f"Model: {config['best_model']} | decision threshold: {THRESHOLD:.2f}")
print(f"Signal verdict from training: {config['signal_verdict']} "
      f"(test ROC-AUC {config['test_roc_auc']:.3f})\n")

for name, data in applicants.items():
    row = pd.DataFrame([data], columns=FEATURES)
    prob = float(model.predict_proba(row)[0][1])
    label = "APPROVED" if prob >= THRESHOLD else "REJECTED"
    print(name)
    print(f"  Estimated probability: {prob * 100:.1f}%")
    print(f"  Prediction: {label}\n")

if config.get("model_quality_warning"):
    print("NOTE:", config["model_quality_warning"])