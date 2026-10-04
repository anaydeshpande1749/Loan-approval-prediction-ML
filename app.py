"""
Loan Approval Prediction - Flask app

Flow:  form input -> validation -> DataFrame -> saved pipeline (preprocess + model)
       -> probability -> decision threshold -> APPROVED / REJECTED
"""

import json
import math
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
CONFIG_PATH = BASE_DIR / "models" / "model_config.json"

app = Flask(__name__)

# --------------------------------------------------------------------------
# Load model + config (the app still starts if they are missing and shows a message)
# --------------------------------------------------------------------------
model, config, LOAD_ERROR = None, None, None
try:
    model = joblib.load(MODEL_PATH)
    config = json.loads(CONFIG_PATH.read_text())
except Exception as exc:  # noqa: BLE001
    LOAD_ERROR = f"Model files could not be loaded ({exc}). Run `python train_model.py` first."

NUMERIC = config["numeric_features"] if config else []
CATEGORICAL = config["categorical_features"] if config else []
FEATURES = NUMERIC + CATEGORICAL
THRESHOLD = config["decision_threshold"] if config else 0.5

# Extra common-sense limits, because the training data can contain junk values
# (e.g. an income of 0). The effective minimum is the larger of the two.
HARD_MIN = {"income_annum": 50_000, "loan_amount": 10_000, "cibil_score": 300, "loan_term": 1}
HARD_MAX = {"cibil_score": 900}
MAX_LOAN_TO_INCOME = 50  # loan more than 50x annual income = clearly invalid

LABELS = {
    "no_of_dependents": "No. of dependents",
    "income_annum": "Annual income (₹)",
    "loan_amount": "Loan amount (₹)",
    "loan_term": "Loan term (years)",
    "cibil_score": "CIBIL score",
    "residential_assets_value": "Residential assets value (₹)",
    "commercial_assets_value": "Commercial assets value (₹)",
    "luxury_assets_value": "Luxury assets value (₹)",
    "bank_asset_value": "Bank assets value (₹)",
}
PLACEHOLDERS = {
    "no_of_dependents": "e.g. 2",
    "income_annum": "e.g. 5000000",
    "loan_amount": "e.g. 15000000",
    "loan_term": "e.g. 10",
    "cibil_score": "e.g. 750",
    "residential_assets_value": "e.g. 8000000",
    "commercial_assets_value": "e.g. 4000000",
    "luxury_assets_value": "e.g. 10000000",
    "bank_asset_value": "e.g. 5000000",
}
INTEGER_FIELDS = {"no_of_dependents", "loan_term", "cibil_score"}


def effective_limits():
    """Min/max per numeric field = training range, tightened by common-sense limits."""
    limits = {}
    for col in NUMERIC:
        lo = config["ranges"][col]["min"]
        hi = config["ranges"][col]["max"]
        lo = max(lo, HARD_MIN.get(col, 0))
        if col.endswith("_value"):
            lo = 0  # having zero of an asset type is perfectly legitimate
        hi = min(hi, HARD_MAX.get(col, hi))
        limits[col] = {"min": math.ceil(lo), "max": math.floor(hi)}
    return limits


LIMITS = effective_limits() if config else {}


def validate(form):
    """Returns (clean_values, list_of_error_messages)."""
    values, errors = {}, []

    for col in NUMERIC:
        raw = (form.get(col) or "").strip()
        label = LABELS.get(col, col)
        if raw == "":
            errors.append(f"{label}: this field is required.")
            continue
        try:
            val = float(raw)
        except ValueError:
            errors.append(f"{label}: '{raw}' is not a valid number.")
            continue
        if math.isnan(val) or math.isinf(val):
            errors.append(f"{label}: invalid number.")
            continue
        if val < 0:
            errors.append(f"{label}: cannot be negative.")
            continue
        if col in INTEGER_FIELDS and not float(val).is_integer():
            errors.append(f"{label}: must be a whole number.")
            continue
        lo, hi = LIMITS[col]["min"], LIMITS[col]["max"]
        if not (lo <= val <= hi):
            errors.append(
                f"{label}: {val:,.0f} is outside the range supported by this model "
                f"({lo:,} to {hi:,})."
            )
            continue
        values[col] = val

    for col in CATEGORICAL:
        raw = (form.get(col) or "").strip()
        allowed = config["categories"][col]
        match = next((a for a in allowed if a.lower() == raw.lower()), None)
        if match is None:
            errors.append(f"{col.replace('_', ' ').title()}: please choose one of {allowed}.")
        else:
            values[col] = match

    # relationship checks (only if the individual fields were valid)
    if "income_annum" in values and "loan_amount" in values:
        if values["loan_amount"] > MAX_LOAN_TO_INCOME * values["income_annum"]:
            errors.append(
                f"The loan amount is more than {MAX_LOAN_TO_INCOME}x the annual income, "
                "which is not a realistic application."
            )

    return values, errors


@app.route("/", methods=["GET", "POST"])
def index():
    context = {
        "numeric_fields": [
            {"name": c, "label": LABELS.get(c, c), "placeholder": PLACEHOLDERS.get(c, ""),
             "min": LIMITS.get(c, {}).get("min"), "max": LIMITS.get(c, {}).get("max")}
            for c in NUMERIC
        ],
        "categories": config["categories"] if config else {},
        "form": {},
        "errors": [],
        "result": None,
        "load_error": LOAD_ERROR,
        "quality_warning": config.get("model_quality_warning") if config else None,
    }

    if request.method == "POST" and not LOAD_ERROR:
        context["form"] = request.form.to_dict()
        values, errors = validate(request.form)
        context["errors"] = errors

        if not errors:
            try:
                # exactly the training feature names -> pipeline does ALL preprocessing
                row = pd.DataFrame([{c: values[c] for c in FEATURES}], columns=FEATURES)
                probability = float(model.predict_proba(row)[0][1])
                approved = probability >= THRESHOLD
                context["result"] = {
                    "approved": approved,
                    "probability": round(probability * 100, 1),
                    "threshold": round(THRESHOLD * 100, 1),
                    "model": config["best_model"],
                }
            except Exception as exc:  # noqa: BLE001
                context["errors"] = [f"Prediction failed: {exc}"]

    return render_template("index.html", **context)


if __name__ == "__main__":
    app.run(debug=True)