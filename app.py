from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

app = Flask(__name__)

MODEL_PATH = "models/best_model.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "Model not found. First run: python train_model.py"
    )

model = joblib.load(MODEL_PATH)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    probability = None
    error = None

    if request.method == "POST":
        try:
            data = {
                "no_of_dependents": int(request.form["no_of_dependents"]),
                "education": request.form["education"],
                "self_employed": request.form["self_employed"],
                "income_annum": float(request.form["income_annum"]),
                "loan_amount": float(request.form["loan_amount"]),
                "loan_term": float(request.form["loan_term"]),
                "cibil_score": float(request.form["cibil_score"]),
                "residential_assets_value": float(request.form["residential_assets_value"]),
                "commercial_assets_value": float(request.form["commercial_assets_value"]),
                "luxury_assets_value": float(request.form["luxury_assets_value"]),
                "bank_asset_value": float(request.form["bank_asset_value"])
            }

            input_df = pd.DataFrame([data])
            pred = int(model.predict(input_df)[0])

            if hasattr(model, "predict_proba"):
                probability = round(float(model.predict_proba(input_df)[0][1]) * 100, 2)

            prediction = "APPROVED" if pred == 1 else "REJECTED"

        except Exception as e:
            error = str(e)

    return render_template(
        "index.html",
        prediction=prediction,
        probability=probability,
        error=error
    )

if __name__ == "__main__":
    app.run(debug=True)
