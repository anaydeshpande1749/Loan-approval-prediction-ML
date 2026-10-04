import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

DATA_PATH = "data/loan_approval.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        "Dataset not found. Put your CSV at data/loan_approval.csv"
    )

df = pd.read_csv(DATA_PATH)
df.columns = [c.strip() for c in df.columns]

# Make column names consistent with the commonly used Loan Approval dataset.
rename_map = {
    "Loan_ID": "loan_id",
    "loan_id": "loan_id",
    "no_of_dependents": "no_of_dependents",
    "No_of_dependents": "no_of_dependents",
    "education": "education",
    "Education": "education",
    "self_employed": "self_employed",
    "Self_employed": "self_employed",
    "income_annum": "income_annum",
    "Income_annum": "income_annum",
    "loan_amount": "loan_amount",
    "Loan_amount": "loan_amount",
    "loan_term": "loan_term",
    "Loan_term": "loan_term",
    "cibil_score": "cibil_score",
    "Cibil_score": "cibil_score",
    "residential_assets_value": "residential_assets_value",
    "Residential_assets_value": "residential_assets_value",
    "commercial_assets_value": "commercial_assets_value",
    "Commercial_assets_value": "commercial_assets_value",
    "luxury_assets_value": "luxury_assets_value",
    "Luxury_assets_value": "luxury_assets_value",
    "bank_asset_value": "bank_asset_value",
    "Bank_asset_value": "bank_asset_value",
    "loan_status": "loan_status",
    "Loan_status": "loan_status",
}

df = df.rename(columns={c: rename_map.get(c, c) for c in df.columns})

required = [
    "no_of_dependents", "education", "self_employed",
    "income_annum", "loan_amount", "loan_term", "cibil_score",
    "residential_assets_value", "commercial_assets_value",
    "luxury_assets_value", "bank_asset_value", "loan_status"
]

missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(
        "Missing required columns: " + ", ".join(missing) +
        "\nCheck the dataset column names."
    )

# Clean target.
def encode_target(value):
    s = str(value).strip().lower()
    if s in ["approved", "approve", "yes", "1", "true"]:
        return 1
    if s in ["rejected", "reject", "no", "0", "false"]:
        return 0
    try:
        return int(float(s))
    except ValueError:
        return np.nan

df["loan_status"] = df["loan_status"].apply(encode_target)
df = df.dropna(subset=["loan_status"])

# Convert numerical fields to numeric.
numeric_cols = [
    "no_of_dependents", "income_annum", "loan_amount", "loan_term",
    "cibil_score", "residential_assets_value",
    "commercial_assets_value", "luxury_assets_value", "bank_asset_value"
]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Basic EDA output.
print("\nDataset shape:", df.shape)
print("\nMissing values:")
print(df[required].isnull().sum())
print("\nTarget distribution:")
print(df["loan_status"].value_counts())

X = df[[
    "no_of_dependents", "education", "self_employed",
    "income_annum", "loan_amount", "loan_term", "cibil_score",
    "residential_assets_value", "commercial_assets_value",
    "luxury_assets_value", "bank_asset_value"
]]
y = df["loan_status"].astype(int)

categorical_cols = ["education", "self_employed"]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, [c for c in X.columns if c not in categorical_cols]),
    ("cat", categorical_pipeline, categorical_cols)
])

models = {
    "Logistic Regression": LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=6,
        class_weight="balanced",
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
}

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

results = []
trained_pipelines = {}

for name, model in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred, zero_division=0)
    recall = recall_score(y_test, pred, zero_division=0)
    f1 = f1_score(y_test, pred, zero_division=0)

    results.append({
        "model": name,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4)
    })

    trained_pipelines[name] = pipeline

    print(f"\n{name}")
    print(classification_report(y_test, pred, zero_division=0))

# Select the model with the highest F1 score.
results_df = pd.DataFrame(results).sort_values("f1_score", ascending=False)
best_name = results_df.iloc[0]["model"]
best_pipeline = trained_pipelines[best_name]

joblib.dump(best_pipeline, f"{MODEL_DIR}/best_model.pkl")

with open(f"{MODEL_DIR}/model_results.json", "w") as f:
    json.dump({
        "best_model": best_name,
        "results": results
    }, f, indent=2)

# Model comparison graph.
plt.figure(figsize=(8, 5))
sns.barplot(data=results_df, x="model", y="accuracy")
plt.ylim(0, 1)
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.xlabel("Model")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(f"{MODEL_DIR}/model_comparison.png", dpi=150)
plt.close()

# Confusion matrix for best model.
best_pred = best_pipeline.predict(X_test)
cm = confusion_matrix(y_test, best_pred)

plt.figure(figsize=(5, 4))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=["Rejected", "Approved"],
    yticklabels=["Rejected", "Approved"]
)
plt.title(f"Confusion Matrix - {best_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(f"{MODEL_DIR}/confusion_matrix.png", dpi=150)
plt.close()

print("\n==============================")
print("MODEL COMPARISON")
print("==============================")
print(results_df.to_string(index=False))
print(f"\nBest model: {best_name}")
print("\nSaved:")
print("- models/best_model.pkl")
print("- models/model_results.json")
print("- models/model_comparison.png")
print("- models/confusion_matrix.png")
