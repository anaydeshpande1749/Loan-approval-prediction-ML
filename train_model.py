"""
Loan Approval Prediction - training script
==========================================

What this script does (in order):
  1. Loads data/loan_approval.csv and cleans column names / text / numbers
  2. Prints dataset diagnostics (shape, missing values, target balance,
     per-feature predictive power, duplicates, ...)
  3. Splits into TRAIN (80%) and TEST (20%), stratified. TEST is not touched
     again until the very last evaluation step.
  4. Compares 6 scikit-learn models with 5-fold cross-validation on TRAIN only
  5. Picks the best model by CROSS-VALIDATED ROC-AUC (reason explained below)
  6. Runs a "shuffled labels" sanity test to check if there is real signal
  7. Wraps the best pipeline in probability calibration (fit on TRAIN only)
  8. Tunes the decision threshold on out-of-fold TRAIN predictions
  9. Evaluates ONCE on the untouched TEST set
 10. Saves model, config, metrics and plots into models/

Every learned step (imputer, scaler, encoder, model, calibration) lives inside
ONE scikit-learn pipeline, so Flask uses exactly the same preprocessing.
"""

import json
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# Paths and constants
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "loan_approval.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TARGET = "loan_status"
ID_COLUMNS = ["id", "loan_id"]  # dropped if present (an ID has no meaning)

NUMERIC_FEATURES = [
    "no_of_dependents",
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score",
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value",
]
CATEGORICAL_FEATURES = ["education", "self_employed"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# --------------------------------------------------------------------------
# 1. Load + clean
# --------------------------------------------------------------------------
def load_and_clean():
    if not DATA_PATH.exists():
        raise SystemExit(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    section("1. RAW DATASET")
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))

    missing_cols = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing_cols:
        raise SystemExit(
            f"\nERROR: these required columns are missing from the CSV: {missing_cols}\n"
            f"Columns found in the CSV: {list(df.columns)}\n"
            "Edit NUMERIC_FEATURES / CATEGORICAL_FEATURES at the top of train_model.py "
            "so they match your CSV."
        )

    print("\nData types:\n", df.dtypes.to_string())
    print("\nMissing values per column:\n", df.isna().sum().to_string())

    found_ids = [c for c in ID_COLUMNS if c in df.columns]
    if found_ids:
        print(f"\nID column(s) found and excluded from features: {found_ids}")
    print("Duplicate rows (excluding ID):", df.drop(columns=found_ids).duplicated().sum())

    # --- clean categorical text: strip spaces, consistent capitalisation ---
    for c in CATEGORICAL_FEATURES:
        s = df[c].astype(str).str.strip().str.title()
        s = s.where(~s.isin(["", "Nan", "None", "<Na>"]), np.nan)
        df[c] = s
        print(f"\nUnique values in '{c}':", df[c].value_counts(dropna=False).to_dict())

    # --- clean numeric columns: non-numbers -> NaN, negatives -> NaN ---
    for c in NUMERIC_FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        n_neg = int((df[c] < 0).sum())
        if n_neg:
            print(f"'{c}': {n_neg} negative values set to missing")
            df.loc[df[c] < 0, c] = np.nan

    # --- target: make sure it is 0/1 ---
    t = df[TARGET]
    if pd.api.types.is_numeric_dtype(t):
        y = t
    else:
        y = (
            t.astype(str).str.strip().str.lower()
            .map({"approved": 1, "rejected": 0, "1": 1, "0": 0})
        )
    keep = y.notna()
    if (~keep).sum():
        print(f"Dropping {(~keep).sum()} rows with an unreadable target value")
    df, y = df[keep].copy(), y[keep].astype(int)
    if set(y.unique()) - {0, 1}:
        raise SystemExit(f"Target must be 0/1 but found: {sorted(y.unique())}")

    section("2. TARGET DISTRIBUTION  (1 = Approved, 0 = Rejected)")
    print(y.value_counts().sort_index().to_string())
    print("Approval rate: %.2f%%" % (100 * y.mean()))

    return df[FEATURES].copy(), y


# --------------------------------------------------------------------------
# 2. Diagnostics: does the data contain any predictive signal?
# --------------------------------------------------------------------------
def feature_diagnostics(X_train, y_train):
    section("3. FEATURE DIAGNOSTICS (training data only)")

    print("\nNumeric summary:")
    print(X_train[NUMERIC_FEATURES].describe().T[["min", "25%", "50%", "75%", "max"]].round(1).to_string())

    print("\nZero values per numeric column (possible placeholder/corruption):")
    print((X_train[NUMERIC_FEATURES] == 0).sum().to_string())

    print("\nOutliers (outside 1.5 x IQR):")
    for c in NUMERIC_FEATURES:
        q1, q3 = X_train[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        n_out = int(((X_train[c] < q1 - 1.5 * iqr) | (X_train[c] > q3 + 1.5 * iqr)).sum())
        print(f"  {c:28s} {n_out}")

    print("\nSingle-feature predictive power (ROC-AUC; 0.50 = pure chance):")
    aucs = {}
    for c in NUMERIC_FEATURES:
        m = X_train[c].notna()
        auc = roc_auc_score(y_train[m], X_train.loc[m, c])
        aucs[c] = max(auc, 1 - auc)
    for c, a in sorted(aucs.items(), key=lambda kv: -kv[1]):
        print(f"  {c:28s} {a:.3f}")

    print("\nApproval rate by category:")
    for c in CATEGORICAL_FEATURES:
        print(f"  {c}:", y_train.groupby(X_train[c]).mean().round(3).to_dict())

    print("\nApproval rate by CIBIL band (a strong signal would show a clear trend):")
    bands = pd.cut(X_train["cibil_score"], [0, 450, 550, 650, 750, 1000])
    print(y_train.groupby(bands, observed=True).agg(["mean", "count"]).round(3).to_string())

    return aucs


# --------------------------------------------------------------------------
# 3. Pipelines
# --------------------------------------------------------------------------
def make_preprocessor():
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", RobustScaler()),  # RobustScaler is less sensitive to outliers
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ])


def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=20, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=RANDOM_STATE),
        "Extra Trees": ExtraTreesClassifier(n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=RANDOM_STATE),
        "Hist Gradient Boosting": HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150, random_state=RANDOM_STATE),
    }


def make_pipeline(estimator):
    return Pipeline([("preprocess", make_preprocessor()), ("model", estimator)])


# --------------------------------------------------------------------------
# 4. Metrics helper
# --------------------------------------------------------------------------
def metrics_at(y_true, proba, threshold):
    pred = (proba >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
    }


def to_py(obj):
    """Make numpy types JSON-serialisable."""
    if isinstance(obj, dict):
        return {k: to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_py(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    X, y = load_and_clean()

    # ---- stratified train/test split. TEST IS LOCKED AWAY UNTIL STEP 9 ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    print(f"\nTrain rows: {len(X_train)} | Test rows (locked): {len(X_test)}")

    univariate_aucs = feature_diagnostics(X_train, y_train)

    # ---- cross-validated model comparison (TRAIN only) ----
    section("4. MODEL COMPARISON  (5-fold CV on training data)")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    cv_results = {}
    for name, est in get_models().items():
        scores = cross_validate(make_pipeline(est), X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
        cv_results[name] = {m: float(scores[f"test_{m}"].mean()) for m in scoring}
        cv_results[name]["roc_auc_std"] = float(scores["test_roc_auc"].std())
        r = cv_results[name]
        print(f"{name:24s} acc={r['accuracy']:.3f} prec={r['precision']:.3f} "
              f"rec={r['recall']:.3f} f1={r['f1']:.3f} auc={r['roc_auc']:.3f} (+/-{r['roc_auc_std']:.3f})")

    # ---- model selection ----
    # WHY ROC-AUC? F1 and accuracy depend on a threshold. When approvals are the
    # majority class and the signal is weak, "approve everybody" gets a high F1
    # without learning anything. ROC-AUC measures how well the model RANKS
    # applicants, independent of the threshold, so it is a fairer way to choose.
    best_name = max(cv_results, key=lambda n: (cv_results[n]["roc_auc"], cv_results[n]["f1"]))
    best_auc = cv_results[best_name]["roc_auc"]
    print(f"\n>>> Selected model: {best_name}  (highest CV ROC-AUC = {best_auc:.3f})")

    # ---- sanity test: is there ANY real signal? ----
    section("5. SIGNAL CHECK (real labels vs shuffled labels)")
    rng = np.random.RandomState(RANDOM_STATE)
    shuffled = pd.Series(rng.permutation(y_train.values), index=y_train.index)
    shuf_auc = cross_val_score(
        make_pipeline(clone(get_models()[best_name])), X_train, shuffled,
        cv=cv, scoring="roc_auc"
    ).mean()
    print(f"CV ROC-AUC with REAL labels    : {best_auc:.3f}")
    print(f"CV ROC-AUC with SHUFFLED labels: {shuf_auc:.3f}  (this is what pure noise looks like)")
    best_single = max(univariate_aucs.values())
    if best_auc < 0.55:
        verdict = "NO_USEFUL_SIGNAL"
        print("\nVERDICT: The features carry almost no information about loan_status.")
        print("         No algorithm can fix that. The labels in this CSV look random")
        print("         relative to the features (check the CIBIL-band table above).")
        print("         Consider a cleaner copy of the dataset.")
    elif best_auc < 0.70:
        verdict = "WEAK_SIGNAL"
        print("\nVERDICT: Weak signal. Predictions are better than chance but unreliable.")
    else:
        verdict = "USEFUL_SIGNAL"
        print("\nVERDICT: The features carry real information about loan_status.")
    print(f"(Best single-feature AUC was {best_single:.3f})")

    # ---- final model = best pipeline + probability calibration ----
    # CalibratedClassifierCV re-fits the pipeline inside its own CV folds, so
    # calibration only ever sees training data.
    def make_final():
        return CalibratedClassifierCV(make_pipeline(clone(get_models()[best_name])), method="sigmoid", cv=5)

    # ---- threshold tuning on OUT-OF-FOLD training predictions ----
    section("6. DECISION THRESHOLD TUNING (training data only)")
    cv_thr = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
    oof = cross_val_predict(make_final(), X_train, y_train, cv=cv_thr, method="predict_proba")[:, 1]
    print("Out-of-fold probability: min=%.3f  mean=%.3f  max=%.3f" % (oof.min(), oof.mean(), oof.max()))
    print("Percentiles 5/25/50/75/95:", np.round(np.percentile(oof, [5, 25, 50, 75, 95]), 3))

    # WHY BALANCED ACCURACY? Same reason as above: maximising F1 pushes the
    # threshold toward 0 ("approve all"). Balanced accuracy treats approved and
    # rejected applicants equally, so the threshold must actually separate them.
    grid = np.round(np.arange(0.20, 0.801, 0.01), 2)
    bal = np.array([balanced_accuracy_score(y_train, (oof >= t).astype(int)) for t in grid])
    best_idx = int(np.argmax(bal))
    bal_at_half = balanced_accuracy_score(y_train, (oof >= 0.5).astype(int))
    if bal[best_idx] - bal_at_half < 0.005:
        threshold = 0.50
        reason = "No threshold beat 0.50 by a meaningful margin (>0.5 pts balanced accuracy), so the default was kept."
    else:
        threshold = float(grid[best_idx])
        reason = (f"Maximises balanced accuracy on out-of-fold training predictions "
                  f"({bal[best_idx]:.3f} vs {bal_at_half:.3f} at 0.50).")
    print(f"Chosen threshold: {threshold:.2f}\nReason: {reason}")
    thr_tuning_metrics = metrics_at(y_train, oof, threshold)

    # ---- fit final model on all TRAIN, evaluate ONCE on TEST ----
    section("7. FINAL EVALUATION ON UNTOUCHED TEST SET (done once)")
    final_model = make_final()
    final_model.fit(X_train, y_train)
    p_test = final_model.predict_proba(X_test)[:, 1]

    test_at_thr = metrics_at(y_test, p_test, threshold)
    test_at_half = metrics_at(y_test, p_test, 0.50)
    brier = brier_score_loss(y_test, p_test)
    cm = confusion_matrix(y_test, (p_test >= threshold).astype(int))
    baseline_acc = max(y_test.mean(), 1 - y_test.mean())

    print(f"Model: {best_name} (calibrated), threshold = {threshold:.2f}")
    for k, v in test_at_thr.items():
        print(f"  {k:18s}: {v:.4f}")
    print(f"  brier score       : {brier:.4f}")
    print(f"  (always predicting the majority class would give accuracy {baseline_acc:.4f})")
    print("\nConfusion matrix [rows=actual, cols=predicted]  (Rejected, Approved):")
    print(cm)
    print("\nFor reference, at threshold 0.50:", {k: round(v, 4) for k, v in test_at_half.items()})

    # ---- plots ----
    names = list(cv_results)
    x = np.arange(len(names))
    w = 0.2
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, m in enumerate(["accuracy", "f1", "roc_auc"]):
        ax.bar(x + (i - 1) * w, [cv_results[n][m] for n in names], w, label=m.upper().replace("_", "-"))
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=1, label="chance (0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title("Model comparison (5-fold cross-validation on training data)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(MODEL_DIR / "model_comparison.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", fontsize=14,
                color="white" if v > cm.max() / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Rejected", "Approved"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Rejected", "Approved"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix (test set, threshold={threshold:.2f})")
    fig.tight_layout()
    fig.savefig(MODEL_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    fpr, tpr, _ = roc_curve(y_test, p_test)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(fpr, tpr, label=f"AUC = {test_at_thr['roc_auc']:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey", label="chance")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve (test set)"); ax.legend()
    fig.tight_layout()
    fig.savefig(MODEL_DIR / "roc_curve.png", dpi=150)
    plt.close(fig)

    prec, rec, _ = precision_recall_curve(y_test, p_test)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(rec, prec, label=f"AP = {average_precision_score(y_test, p_test):.3f}")
    ax.axhline(y_test.mean(), linestyle="--", color="grey", label="baseline (approval rate)")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curve (test set)"); ax.legend()
    fig.tight_layout()
    fig.savefig(MODEL_DIR / "precision_recall_curve.png", dpi=150)
    plt.close(fig)

    # permutation importance: diagnostic only, not used for any decision
    imp = permutation_importance(final_model, X_test, y_test, scoring="roc_auc",
                                 n_repeats=10, random_state=RANDOM_STATE)
    order = np.argsort(imp.importances_mean)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh([FEATURES[i] for i in order], imp.importances_mean[order], xerr=imp.importances_std[order])
    ax.set_xlabel("Drop in ROC-AUC when the feature is shuffled")
    ax.set_title("Permutation feature importance (test set, diagnostic)")
    fig.tight_layout()
    fig.savefig(MODEL_DIR / "feature_importance.png", dpi=150)
    plt.close(fig)
    importance = {FEATURES[i]: float(imp.importances_mean[i]) for i in order[::-1]}
    print("\nFeature importance (drop in AUC):", {k: round(v, 4) for k, v in importance.items()})

    # ---- supported input ranges (from training data) for the web form ----
    ranges = {c: {"min": float(X_train[c].min()), "max": float(X_train[c].max())} for c in NUMERIC_FEATURES}
    categories = {c: sorted(X_train[c].dropna().unique().tolist()) for c in CATEGORICAL_FEATURES}

    # ---- save everything ----
    joblib.dump(final_model, MODEL_DIR / "best_model.pkl")

    warning_text = None
    if test_at_thr["roc_auc"] < 0.60:
        warning_text = ("This model performs close to random guessing on held-out data, "
                        "so its output should not be treated as a meaningful prediction.")

    config = {
        "best_model": best_name,
        "calibration": "sigmoid (CalibratedClassifierCV, cv=5)",
        "selection_metric": "5-fold cross-validated ROC-AUC",
        "decision_threshold": threshold,
        "threshold_method": "maximise balanced accuracy on out-of-fold training predictions",
        "threshold_reason": reason,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "categories": categories,
        "ranges": ranges,
        "signal_verdict": verdict,
        "test_roc_auc": float(test_at_thr["roc_auc"]),
        "test_accuracy": float(test_at_thr["accuracy"]),
        "model_quality_warning": warning_text,
    }
    results = {
        "cross_validation_on_train": cv_results,
        "selected_model": best_name,
        "shuffled_label_cv_auc": float(shuf_auc),
        "signal_verdict": verdict,
        "threshold": threshold,
        "threshold_reason": reason,
        "threshold_tuning_metrics_oof_train": thr_tuning_metrics,
        "final_test_metrics_at_threshold": test_at_thr,
        "final_test_metrics_at_0.50": test_at_half,
        "test_brier_score": brier,
        "test_confusion_matrix": {"rows_actual": ["Rejected", "Approved"],
                                  "cols_predicted": ["Rejected", "Approved"],
                                  "matrix": cm.tolist()},
        "majority_class_baseline_accuracy": float(baseline_acc),
        "feature_importance_auc_drop": importance,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    (MODEL_DIR / "model_config.json").write_text(json.dumps(to_py(config), indent=2))
    (MODEL_DIR / "model_results.json").write_text(json.dumps(to_py(results), indent=2))

    section("DONE - SUMMARY")
    print(f"Selected model      : {best_name} (+ sigmoid calibration)")
    print(f"Decision threshold  : {threshold:.2f}")
    print(f"Test accuracy       : {test_at_thr['accuracy']:.4f}  (majority baseline {baseline_acc:.4f})")
    print(f"Test F1             : {test_at_thr['f1']:.4f}")
    print(f"Test ROC-AUC        : {test_at_thr['roc_auc']:.4f}")
    print(f"Signal verdict      : {verdict}")
    if warning_text:
        print("WARNING:", warning_text)
    print(f"\nSaved files in {MODEL_DIR}:")
    for p in sorted(MODEL_DIR.iterdir()):
        print("  -", p.name)


if __name__ == "__main__":
    main()