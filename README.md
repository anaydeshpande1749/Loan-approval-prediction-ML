# Loan Approval Prediction Using Machine Learning

A 7th-semester Computer Engineering ML mini-project. It trains and compares several scikit-learn classifiers on a loan-application dataset, selects the best one with cross-validation, tunes a decision threshold, calibrates probabilities, and serves the result through a Flask web app with input validation.

> **Honesty note:** every number in this README comes from running `train_model.py` on the dataset in `data/loan_approval.csv`. Nothing was hard-coded or tuned on the test set. See [Section 12](#12-limitations-read-this-before-your-viva) for the limitations.

---

## Table of contents

1. [Problem statement](#1-problem-statement)
2. [Project journey: what went wrong first and how it was fixed](#2-project-journey)
3. [Dataset](#3-dataset)
4. [Project structure](#4-project-structure)
5. [Technology stack](#5-technology-stack)
6. [The ML pipeline, step by step](#6-the-ml-pipeline-step-by-step)
7. [Results](#7-results)
8. [The web application](#8-the-web-application)
9. [How to run](#9-how-to-run)
10. [Test inputs](#10-test-inputs)
11. [File-by-file explanation](#11-file-by-file-explanation)
12. [Limitations](#12-limitations-read-this-before-your-viva)
13. [Concepts glossary](#13-concepts-glossary)
14. [Future improvements](#14-future-improvements)

---

## 1. Problem statement

Given an applicant's details (dependents, education, employment type, income, loan amount, loan term, CIBIL credit score and four asset values), predict whether the loan will be **Approved (1)** or **Rejected (0)**, and give an estimated approval probability.

This is a **binary classification** problem on **tabular data**.

---

## 2. Project journey

Understanding this story helps in the viva, because it shows the project was debugged properly.

**Stage 1: first version.** A Flask app with Logistic Regression, Random Forest and Decision Tree. Test accuracy was only about 52%, and a clearly bad applicant (60 dependents, income ₹100, CIBIL 350) was still "Approved" at about 60%.

**Stage 2: diagnosis.** The training script was rewritten to print diagnostics before modelling. They showed:

- Every feature had a single-feature ROC-AUC of about 0.50 (pure chance).
- The approval rate was about 54% in *every* CIBIL band. In real credit data, a low CIBIL score should mean far fewer approvals.
- A **shuffled-label test** gave the same score as the real labels (0.516 vs 0.503). When real labels score no better than random ones, the features contain no information about the target.
- Engineered features (loan/income ratio, total assets, missing-value counts) did not help either.

**Conclusion: the original 32,000-row CSV had labels unrelated to its features.** No algorithm or threshold tweak can fix that. The "predicted 61.5%" was just the model outputting roughly the base approval rate for everyone. Lowering the threshold to reject bad demo inputs would have been cheating, so it was not done.

**Stage 3: fix.** The dataset was replaced with the standard public *Loan Approval Prediction Dataset* (4,278 rows). The same code then gave about 97.6% test accuracy and the shuffled-label control stayed at about 0.48, confirming the signal is real.

---

## 3. Dataset

| Property | Value |
|---|---|
| Source | Public "Loan Approval Prediction Dataset" (originally on Kaggle) |
| Rows | 4,278 |
| Train / Test | 3,422 / 856 (80/20, stratified) |
| Target | `loan_status` (1 = Approved, 0 = Rejected) |
| Class balance | 62.16% approved, 37.84% rejected |
| Missing values | A handful (for example 4 in `self_employed`); handled by imputation inside the pipeline |
| Identifier | `id` (excluded from features) |

### Columns

| Column | Type | Meaning | Range in data |
|---|---|---|---|
| `no_of_dependents` | numeric | People financially dependent on the applicant | 0 – 5 |
| `education` | categorical | Graduate / Not Graduate | |
| `self_employed` | categorical | Yes / No | |
| `income_annum` | numeric | Annual income (₹) | 200,000 – 9,900,000 |
| `loan_amount` | numeric | Requested loan (₹) | 300,000 – 38,800,000 (train) |
| `loan_term` | numeric | Loan duration (years) | 2 – 20 |
| `cibil_score` | numeric | Credit score | 300 – 900 |
| `residential_assets_value` | numeric | Residential property value (₹) | 0 – 29,100,000 |
| `commercial_assets_value` | numeric | Commercial property value (₹) | 0 – 19,400,000 |
| `luxury_assets_value` | numeric | Luxury assets value (₹) | 0 – 39,200,000 |
| `bank_asset_value` | numeric | Bank balance / deposits (₹) | 0 – 14,700,000 |

### Key findings from the diagnostics

- **CIBIL score dominates.** Its single-feature ROC-AUC is 0.956; every other feature is at most 0.566.
- **Approval rate by CIBIL band** (training data):

| CIBIL band | Approval rate |
|---|---|
| 0–450 | 11.4% |
| 450–550 | 10.8% |
| 550–650 | 99.6% |
| 650–750 | 99.3% |
| 750+ | 99.3% |

  The data behaves almost like a rule: roughly "CIBIL above about 550 means approved". This is why accuracy is so high.
- 28 negative `residential_assets_value` entries are data errors and are converted to missing values, then imputed.
- Zero values are legitimate for assets and dependents.

---

## 4. Project structure

```
loan_approval_ml_project/
├── data/
│   └── loan_approval.csv          # dataset
├── models/                        # created by train_model.py
│   ├── best_model.pkl             # full pipeline: preprocessing + model + calibration
│   ├── model_config.json          # threshold, features, valid ranges, categories
│   ├── model_results.json         # all metrics
│   ├── model_comparison.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── precision_recall_curve.png
│   └── feature_importance.png
├── templates/
│   └── index.html                 # web form + result display
├── static/
│   └── style.css
├── app.py                         # Flask app
├── train_model.py                 # training / evaluation script
├── test_predictions.py            # tries 3 hand-made applicants
├── requirements.txt
└── README.md
```

---

## 5. Technology stack

| Tool | Used for |
|---|---|
| Python 3 | Language |
| pandas / NumPy | Data handling |
| scikit-learn | Pipelines, models, CV, metrics, calibration |
| matplotlib | Plots |
| joblib | Saving / loading the trained pipeline |
| Flask | Web application |
| HTML / CSS / Jinja2 | Front end |

---

## 6. The ML pipeline, step by step

All steps are in `train_model.py`.

### Step 1: Load and clean

- Column names are stripped, lower-cased and spaces replaced by underscores.
- The script checks that every required column exists and stops with a clear message if not.
- Categorical text is stripped and title-cased (fixes values like `" Graduate"`).
- Numeric columns are converted with `pd.to_numeric(errors="coerce")`. Negatives become missing.
- The target is converted to 0/1 (it also understands "Approved"/"Rejected").
- The `id` column is not used as a feature.

### Step 2: Diagnostics

Printed before any modelling: shape, dtypes, missing values, unique values, duplicates, target balance, numeric summary, zero counts, IQR outlier counts, single-feature ROC-AUC, approval rate by category and by CIBIL band.

### Step 3: Stratified train/test split

`train_test_split(test_size=0.20, stratify=y, random_state=42)`.

- **Stratified** keeps the same approval ratio in both parts.
- The 856 test rows are **not touched** until the final evaluation.

### Step 4: Preprocessing (inside the pipeline)

| Column type | Steps |
|---|---|
| Numeric | `SimpleImputer(median, add_indicator=True)`, then `RobustScaler` |
| Categorical | `SimpleImputer(most_frequent)`, then `OneHotEncoder(handle_unknown="ignore")` |

- **Median imputation:** robust to outliers.
- **Missing indicator:** adds a flag column so the model can learn if "missing" itself is informative.
- **RobustScaler:** scales using median and IQR, so outliers affect it less than StandardScaler.
- **`handle_unknown="ignore"`:** an unseen category at prediction time will not crash the app.
- Everything is learned **inside a scikit-learn `Pipeline`**, so statistics such as medians are computed from training folds only. This prevents data leakage.

### Step 5: Six models compared

| Model | Settings |
|---|---|
| Logistic Regression | `max_iter=2000` |
| Decision Tree | `max_depth=5`, `min_samples_leaf=20` |
| Random Forest | 300 trees, `min_samples_leaf=5` |
| Extra Trees | 300 trees, `min_samples_leaf=5` |
| Gradient Boosting | 100 estimators, `max_depth=3` |
| Hist Gradient Boosting | `max_depth=3`, `learning_rate=0.05`, 150 iterations |

Each is evaluated with **5-fold stratified cross-validation on the training set only**, recording accuracy, precision, recall, F1 and ROC-AUC.

### Step 6: Model selection by ROC-AUC

The best model is the one with the highest **cross-validated ROC-AUC**.

**Why AUC and not F1 or accuracy?** F1, accuracy and recall depend on a threshold. Approvals are the majority class, so a useless model that approves everyone still scores a high F1 (this happened with the random-label CSV: recall 99.97%, F1 0.705, accuracy equal to the baseline). ROC-AUC measures how well the model *ranks* approved above rejected applicants regardless of the threshold, so it is a fairer selection criterion.

### Step 7: Signal check

The selected model is cross-validated again with the **training labels randomly shuffled**.

- Real labels: AUC 0.996
- Shuffled labels: AUC 0.484 (pure noise)

The big gap proves the model learned real structure. The script also prints a verdict (`NO_USEFUL_SIGNAL`, `WEAK_SIGNAL` or `USEFUL_SIGNAL`).

### Step 8: Probability calibration

The final model is wrapped in `CalibratedClassifierCV(method="sigmoid", cv=5)`. Random Forests tend to output badly scaled probabilities; sigmoid (Platt) calibration maps the raw scores so that "0.8" is closer to "about 80% of such cases are approved". The inner pipeline is re-fitted inside the calibration folds, so calibration only sees training data.

### Step 9: Decision threshold tuning

The model outputs a **probability**. The **decision** (Approved/Rejected) needs a threshold.

1. Get **out-of-fold** predicted probabilities on the training set (every row is predicted by a model that never saw it).
2. Try thresholds from 0.20 to 0.80 in steps of 0.01.
3. Choose the one with the highest **balanced accuracy** (the average of recall on both classes).
4. If the best threshold is not at least 0.5 percentage points better than 0.50, keep **0.50**.

**Result: 0.50.** No other threshold improved things meaningfully, because the model separates the classes very sharply (most probabilities are below 0.02 or above 0.99). The threshold is **never tuned on the test set**. The reason is saved in `model_config.json`.

*Why balanced accuracy instead of F1?* Maximising F1 on an imbalanced or weak-signal problem pushes the threshold toward 0 ("approve everyone"). Balanced accuracy treats both classes equally.

### Step 10: Final evaluation (once)

The calibrated pipeline is fitted on the full training set and evaluated **once** on the 856 held-out test rows.

### Step 11: Diagnostics and saving

Confusion matrix, model-comparison chart, ROC curve, precision-recall curve and permutation feature importance are generated. The pipeline is saved with `joblib`, and the threshold, valid input ranges and categories are saved in `model_config.json`.

### Data-leakage checklist

| Risk | How it is avoided |
|---|---|
| Scaler/imputer fitted on all data | They are inside the Pipeline and fit on training folds only |
| Model chosen with test data | Selection uses CV on the training set only |
| Threshold tuned on test data | Tuned on out-of-fold training predictions |
| Target used as a feature | `loan_status` and `id` are excluded |
| Test set reused | Evaluated exactly once |

---

## 7. Results

### Cross-validation on the training set (5-fold)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.916 | 0.932 | 0.934 | 0.933 | 0.966 |
| Decision Tree | 0.964 | 0.977 | 0.966 | 0.971 | 0.993 |
| **Random Forest** | 0.976 | 0.976 | 0.985 | 0.981 | **0.996** |
| Extra Trees | 0.940 | 0.945 | 0.960 | 0.952 | 0.985 |
| Gradient Boosting | 0.980 | 0.979 | 0.988 | 0.984 | 0.995 |
| Hist Gradient Boosting | 0.980 | 0.980 | 0.988 | 0.984 | 0.995 |

**Selected: Random Forest** (highest ROC-AUC). Gradient Boosting and HistGB have marginally higher accuracy and F1, but the differences are within the fold-to-fold noise (AUC std about 0.002), so all three tree ensembles are essentially tied. Logistic Regression is clearly worse because the CIBIL relationship is a step (a cut-off near 550), not a smooth linear one.

### Final test-set results (856 unseen rows, threshold 0.50)

| Metric | Value |
|---|---|
| Accuracy | **97.55%** |
| Balanced accuracy | 97.42% |
| Precision | 98.12% |
| Recall | 97.93% |
| F1 | 98.02% |
| ROC-AUC | 99.72% |
| Brier score | 0.0170 (lower is better; 0.25 means no skill) |
| Majority-class baseline accuracy | 62.15% |

### Confusion matrix

|  | Predicted Rejected | Predicted Approved |
|---|---|---|
| **Actual Rejected** | 314 | 10 |
| **Actual Approved** | 11 | 521 |

- 10 false approvals (a rejected applicant predicted approved)
- 11 false rejections (an approved applicant predicted rejected)

### Feature importance (permutation, drop in AUC)

| Feature | Importance |
|---|---|
| **cibil_score** | **0.4724** |
| loan_term | 0.0197 |
| loan_amount | 0.0029 |
| commercial_assets_value | 0.0022 |
| income_annum | 0.0012 |
| others | about 0 |

CIBIL score carries almost all the predictive power.

---

## 8. The web application

### Data flow

```
User fills form
      ↓
Server-side validation (types, ranges, relationships)
      ↓
One-row DataFrame with the exact training column names
      ↓
Saved pipeline (imputer → scaler → encoder → Random Forest → calibration)
      ↓
Estimated approval probability
      ↓
Compare with decision threshold from model_config.json (0.50)
      ↓
APPROVED / REJECTED + probability + disclaimer
```

### Validation (HTML and Flask)

- Each field has `min`/`max` from the training data's observed range (shown as "Supported: ..." under each field).
- Flask re-checks everything on the server, because HTML checks can be bypassed.
- Extra sanity floors: income at least ₹50,000, loan at least ₹10,000, CIBIL between 300 and 900.
- Zero assets are allowed (it is legitimate to have none).
- Dependents and term must be whole numbers.
- Relationship check: a loan more than 50 times the annual income is rejected as unrealistic.
- Out-of-range input shows: *"... is outside the range supported by this model"*.
- The app never crashes on bad input; it shows a list of errors instead.

These checks do **not** decide approval. They only stop clearly invalid inputs. The decision always comes from the model.

### Why the same preprocessing is guaranteed at prediction time

`best_model.pkl` contains the *entire* pipeline. Flask only builds a DataFrame with the right column names; scaling, imputing and encoding happen inside the pipeline exactly as in training.

### How to read the probability

"Estimated approval probability: 99.4%" means: *according to this model trained on this dataset, applicants with these characteristics were approved about 99% of the time.* It is **not** a guarantee and **not** a real bank decision. Probabilities are so extreme (about 1% or about 99%) because in this dataset approval is almost fully determined by the CIBIL score.

---

## 9. How to run

### Windows

```bash
# 1. activate the virtual environment
venv\Scripts\activate

# 2. install dependencies
pip install -r requirements.txt

# 3. train (creates everything inside models/)
python train_model.py

# 4. optional: test hand-made applicants
python test_predictions.py

# 5. start the web app
python app.py
```

Open **http://127.0.0.1:5000**.

If you replace the dataset, delete the contents of `models/` and retrain. `requirements.txt`:

```
flask>=2.3
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
matplotlib>=3.7
joblib>=1.3
```

---

## 10. Test inputs

Use **Self employed = No** in all rows.

| Case | Dep. | Education | Income | Loan | Term | CIBIL | Residential | Commercial | Luxury | Bank | Result |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Strong | 2 | Graduate | 5000000 | 15000000 | 10 | 750 | 8000000 | 4000000 | 10000000 | 5000000 | **APPROVED** (about 99%) |
| Weak | 4 | Not Graduate | 400000 | 9000000 | 18 | 350 | 0 | 0 | 500000 | 20000 | **REJECTED** (about 2%) |
| Low CIBIL only | 2 | Graduate | 5000000 | 15000000 | 10 | 420 | 8000000 | 4000000 | 10000000 | 5000000 | **REJECTED** |
| Borderline high | same as Strong, CIBIL 560 | | | | | | | | | | **APPROVED** |
| Borderline low | same as Strong, CIBIL 540 | | | | | | | | | | **REJECTED** |
| Invalid | 60 dependents, income 100, term 35 | | | | | | | | | | **Validation errors, no prediction** |

---

## 11. File-by-file explanation

| File | Role |
|---|---|
| `train_model.py` | Loads and cleans data, prints diagnostics, splits, compares 6 models with CV, selects the best by ROC-AUC, runs the shuffled-label test, calibrates, tunes the threshold, evaluates once on test, saves model/config/results/plots |
| `app.py` | Loads the pipeline and config, validates input, builds the DataFrame, gets the probability, applies the threshold, renders the page |
| `templates/index.html` | The form (generated from the config ranges), error box, result card, quality warning and disclaimer |
| `static/style.css` | Styling |
| `test_predictions.py` | Scores three hand-made applicants; the script does not know the expected answer |
| `models/model_config.json` | Selected model name, threshold and why, feature lists, categories, valid ranges, signal verdict |
| `models/model_results.json` | CV results, test metrics, confusion matrix, importances, shuffled-label AUC |

---

## 12. Limitations (read this before your viva)

1. **CIBIL dominates.** About 99% of the predictive power comes from one feature, and the data behaves almost like "CIBIL above about 550 is approved". The dataset is clean and probably synthetic or rule-generated. A real bank dataset would be messier and the accuracy would be lower.
2. **High accuracy does not mean real-world accuracy.** The model has only learned the pattern in this dataset.
3. **Probabilities are extreme** (about 1% or 99%) for the same reason. Calibration fixes the scale, not the dataset's sharpness.
4. **The three tree ensembles are statistically tied.** Random Forest was chosen by a very small AUC margin.
5. **Other features barely matter** (education, employment, dependents, and the asset values), unlike in real lending.
6. **No fairness analysis** was done.
7. **Small dataset** (4,278 rows).
8. **Not a real banking system.** It is an educational project.

---

## 13. Concepts glossary

| Term | Meaning |
|---|---|
| Binary classification | Predicting one of two classes |
| Train/test split | Training data builds the model; test data measures it fairly. Stratified keeps class ratios equal |
| Cross-validation | Split the training data into k folds; train on k-1, validate on 1, rotate, average |
| Data leakage | Information from test/validation data influencing training, which inflates scores |
| Pipeline | Chains preprocessing and model so they are fitted together |
| Imputation | Filling missing values (median for numbers, most frequent for categories) |
| One-hot encoding | Turning categories into 0/1 columns |
| Scaling | Putting features on comparable scales (RobustScaler uses median/IQR) |
| Accuracy | Correct predictions / all predictions |
| Precision | Of those predicted Approved, how many truly were |
| Recall | Of truly Approved, how many were found |
| F1 | Harmonic mean of precision and recall |
| ROC-AUC | Probability that a random approved applicant is ranked above a random rejected one (0.5 = chance, 1.0 = perfect) |
| Balanced accuracy | Average of recall on each class |
| Confusion matrix | Table of true/false positives and negatives |
| Brier score | Mean squared error of probabilities (lower is better) |
| Calibration | Making predicted probabilities match real frequencies |
| Decision threshold | The probability cut-off above which we say "Approved" |
| Out-of-fold prediction | A prediction made by a model that did not see that row in training |
| Permutation importance | How much a score drops when one feature's values are shuffled |
| Decision tree | A model of nested if/else splits |
| Random forest | Many decorrelated trees voting; reduces overfitting |
| Gradient boosting | Trees built one after another, each correcting the previous errors |
| Overfitting | Memorising training data and failing on new data |

---


## 14. Future improvements

- Hyperparameter search (`GridSearchCV` / `RandomizedSearchCV`).
- SHAP or LIME explanations for each prediction.
- Fairness and bias analysis across education and employment groups.
- Train on a real, noisier lending dataset.
- Add the loan-to-income ratio and total assets as engineered features.
- Deploy to a cloud host with logging and monitoring.
- Add automated unit tests for validation and prediction.