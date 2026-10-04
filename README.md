# Loan Approval Prediction Using Machine Learning

## Group
Group 6

## Project Title
**Loan Approval Prediction Using Machine Learning**

## Team Members
- Aryan Achary — Roll No. 1
- Vikas Bhagat — Roll No. 8
- Anay Deshpande — Roll No. 20


## 1. Project Objective

The objective of this project is to build a machine learning based classification system that predicts whether a loan application is likely to be **Approved** or **Rejected** using applicant financial, personal and asset-related information.

The project demonstrates the complete machine learning workflow:

**Dataset → EDA → Data Preprocessing → Model Training → Model Evaluation → Best Model → User Input Prediction**

## 2. Problem Definition

Loan approval depends on several factors such as income, loan amount, credit score, employment status, loan term and assets owned by the applicant.

The problem is formulated as a **binary classification problem**:

- `1` → Loan Approved
- `0` → Loan Rejected

## 3. Dataset

The project uses a publicly available Loan Approval Prediction dataset.

The dataset contains applicant-related features such as:

- Number of dependents
- Education
- Self employment status
- Annual income
- Loan amount
- Loan term
- CIBIL score
- Residential asset value
- Commercial asset value
- Luxury asset value
- Bank asset value
- Loan status

Dataset source:

https://www.kaggle.com/competitions/loan-approval-prediction-cpe-232-data-models-intl2/data

The dataset used locally is stored as:

`data/loan_approval.csv`

## 4. Machine Learning Algorithms

Three supervised classification algorithms are implemented:

### 4.1 Logistic Regression
A linear classification algorithm used to estimate the probability of a binary outcome.

### 4.2 Decision Tree
A tree-based model that makes decisions using feature-based conditions.

### 4.3 Random Forest
An ensemble of multiple decision trees whose predictions are combined to improve generalization.

## 5. Data Preprocessing

The preprocessing pipeline performs:

- Missing-value handling
- Numerical feature imputation
- Numerical feature scaling
- Categorical feature imputation
- One-hot encoding of categorical variables

The preprocessing is included inside the scikit-learn pipeline so that the same transformations are applied during both training and user prediction.

## 6. Model Evaluation

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

The F1 Score is particularly useful for this binary classification problem because it balances precision and recall.

### Initial local run

The first training run on the current 32,000-row dataset produced:

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.5464 | 0.5469 | 0.9748 | 0.7007 |
| Random Forest | 0.5461 | 0.5470 | 0.9696 | 0.6994 |
| Decision Tree | 0.5347 | 0.5474 | 0.8411 | 0.6632 |

**Important:** these are the initial baseline results. They should not be treated as the final project performance. The model should be improved and re-evaluated before the final report/presentation.

## 7. Visualizations

The training script generates:

- `models/model_comparison.png`
- `models/confusion_matrix.png`

These visualizations are used to compare model performance and interpret classification results.

## 8. Web Application

A Flask web application is provided for prediction on new applicant information.

The user enters:

- Number of dependents
- Education
- Self employment
- Annual income
- Loan amount
- Loan term
- CIBIL score
- Residential assets
- Commercial assets
- Luxury assets
- Bank assets

The application sends the input to the trained machine learning pipeline and displays:

**LOAN APPROVED** or **LOAN REJECTED**

The application can also display the predicted approval probability when supported by the selected model.

## 9. Project Structure

```text
loan_approval_ml_project/
│
├── data/
│   └── loan_approval.csv
│
├── models/
│   ├── best_model.pkl
│   ├── model_results.json
│   ├── model_comparison.png
│   └── confusion_matrix.png
│
├── notebooks/
│   └── 01_eda_and_model_analysis.ipynb
│
├── templates/
│   └── index.html
│
├── static/
│   └── style.css
│
├── app.py
├── train_model.py
├── requirements.txt
├── .gitignore
└── README.md
```

## 10. How to Run the Project

### Step 1: Create virtual environment

```bash
python -m venv venv
```

### Step 2: Activate virtual environment

```bash
venv\Scripts\activate
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Add the dataset

Place the dataset at:

```text
data/loan_approval.csv
```

### Step 5: Train the models

```bash
python train_model.py
```

### Step 6: Start the Flask application

```bash
python app.py
```

### Step 7: Open the application

```text
http://127.0.0.1:5000
```

## 11. Expected Project Workflow

```text
Loan Application Dataset
          ↓
Exploratory Data Analysis
          ↓
Data Preprocessing
          ↓
Train/Test Split
          ↓
Logistic Regression
Decision Tree
Random Forest
          ↓
Model Evaluation
          ↓
Best Model Selection
          ↓
Flask Web Application
          ↓
New Applicant Input
          ↓
Approved / Rejected
```

## 12. Deliverables

The project covers the required mini-project components:

1. Introduction to the topic
2. Problem definition
3. Dataset overview
4. EDA
5. Dataset visualization
6. Machine learning algorithm explanation
7. Model evaluation
8. Confusion matrix
9. Model comparison
10. Unseen/user input prediction
11. Result interpretation
12. Conclusion
13. Project/GitHub link

## 13. Important Notes

- Do not upload the `venv/` folder to GitHub.
- Do not upload generated Python cache files.
- The trained model and model-result files may be committed so that the Flask application can run directly after cloning.
- Do not upload the CSV if the dataset's competition/licensing rules do not permit redistribution. Instead, provide the official dataset source in this README.
- Final performance values in the report should be taken from the final verified training run, not copied from this README.

## 14. References

1. Kaggle — Loan Approval Prediction dataset:
https://www.kaggle.com/competitions/loan-approval-prediction-cpe-232-data-models-intl2/data

2. Scikit-learn documentation:
https://scikit-learn.org/stable/

3. Flask documentation:
https://flask.palletsprojects.com/