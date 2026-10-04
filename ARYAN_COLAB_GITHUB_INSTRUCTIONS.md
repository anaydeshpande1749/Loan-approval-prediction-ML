# Aryan - Colab Notebook + GitHub Instructions

## Loan Approval ML Project

### Your Task

1. Create an EDA notebook in Google Colab.
2. Run the EDA cells provided below.
3. Download the notebook as `.ipynb`.
4. Put the notebook inside the project's `notebooks/` folder.
5. Push the complete project to GitHub.

---

# Part 1 - Create the Colab Notebook

Open Google Colab and create a new notebook.

Rename it:

```text
01_eda_and_model_analysis.ipynb
```

Google Colab notebooks are Jupyter notebooks and can be downloaded as `.ipynb` files.

---

# Part 2 - Colab Notebook Cells

## Cell 1 - Title

Use a Markdown cell:

```markdown
# Loan Approval Prediction Using Machine Learning

## Exploratory Data Analysis and Model Analysis

Group 6
```

## Cell 2 - Import Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
```

## Cell 3 - Upload Dataset

```python
from google.colab import files

uploaded = files.upload()
```

Upload the project dataset:

```text
loan_approval.csv
```

## Cell 4 - Load Dataset

```python
df = pd.read_csv("loan_approval.csv")

print("Dataset shape:", df.shape)

df.head()
```

## Cell 5 - Dataset Information

```python
df.info()
```

## Cell 6 - Statistical Summary

```python
df.describe(include="all")
```

## Cell 7 - Missing Values

```python
missing = df.isnull().sum()

print(missing)
```

## Cell 8 - Target Distribution

```python
print(df["loan_status"].value_counts())
```

```python
plt.figure(figsize=(6, 4))

sns.countplot(data=df, x="loan_status")

plt.title("Loan Approval Status Distribution")
plt.xlabel("Loan Status")
plt.ylabel("Number of Applicants")

plt.show()
```

## Cell 9 - Education vs Loan Status

```python
plt.figure(figsize=(7, 4))

sns.countplot(
    data=df,
    x="education",
    hue="loan_status"
)

plt.title("Education vs Loan Approval Status")
plt.xlabel("Education")
plt.ylabel("Number of Applicants")

plt.show()
```

## Cell 10 - CIBIL Score Distribution

```python
plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="cibil_score",
    hue="loan_status",
    kde=True,
    bins=30
)

plt.title("CIBIL Score Distribution by Loan Status")
plt.xlabel("CIBIL Score")
plt.ylabel("Count")

plt.show()
```

## Cell 11 - Annual Income vs Loan Amount

```python
plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="income_annum",
    y="loan_amount",
    hue="loan_status",
    alpha=0.6
)

plt.title("Annual Income vs Loan Amount")
plt.xlabel("Annual Income")
plt.ylabel("Loan Amount")

plt.show()
```

## Cell 12 - Self Employment vs Loan Status

```python
plt.figure(figsize=(7, 4))

sns.countplot(
    data=df,
    x="self_employed",
    hue="loan_status"
)

plt.title("Self Employment vs Loan Approval")
plt.xlabel("Self Employed")
plt.ylabel("Count")

plt.show()
```

## Cell 13 - Correlation Heatmap

```python
numeric_df = df.select_dtypes(include=np.number)

plt.figure(figsize=(12, 8))

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.show()
```

## Cell 14 - Observations

Use a Markdown cell:

```markdown
## Observations

1. The dataset contains applicant financial and personal information.
2. Loan status is the target variable.
3. CIBIL score is an important credit-related feature.
4. Income and loan amount provide information about the applicant's financial profile.
5. Some columns contain missing values, which need to be handled before model training.
6. Categorical features such as education and self-employment need to be encoded before applying machine learning algorithms.
```

---

# Part 3 - Download the Notebook

In Google Colab:

```text
File
→ Download
→ Download .ipynb
```

The downloaded file should be:

```text
01_eda_and_model_analysis.ipynb
```

---

# Part 4 - Add the Notebook to the Project

The project should have this structure:

```text
loan_approval_ml_project/
│
├── data/
├── models/
├── notebooks/
│   └── 01_eda_and_model_analysis.ipynb
├── templates/
├── static/
├── app.py
├── train_model.py
├── requirements.txt
├── .gitignore
└── README.md
```

If `notebooks` does not exist, create it.

---

# Part 5 - IMPORTANT: Check .gitignore

The project `.gitignore` should contain:

```gitignore
venv/
__pycache__/
*.py[cod]
.ipynb_checkpoints/
.DS_Store

data/*.csv
```

Do NOT ignore:

```text
models/best_model.pkl
models/model_results.json
```

The trained model is required by `app.py`.

---

# Part 6 - GitHub Push

## Recommended method

After downloading the notebook, put it into the project folder on your computer.

Then open Command Prompt / Git Bash inside:

```text
loan_approval_ml_project
```

Run:

```bash
git status
```

Check that:

```text
notebooks/01_eda_and_model_analysis.ipynb
```

is visible.

Also check that `venv/` and `data/loan_approval.csv` are NOT staged.

Then:

```bash
git add .
```

Check:

```bash
git status
```

Then commit:

```bash
git commit -m "Add EDA and model analysis notebook"
```

If the GitHub repository is already connected:

```bash
git pull --rebase origin main
```

Then:

```bash
git push origin main
```

---

# Part 7 - If This Is a New GitHub Repository

If the repository has NOT been created yet:

1. Create an empty repository on GitHub.
2. Do not initialize it with another README, `.gitignore`, or license.
3. Open the project folder in Command Prompt / Git Bash.

Run:

```bash
git init
```

Then:

```bash
git add .
```

Then:

```bash
git commit -m "Initial loan approval ML project"
```

Then:

```bash
git branch -M main
```

Add the GitHub repository:

```bash
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
```

Verify:

```bash
git remote -v
```

Then:

```bash
git push -u origin main
```

Replace:

```text
YOUR_USERNAME
YOUR_REPOSITORY
```

with the actual GitHub username and repository name.

---

# Part 8 - If You Really Want to Push Directly from Colab

This is optional. The easier method is to download the `.ipynb` and push from your computer.

If the project ZIP is uploaded to Colab, extract it:

```python
!unzip -q Loan_Approval_ML_Project.zip -d project
```

Check the project:

```python
!find project -maxdepth 2 -type f
```

Create the notebooks directory:

```python
!mkdir -p project/loan_approval_ml_project/notebooks
```

Move the downloaded/created notebook into that directory if it is available in the Colab runtime.

Then:

```python
%cd project/loan_approval_ml_project
```

Initialize Git only if this is not already a Git repository:

```bash
!git init
```

Configure identity:

```bash
!git config user.name "Aryan Achary"
!git config user.email "YOUR_GITHUB_EMAIL"
```

Add files:

```bash
!git add .
```

Check:

```bash
!git status
```

Commit:

```bash
!git commit -m "Add EDA and model analysis notebook"
```

Add the GitHub remote:

```bash
!git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
```

Verify:

```bash
!git remote -v
```

Push:

```bash
!git branch -M main
!git push -u origin main
```

### Important

GitHub authentication may be required when pushing from Colab. Do not put a GitHub password or personal access token directly into a notebook that will be committed to the repository.

For this project, downloading the notebook and pushing from the computer is the safer and simpler option.

---

# Part 9 - Final Verification

After pushing, open the GitHub repository and confirm:

```text
app.py
train_model.py
requirements.txt
README.md
.gitignore

models/
    best_model.pkl
    model_results.json
    model_comparison.png
    confusion_matrix.png

notebooks/
    01_eda_and_model_analysis.ipynb

templates/
    index.html

static/
    style.css
```

Do NOT upload:

```text
venv/
__pycache__/
data/loan_approval.csv
```

---

# Current Model Results

The latest training run with balanced class weights produced:

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.5238 | 0.5666 | 0.5341 | 0.5499 |
| Random Forest | 0.5075 | 0.5530 | 0.5000 | 0.5252 |
| Decision Tree | 0.4789 | 0.5313 | 0.3680 | 0.4348 |

Best model according to the current F1-score comparison:

```text
Logistic Regression
```

The latest run generated:

```text
models/best_model.pkl
models/model_results.json
models/model_comparison.png
models/confusion_matrix.png
```

Do not manually change these results in the notebook or README. Use the values produced by the final verified training run.

---

# Final Task for Aryan

Your main responsibility is:

```text
Create EDA notebook
        ↓
Run all cells
        ↓
Check outputs/graphs
        ↓
Download .ipynb
        ↓
Put it in notebooks/
        ↓
Commit
        ↓
Push to GitHub
```

Do not modify:

```text
train_model.py
app.py
templates/index.html
static/style.css
```

unless Anay specifically asks you to.
