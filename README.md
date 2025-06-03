# Data Quality Classification Project

> Notebook: `main.ipynb`

---

## 1. Project Objective

Predict the **Data Quality** labels of water‑quality measurement records by building an end‑to‑end machine‑learning pipeline that cleans raw monitoring data, engineers domain‑aware features, and trains a robust XGBoost classifier.

## 2. Dataset

```
dataset/
├── train.csv              # 26 958 rows × 21 columns
├── test.csv               # 28 282 rows × 20 columns
├── train_modified.csv     # cleaned & feature‑engineered training data
└── test_modified.csv      # cleaned & feature‑engineered test data
```

Key target:

* **Data Quality** — categorical label with three classes

  * `Fair`
  * `Unknown`
  * `Good+Pending`

## 3. Notebook Workflow

| Section                              | Purpose                     | Key Actions                                                                                                                                                                                |
| ------------------------------------ | --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Exploratory Data Analysis (EDA)**  | Understand raw data         | ° Missing‑value heat‑map with *missingno*  \|  ° Statistical summary  \|  ° Histogram, boxplot & QQ‑plot per feature  \|  ° Shapiro & KS normality tests                                   |
| **Correlation Analysis**             | Identify redundant features | ° Pearson/Spearman heat‑maps for numeric  \|  ° Cramér’s V, ANOVA & Point‑Biserial tests for categorical                                                                                   |
| **Outlier & Low‑Variance Detection** | Improve signal‑to‑noise     | ° IQR boxplot inspection  \|  ° `VarianceThreshold` for numerics  \|  ° Top‑ratio filter for categoricals                                                                                  |
| **Data Pre‑processing**              | Prepare modelling tables    | ° Type casting  \|  ° KNN imputation (numerical)  \|  ° Domain‑specific grouping (`water_body_type_grouped`, `sampling_weather_grouped`)  \|  ° One‑hot encoding of 11 categorical columns |
| **Class‑Imbalance Handling**         | Stabilise learning          | ° `compute_class_weight` (sklearn) – weights: *Fair* 0.39, *Unknown* 1.11, *Good+Pending* 14.6                                                                                             |
| **Model Training & Evaluation**      | Build predictor             | ° 80∶20 stratified split  \|  ° **XGBClassifier** (max\_depth 6, eta 0.05, colsample\_bytree 0.8, subsample 0.9, n\_estimators 600)                                                        |
| **Submission Generation**            | Produce Kaggle file         | Apply identical pipeline to `test_modified.csv` ‑→ `submission/submission_xgboost.csv`                                                                                                     |

## 4. Model Performance

```
Confusion Matrix
                 Predicted
             Fair  Unknown  Good+Pending
Actual Fair    2 009     10              8
Actual Unknown    28    690              0
Actual G+P        33      0             22
```

| Metric       | Value    |
| ------------ | -------- |
| **Accuracy** | **0.97** |
| Macro F1     | 0.82     |
| Weighted F1  | 0.97     |

> *Insight*  Classes with scant support (especially **Good+Pending**) remain hard; synthetic oversampling (SMOTE), focal‑loss XGBoost, or calibrated thresholding are recommended next steps.

## 5. Repository Layout

```
.
├── dataset/               # raw & processed data
├── submission/
│   └── submission_xgboost.csv
├── main.ipynb            # full analysis notebook
├── requirements.txt      # exact package list
└── README.md             # ← you are here
```

## 6. Reproducing the Results

```bash
# 1. clone the repo
$ git clone <your‑repo‑url>
$ cd data‑quality‑ml

# 2. create & activate environment
$ python -m venv .env
$ source .env/bin/activate  # Windows: .env\Scripts\activate

# 3. install dependencies
$ pip install -r requirements.txt

# 4. run the notebook
$ jupyter lab  # or jupyter notebook
# Execute `main.ipynb` top‑to‑bottom
```

Python ≥ 3.10 | Core libraries: **pandas**, **numpy**, **scikit‑learn 1.5+**, **xgboost 2.0+**, **seaborn**, **matplotlib**, **missingno**.

## 7. Future Work

* **Hyper‑parameter optimisation** with Optuna/Bayesian search.
* Benchmark **LightGBM**, **CatBoost**, and **TabNet** for skewed‑categorical robustness.
* **Explainability** with SHAP to trace feature contributions.
* Deploy as a **FastAPI** micro‑service with a CI/CD test suite.

---

*Last updated : 3 June 2025*
