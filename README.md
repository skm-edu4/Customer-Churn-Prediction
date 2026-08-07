# 📉 Customer Churn Prediction — End-to-End ML Project

> A single-file machine learning project that walks through **every fundamental of classical ML** — from raw data to a saved, reusable model.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🎯 About

This project predicts whether a customer will **churn** (cancel their service). The dataset is generated synthetically in code, so there are **no downloads or API keys** — just clone and run.

It was built as a **learning project**: one script, twelve clearly labeled sections, covering almost every core concept of classical machine learning.

## 🧠 Concepts Covered

| # | Section | ML Fundamentals |
|---|---------|-----------------|
| 1 | Dataset | Data generation, mixed types, missing values |
| 2 | EDA | Exploratory Data Analysis, visualization |
| 3 | Train/Test Split | Holdout evaluation, stratification, avoiding leakage |
| 4 | Preprocessing | Imputation, scaling, one-hot encoding, `Pipeline` + `ColumnTransformer` |
| 5 | Model Candidates | Logistic Regression, Random Forest, Gradient Boosting |
| 6 | Cross-Validation | k-fold CV, model comparison, overfitting diagnosis |
| 7 | Hyperparameter Tuning | `GridSearchCV` |
| 8 | Evaluation | Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrix |
| 9 | Interpretability | Feature importance |
| 10 | Unsupervised Learning | PCA, K-Means clustering |
| 11 | Persistence | Save/load models with `joblib` |
| 12 | Inference | Predictions on brand-new data |

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- Dependencies:

```bash
pip install numpy pandas scikit-learn matplotlib
```

### Installation & Usage

```bash
git clone https://github.com/<your-username>/churn-ml-project.git
cd churn-ml-project
python churn_project.py
```

The script prints results to the console and saves:

| File | Description |
|------|-------------|
| `01_eda.png` | Exploratory data analysis plots |
| `02_evaluation.png` | Confusion matrix + ROC curve |
| `03_feature_importance.png` | Top-10 feature importance |
| `04_clusters.png` | K-Means clusters (PCA projection) |
| `churn_model.joblib` | The full trained pipeline |

## 📊 Sample Output

```
model                train_acc  val_acc  val_f1  val_roc_auc
LogisticRegression       0.845    0.842   0.761        0.912
RandomForest             0.968    0.879   0.812        0.934
GradientBoosting         0.889    0.881   0.820        0.938

Best parameters: {'model__max_depth': 8, 'model__min_samples_leaf': 4, ...}
Best CV ROC-AUC:  0.941

ROC-AUC on test set: 0.937

Customer 1: churn probability = 78.4%   ← risky profile
Customer 2: churn probability =  4.1%   ← loyal profile
```

> Your exact numbers will vary slightly depending on library versions.

## 📁 Project Structure

```
churn-ml-project/
├── churn_project.py      # the entire project (single file)
├── README.md
└── requirements.txt      # dependencies
```

`requirements.txt`:

```
numpy>=1.24
pandas>=2.0
scikit-learn>=1.2
matplotlib>=3.7
```

## 🔮 Roadmap / Ideas to Extend

- [ ] Swap in the real [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) dataset (the pipeline works as-is)
- [ ] Replace `GridSearchCV` with `RandomizedSearchCV` or `Optuna`
- [ ] Add XGBoost / LightGBM as candidates
- [ ] Add learning curves to visualize bias–variance tradeoff
- [ ] Serve the model with FastAPI or Streamlit
- [ ] Try SMOTE instead of `class_weight` for imbalance

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

Built with [scikit-learn](https://scikit-learn.org/) as a hands-on tour of classical machine learning fundamentals.

---

⭐ If this helped you learn ML, feel free to star the repo!
