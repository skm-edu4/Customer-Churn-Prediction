import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (classification_report, roc_auc_score, roc_curve,
                             ConfusionMatrixDisplay)

SEED = 42            # fix all randomness -> reproducible results
SHOW_PLOTS = True    # set False on headless servers; PNGs are still saved


def banner(title):
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


# --------------------------------------------------------------------------
# [1] DATA — build a realistic synthetic churn dataset.
#     With your own data, just replace this with:  df = pd.read_csv("file.csv")
# --------------------------------------------------------------------------
def generate_churn_dataset(n_customers=5000, seed=SEED):
    rng = np.random.default_rng(seed)

    tenure = np.clip(rng.normal(32, 22, n_customers), 1, 72).round()

    contract_type = rng.choice(["Month-to-month", "One year", "Two year"],
                               size=n_customers, p=[0.55, 0.25, 0.20])
    internet_service = rng.choice(["Fiber optic", "DSL", "No internet"],
                                  size=n_customers, p=[0.45, 0.35, 0.20])
    tech_support = rng.choice(["Yes", "No"], size=n_customers, p=[0.40, 0.60])
    payment_method = rng.choice(["Electronic check", "Mailed check",
                                 "Bank transfer", "Credit card"],
                                size=n_customers, p=[0.35, 0.25, 0.20, 0.20])

    base = {"Fiber optic": 85.0, "DSL": 60.0, "No internet": 25.0}
    monthly_charges = np.array([base[s] for s in internet_service]) \
                      + rng.normal(0, 12, n_customers)
    monthly_charges = np.clip(monthly_charges, 18, 120).round(2)

    total_charges = (monthly_charges * tenure
                     * rng.normal(1.0, 0.05, n_customers)).round(2)

    num_tickets = (rng.poisson(1.2, n_customers)
                   + rng.poisson(np.where(monthly_charges > 80, 0.6, 0.1),
                                 n_customers))

    # The hidden "true" churn mechanism that our model must rediscover:
    logit = (-1.40
             + 1.50 * (contract_type == "Month-to-month")
             - 0.90 * (contract_type == "Two year")
             + 0.80 * (internet_service == "Fiber optic")
             - 0.70 * (tech_support == "Yes")
             + 0.30 * num_tickets
             + 0.012 * (monthly_charges - 65)
             - 0.018 * tenure
             + rng.normal(0, 0.8, n_customers))
    churned = (rng.random(n_customers) < 1 / (1 + np.exp(-logit))).astype(int)

    df = pd.DataFrame({
        "tenure": tenure,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "num_tickets": num_tickets.astype(float),
        "contract_type": contract_type,
        "internet_service": internet_service,
        "tech_support": tech_support,
        "payment_method": payment_method,
        "churned": churned,                 # target variable (0/1)
    })

    # Real data is messy: inject some missing values
    df.loc[rng.random(n_customers) < 0.03, "total_charges"] = np.nan
    df.loc[rng.random(n_customers) < 0.02, "num_tickets"] = np.nan
    return df


banner("[1] DATASET")
df = generate_churn_dataset()
print(f"Shape: {df.shape}")
print(df.head())
print("\nMissing values per column:")
print(df.isna().sum())


# --------------------------------------------------------------------------
# [2] EXPLORATORY DATA ANALYSIS — look before you model
# --------------------------------------------------------------------------
banner("[2] EXPLORATORY DATA ANALYSIS")
print(f"Churn rate: {df['churned'].mean():.1%}  "
      "(imbalanced -> we'll need metrics beyond plain accuracy)")

fig, axes = plt.subplots(2, 2, figsize=(11, 8))

df["churned"].value_counts().sort_index().plot.bar(
    ax=axes[0, 0], color=["steelblue", "tomato"], rot=0)
axes[0, 0].set_title("Churn distribution (0 = stayed, 1 = left)")

for label, color in [(0, "steelblue"), (1, "tomato")]:
    df.loc[df["churned"] == label, "tenure"].hist(
        ax=axes[0, 1], alpha=0.6, bins=18, color=color,
        label="stayed" if label == 0 else "churned")
axes[0, 1].set_title("Tenure by outcome")
axes[0, 1].legend()

(df.groupby("contract_type")["churned"].mean()
   .plot.bar(ax=axes[1, 0], color="seagreen", rot=15))
axes[1, 0].set_title("Churn rate by contract type")
axes[1, 0].set_ylim(0, 1)

axes[1, 1].scatter(df["tenure"], df["monthly_charges"],
                   c=df["churned"], cmap="coolwarm", alpha=0.35, s=12)
axes[1, 1].set_title("Tenure vs monthly charges")

plt.tight_layout()
plt.savefig("01_eda.png", dpi=130)


# --------------------------------------------------------------------------
# [3] TRAIN/TEST SPLIT — never evaluate on data the model trained on
# --------------------------------------------------------------------------
banner("[3] TRAIN / TEST SPLIT")
X = df.drop(columns=["churned"])
y = df["churned"]

# stratify=y keeps the churn ratio identical in both halves
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=SEED)
print(f"Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")
print("Rule: the test set stays UNTOUCHED until the very end.")


# --------------------------------------------------------------------------
# [4] PREPROCESSING PIPELINE
#     Fitting happens inside the pipeline on TRAIN data only,
#     which prevents data leakage.
# --------------------------------------------------------------------------
banner("[4] PREPROCESSING PIPELINE")
numeric_features = ["tenure", "monthly_charges", "total_charges", "num_tickets"]
categorical_features = ["contract_type", "internet_service",
                        "tech_support", "payment_method"]

numeric_transform = Pipeline([
    ("impute", SimpleImputer(strategy="median")),   # fill missing values
    ("scale", StandardScaler()),                    # mean 0, std 1 (matters for linear models)
])
categorical_transform = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("encode", OneHotEncoder(handle_unknown="ignore")),  # categories -> numbers
])

preprocessor = ColumnTransformer([
    ("num", numeric_transform, numeric_features),
    ("cat", categorical_transform, categorical_features),
])
print("num features -> median impute + standard scaling")
print("cat features -> mode impute + one-hot encoding")


# --------------------------------------------------------------------------
# [5] MODEL CANDIDATES
# --------------------------------------------------------------------------
banner("[5] MODEL CANDIDATES")

def make_pipeline(estimator):
    return Pipeline([("pre", preprocessor), ("model", estimator)])

candidates = {
    # class_weight="balanced" up-weights the minority class -> handles imbalance
    "LogisticRegression": make_pipeline(
        LogisticRegression(class_weight="balanced", max_iter=2000)),
    "RandomForest": make_pipeline(
        RandomForestClassifier(class_weight="balanced", random_state=SEED)),
    "GradientBoosting": make_pipeline(
        GradientBoostingClassifier(random_state=SEED)),
}
print("Candidates: " + ", ".join(candidates))


# --------------------------------------------------------------------------
# [6] CROSS-VALIDATION — compare models robustly (5 folds)
# --------------------------------------------------------------------------
banner("[6] 5-FOLD CROSS-VALIDATION")
cv_summary = []
for name, pipe in candidates.items():
    cv = cross_validate(pipe, X_train, y_train, cv=5,
                        scoring=["accuracy", "f1", "roc_auc"],
                        return_train_score=True, n_jobs=-1)
    cv_summary.append({
        "model": name,
        "train_acc": cv["train_accuracy"].mean(),   # diagnose overfitting
        "val_acc": cv["test_accuracy"].mean(),
        "val_f1": cv["test_f1"].mean(),
        "val_roc_auc": cv["test_roc_auc"].mean(),
    })

cv_table = pd.DataFrame(cv_summary).set_index("model").round(3)
print(cv_table)
print("\ntrain_acc > val_acc for the trees = mild overfitting.")
print("Cross-validation is what keeps us honest about it.")


# --------------------------------------------------------------------------
# [7] HYPERPARAMETER TUNING — GridSearchCV over Random Forest
# --------------------------------------------------------------------------
banner("[7] HYPERPARAMETER TUNING (GridSearchCV)")
tuning_pipe = make_pipeline(RandomForestClassifier(class_weight="balanced",
                                                   random_state=SEED))
param_grid = {
    "model__n_estimators": [100, 300],
    "model__max_depth": [4, 8, None],
    "model__min_samples_leaf": [1, 4],
}
search = GridSearchCV(tuning_pipe, param_grid, cv=5,
                      scoring="roc_auc", n_jobs=-1)
search.fit(X_train, y_train)

print("Best parameters:", search.best_params_)
print(f"Best CV ROC-AUC:  {search.best_score_:.3f}")
best_model = search.best_estimator_


# --------------------------------------------------------------------------
# [8] FINAL EVALUATION — on the held-out test set, exactly once
# --------------------------------------------------------------------------
banner("[8] FINAL EVALUATION ON THE TEST SET")
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred, target_names=["stayed", "churned"]))
print(f"ROC-AUC on test set: {roc_auc_score(y_test, y_proba):.3f}")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test,
                                      cmap="Blues", ax=axes[0])
axes[0].set_title("Confusion matrix")

fpr, tpr, _ = roc_curve(y_test, y_proba)
axes[1].plot(fpr, tpr, label=f"ROC (AUC = {roc_auc_score(y_test, y_proba):.3f})")
axes[1].plot([0, 1], [0, 1], "k--", label="random guess")
axes[1].set_xlabel("False positive rate")
axes[1].set_ylabel("True positive rate")
axes[1].set_title("ROC curve")
axes[1].legend()
plt.tight_layout()
plt.savefig("02_evaluation.png", dpi=130)


# --------------------------------------------------------------------------
# [9] FEATURE IMPORTANCE — what drives predictions?
# --------------------------------------------------------------------------
banner("[9] FEATURE IMPORTANCE")
feature_names = best_model.named_steps["pre"].get_feature_names_out()
importances = best_model.named_steps["model"].feature_importances_

top = pd.Series(importances, index=feature_names).sort_values().tail(10)
print(top.sort_values(ascending=False).round(3))

plt.figure(figsize=(8, 5))
top.plot.barh(color="mediumpurple")
plt.title("Top 10 features (Random Forest importance)")
plt.tight_layout()
plt.savefig("03_feature_importance.png", dpi=130)


# --------------------------------------------------------------------------
# [10] UNSUPERVISED LEARNING — no labels this time (PCA + K-Means)
# --------------------------------------------------------------------------
banner("[10] UNSUPERVISED LEARNING: PCA + K-MEANS")
X_train_proc = preprocessor.fit_transform(X_train)
if hasattr(X_train_proc, "toarray"):      # sparse output from one-hot encoding
    X_train_proc = X_train_proc.toarray()

pca = PCA(n_components=2, random_state=SEED)
X_pca = pca.fit_transform(X_train_proc)
print(f"2 PCA components explain {pca.explained_variance_ratio_.sum():.1%} of variance")

clusters = KMeans(n_clusters=3, n_init=10, random_state=SEED).fit_predict(X_train_proc)

stats = (pd.DataFrame({"cluster": clusters, "churned": y_train.values})
           .groupby("cluster")["churned"].agg(["mean", "count"]).round(3))
print("\nChurn rate per cluster (K-Means never saw the labels!):")
print(stats)

plt.figure(figsize=(7, 5))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap="viridis", alpha=0.35, s=10)
plt.title("K-Means clusters (PCA projection)")
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.tight_layout()
plt.savefig("04_clusters.png", dpi=130)


# --------------------------------------------------------------------------
# [11] MODEL PERSISTENCE — save it, reload it, verify it
# --------------------------------------------------------------------------
banner("[11] MODEL PERSISTENCE")
MODEL_PATH = "churn_model.joblib"
joblib.dump(best_model, MODEL_PATH)
print(f"Saved full fitted pipeline to '{MODEL_PATH}'")

loaded_model = joblib.load(MODEL_PATH)
print("Reloaded model reproduces predictions:",
      (loaded_model.predict(X_test) == y_pred).all())


# --------------------------------------------------------------------------
# [12] INFERENCE — use the model on brand-new data
# --------------------------------------------------------------------------
banner("[12] INFERENCE ON NEW DATA")
new_customers = pd.DataFrame([
    {"tenure": 2,  "monthly_charges": 92.5, "total_charges": 185.0,
     "num_tickets": 3, "contract_type": "Month-to-month",
     "internet_service": "Fiber optic", "tech_support": "No",
     "payment_method": "Electronic check"},                      # risky profile
    {"tenure": 48, "monthly_charges": 55.0, "total_charges": 2640.0,
     "num_tickets": 0, "contract_type": "Two year",
     "internet_service": "DSL", "tech_support": "Yes",
     "payment_method": "Credit card"},                           # loyal profile
])

for i, p in enumerate(loaded_model.predict_proba(new_customers)[:, 1], 1):
    print(f"Customer {i}: churn probability = {p:.1%}")

print("\nDone! Saved figures: 01_eda.png, 02_evaluation.png, "
      "03_feature_importance.png, 04_clusters.png")

if SHOW_PLOTS:
    plt.show()
