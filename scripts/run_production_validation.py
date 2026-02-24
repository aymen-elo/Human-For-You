"""
Production validation pipeline — 24 features.
Runs the full pipeline: preprocess, benchmark 9 models, tune top 3,
calibrate threshold via CV, evaluate on test set.
"""
import pandas as pd
import numpy as np
import warnings
import time
import json
import joblib
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_predict, cross_val_score,
    GridSearchCV, RandomizedSearchCV
)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    average_precision_score, classification_report, confusion_matrix
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# =====================================================================
# 1. Rebuild corrected dataset
# =====================================================================
print("=" * 70)
print("PRODUCTION VALIDATION — 24 Features")
print("=" * 70)

print("\n[1/5] Building dataset...")
general = pd.read_csv(DATA_DIR / "general_data.csv")
emp = pd.read_csv(DATA_DIR / "employee_survey_data.csv")
mgr = pd.read_csv(DATA_DIR / "manager_survey_data.csv")
in_time = pd.read_csv(DATA_DIR / "in_time.csv")
out_time = pd.read_csv(DATA_DIR / "out_time.csv")

# Badge H1 2015
emp_ids = in_time.iloc[:, 0].astype(int)
in_dates = in_time.iloc[:, 1:]
out_dates = out_time.iloc[:, 1:]
date_cols = [c for c in in_dates.columns
             if pd.to_datetime(c, errors="coerce") is not pd.NaT
             and pd.to_datetime(c, errors="coerce") <= pd.Timestamp("2015-06-30")]
in_d = in_dates[date_cols].apply(pd.to_datetime, errors="coerce")
out_d = out_dates[date_cols].apply(pd.to_datetime, errors="coerce")
in_h = in_d.apply(lambda c: c.dt.hour + c.dt.minute / 60)
out_h = out_d.apply(lambda c: c.dt.hour + c.dt.minute / 60)
wh = out_h - in_h
valid = in_h.notna() & out_h.notna()
n_days = in_h.shape[1]
n_present = valid.sum(axis=1)
n_absent = n_days - n_present
late_mask = valid & (in_h >= 10)
n_late = late_mask.sum(axis=1)
badge = pd.DataFrame({
    "EmployeeID": emp_ids.values,
    "avg_working_hours": wh.where(valid).mean(axis=1),
    "absence_rate": n_absent / n_days,
    "late_arrival_rate": (n_late / n_present).where(n_present > 0),
})

df = general.copy()
df = df.merge(emp, on="EmployeeID").merge(mgr, on="EmployeeID").merge(badge, on="EmployeeID")
df = df.drop(columns=["EmployeeCount", "Over18", "StandardHours"])
df["Attrition"] = (df["Attrition"] == "Yes").astype(int)

# Feature engineering
df["IncomePerJobLevel"] = df["MonthlyIncome"] / df["JobLevel"]
df["PromotionStagnation"] = df["YearsSinceLastPromotion"] / (df["YearsAtCompany"] + 1)
df["SatisfactionScore"] = df[["EnvironmentSatisfaction", "JobSatisfaction", "WorkLifeBalance"]].mean(axis=1)
df["ManagerStability"] = df["YearsWithCurrManager"] / (df["YearsAtCompany"] + 1)
df["BusinessTravel"] = df["BusinessTravel"].map(
    {"Non-Travel": 0, "Travel_Rarely": 1, "Travel_Frequently": 2}
)
cat_cols = df.select_dtypes(include="object").columns.tolist()
df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)

# Feature whitelist (24)
FINAL_FEATURES = [
    "avg_working_hours", "late_arrival_rate",
    "Age", "TotalWorkingYears", "YearsAtCompany", "MonthlyIncome",
    "YearsWithCurrManager", "NumCompaniesWorked", "DistanceFromHome",
    "PercentSalaryHike", "TrainingTimesLastYear", "YearsSinceLastPromotion",
    "BusinessTravel", "MaritalStatus_Single",
    "EnvironmentSatisfaction", "JobSatisfaction", "WorkLifeBalance",
    "IncomePerJobLevel", "ManagerStability", "SatisfactionScore", "PromotionStagnation",
    "MaritalStatus_Married", "JobRole_Manufacturing Director",
    "EducationField_Technical Degree",
]

df = df.drop(columns=["EmployeeID"])
X = df[[f for f in FINAL_FEATURES if f in df.columns]]
y = df["Attrition"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train = X_train.copy()
X_test = X_test.copy()

# Impute
cols_na = X_train.columns[X_train.isnull().any()].tolist()
if cols_na:
    imp = KNNImputer(n_neighbors=5, weights="distance")
    X_train[cols_na] = imp.fit_transform(X_train[cols_na])
    X_test[cols_na] = imp.transform(X_test[cols_na])
    joblib.dump(imp, OUTPUT_DIR / "imputer.joblib")

# Save unscaled for fairness
X_train.to_csv(OUTPUT_DIR / "X_train_unscaled.csv", index=False)
X_test.to_csv(OUTPUT_DIR / "X_test_unscaled.csv", index=False)

# Scale
scaler = StandardScaler()
X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_test_s = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
joblib.dump(scaler, OUTPUT_DIR / "scaler.joblib")

# Save
X_train_s.to_csv(OUTPUT_DIR / "X_train_no_smote.csv", index=False)
X_test_s.to_csv(OUTPUT_DIR / "X_test.csv", index=False)
y_train.to_csv(OUTPUT_DIR / "y_train_no_smote.csv", index=False)
y_test.to_csv(OUTPUT_DIR / "y_test.csv", index=False)
pd.Series(list(X_train.columns)).to_csv(OUTPUT_DIR / "feature_names.csv", index=False, header=False)

print(f"  Train: {X_train_s.shape} | Test: {X_test_s.shape}")
print(f"  Features: {list(X_train_s.columns)}")

# =====================================================================
# 2. Benchmark 9 models
# =====================================================================
print("\n[2/5] Benchmarking 9 models...")

_spw = (y_train == 0).sum() / (y_train == 1).sum()

MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight="balanced"),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss",
                              verbosity=0, scale_pos_weight=_spw),
    "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=42, class_weight="balanced"),
    "KNN": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    "Decision Tree": DecisionTreeClassifier(random_state=42, class_weight="balanced"),
    "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
    "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42),
}

bench_results = []
for name, model in MODELS.items():
    t0 = time.time()
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    y_proba = model.predict_proba(X_test_s)[:, 1] if hasattr(model, "predict_proba") else None
    elapsed = time.time() - t0

    bench_results.append({
        "Model": name,
        "Accuracy": round(float((y_pred == y_test).mean()), 4),
        "Precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "Recall": round(float(recall_score(y_test, y_pred)), 4),
        "F1-Score": round(float(f1_score(y_test, y_pred)), 4),
        "AUC-ROC": round(float(roc_auc_score(y_test, y_proba)), 4) if y_proba is not None else np.nan,
        "Train Time (s)": round(elapsed, 2),
    })

df_bench = pd.DataFrame(bench_results).sort_values("F1-Score", ascending=False).reset_index(drop=True)
df_bench.to_csv(OUTPUT_DIR / "model_benchmark_results.csv", index=False)
print(df_bench[["Model", "F1-Score", "AUC-ROC", "Recall"]].to_string(index=False))

# =====================================================================
# 3. Tune top 3
# =====================================================================
print("\n[3/5] Tuning top 3 models...")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

TUNING = {
    "Random Forest": {
        "model": RandomForestClassifier(random_state=42, n_jobs=-1, class_weight="balanced"),
        "params": {
            "n_estimators": [100, 200, 300],
            "max_depth": [10, 15, 20, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        }
    },
    "XGBoost": {
        "model": XGBClassifier(random_state=42, eval_metric="logloss", verbosity=0),
        "params": {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 5, 7, 9],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "subsample": [0.7, 0.8, 1.0],
            "scale_pos_weight": [1, 3, 5],
        }
    },
    "Gradient Boosting": {
        "model": GradientBoostingClassifier(random_state=42),
        "params": {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.05, 0.1],
            "min_samples_split": [2, 5, 10],
            "subsample": [0.7, 0.8, 1.0],
        }
    },
}

tuned = {}
for name, cfg in TUNING.items():
    n_combos = int(np.prod([len(v) for v in cfg["params"].values()]))
    if n_combos > 200:
        search = RandomizedSearchCV(
            cfg["model"], cfg["params"], n_iter=50,
            scoring="f1", cv=cv, n_jobs=-1, random_state=42, verbose=0
        )
    else:
        search = GridSearchCV(
            cfg["model"], cfg["params"],
            scoring="f1", cv=cv, n_jobs=-1, verbose=0
        )
    search.fit(X_train_s, y_train)
    tuned[name] = search.best_estimator_
    print(f"  {name}: CV-F1={search.best_score_:.4f} | {search.best_params_}")

# =====================================================================
# 4. Threshold calibration via CV
# =====================================================================
print("\n[4/5] Threshold calibration (CV-based)...")

best_overall = {"name": None, "f1": 0, "threshold": 0.5, "model": None}

for name, model in tuned.items():
    # Out-of-fold probabilities
    y_proba_cv = cross_val_predict(
        model.__class__(**model.get_params()), X_train_s, y_train,
        cv=cv, method="predict_proba", n_jobs=-1
    )[:, 1]

    # Sweep thresholds
    best_t, best_f1 = 0.5, 0
    for t in np.arange(0.10, 0.91, 0.01):
        yp = (y_proba_cv >= t).astype(int)
        f1 = f1_score(y_train, yp, zero_division=0)
        rec = recall_score(y_train, yp, zero_division=0)
        if f1 > best_f1 and rec >= 0.60:
            best_f1 = f1
            best_t = t

    y_cv_pred = (y_proba_cv >= best_t).astype(int)
    cv_f1 = f1_score(y_train, y_cv_pred)
    cv_prec = precision_score(y_train, y_cv_pred)
    cv_rec = recall_score(y_train, y_cv_pred)
    cv_auc = roc_auc_score(y_train, y_proba_cv)
    cv_pr_auc = average_precision_score(y_train, y_proba_cv)

    # Test set (no re-tuning)
    model.fit(X_train_s, y_train)
    y_proba_test = model.predict_proba(X_test_s)[:, 1]
    y_pred_test = (y_proba_test >= best_t).astype(int)
    test_f1 = f1_score(y_test, y_pred_test)
    test_prec = precision_score(y_test, y_pred_test, zero_division=0)
    test_rec = recall_score(y_test, y_pred_test)
    test_auc = roc_auc_score(y_test, y_proba_test)
    test_pr_auc = average_precision_score(y_test, y_proba_test)

    print(f"\n  {name} (threshold={best_t:.2f}):")
    print(f"    CV:   F1={cv_f1:.4f}  Prec={cv_prec:.4f}  Rec={cv_rec:.4f}  AUC={cv_auc:.4f}  PR-AUC={cv_pr_auc:.4f}")
    print(f"    Test: F1={test_f1:.4f}  Prec={test_prec:.4f}  Rec={test_rec:.4f}  AUC={test_auc:.4f}  PR-AUC={test_pr_auc:.4f}")

    if cv_f1 > best_overall["f1"]:
        best_overall = {
            "name": name, "f1": cv_f1, "threshold": best_t, "model": model,
            "cv": {"f1": cv_f1, "precision": cv_prec, "recall": cv_rec,
                   "auc": cv_auc, "pr_auc": cv_pr_auc},
            "test": {"f1": test_f1, "precision": test_prec, "recall": test_rec,
                     "auc": test_auc, "pr_auc": test_pr_auc},
        }

# =====================================================================
# 5. Save final model and report data
# =====================================================================
print(f"\n[5/5] Best model: {best_overall['name']} (CV-F1={best_overall['f1']:.4f})")

final_model = best_overall["model"]
final_threshold = best_overall["threshold"]

# Save model
joblib.dump(final_model, OUTPUT_DIR / "final_model.joblib")

# Feature importance
importances = pd.Series(final_model.feature_importances_, index=X_train_s.columns)
top10 = importances.nlargest(10)

# Operating point
op = {
    "model_name": best_overall["name"],
    "optimal_threshold": float(final_threshold),
    "n_features": int(X_train_s.shape[1]),
    "features": list(X_train_s.columns),
    "cv_metrics": {k: round(float(v), 4) for k, v in best_overall["cv"].items()},
    "test_metrics": {k: round(float(v), 4) for k, v in best_overall["test"].items()},
    "top10_feature_importance": {f: round(float(v), 4) for f, v in top10.items()},
    "best_params": {str(k): str(v) for k, v in final_model.get_params().items()
                    if k not in ["n_jobs", "verbose", "verbosity"]},
}
with open(OUTPUT_DIR / "operating_point.json", "w") as f:
    json.dump(op, f, indent=2)

# Final classification report
y_final = (final_model.predict_proba(X_test_s)[:, 1] >= final_threshold).astype(int)
print(f"\nFINAL CLASSIFICATION REPORT ({best_overall['name']} @ threshold={final_threshold:.2f})")
print("=" * 65)
print(classification_report(y_test, y_final, target_names=["Stay", "Leave"]))

cm = confusion_matrix(y_test, y_final)
print(f"Confusion Matrix:")
print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
print(f"  FN={cm[1,0]}  TP={cm[1,1]}")

print(f"\nTop 10 feature importance:")
for i, (feat, imp) in enumerate(top10.items(), 1):
    print(f"  {i:2d}. {feat:45s} {imp:.4f}")

print(f"\nAll outputs saved to {OUTPUT_DIR}/")
print("Done.")
