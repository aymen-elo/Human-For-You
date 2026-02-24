# ==============================================================================
# ABLATION TEST PIPELINE — Full re-processing + 5 scenarios
# ==============================================================================
import pandas as pd
import numpy as np
import warnings
import time
import json
import os
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    average_precision_score, classification_report
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

print("=" * 70)
print("ABLATION TEST PIPELINE")
print("=" * 70)

# -- Step 1: Load raw data ------------------------------------------------
print("\n[1/6] Loading raw data...")
general = pd.read_csv(DATA_DIR / "general_data.csv")
emp_survey = pd.read_csv(DATA_DIR / "employee_survey_data.csv")
mgr_survey = pd.read_csv(DATA_DIR / "manager_survey_data.csv")
in_time = pd.read_csv(DATA_DIR / "in_time.csv")
out_time = pd.read_csv(DATA_DIR / "out_time.csv")


def process_badge(df_in, df_out, cutoff=None):
    """Process badge data with optional date cutoff."""
    emp_ids = df_in.iloc[:, 0].astype(int)
    in_dates = df_in.iloc[:, 1:]
    out_dates = df_out.iloc[:, 1:]

    if cutoff is not None:
        date_cols = []
        for c in in_dates.columns:
            try:
                dt = pd.to_datetime(c, errors="coerce")
                if dt is not pd.NaT and dt <= cutoff:
                    date_cols.append(c)
            except Exception:
                pass
        in_dates = in_dates[date_cols]
        out_dates = out_dates[date_cols]
        print(f"  Badge: {len(date_cols)} days (up to {cutoff.date()})")
    else:
        print(f"  Badge: all {in_dates.shape[1]} days")

    in_parsed = in_dates.apply(pd.to_datetime, errors="coerce")
    out_parsed = out_dates.apply(pd.to_datetime, errors="coerce")

    in_hours = in_parsed.apply(lambda c: c.dt.hour + c.dt.minute / 60)
    out_hours = out_parsed.apply(lambda c: c.dt.hour + c.dt.minute / 60)

    work_hours = out_hours - in_hours
    n_days = in_hours.shape[1]
    valid = in_hours.notna() & out_hours.notna()
    n_present = valid.sum(axis=1)
    n_absent = n_days - n_present
    late = valid & (in_hours >= 10)
    n_late = late.sum(axis=1)

    return pd.DataFrame({
        "EmployeeID": emp_ids.values,
        "avg_working_hours": work_hours.where(valid).mean(axis=1),
        "absence_rate": n_absent / n_days,
        "late_arrival_rate": (n_late / n_present).where(n_present > 0),
    })


# Process badge with two configs
print("\n[2/6] Processing badge data...")
badge_h1 = process_badge(in_time, out_time, cutoff=pd.Timestamp("2015-06-30"))
badge_full = process_badge(in_time, out_time, cutoff=None)


# -- Step 2: Build merged dataset -----------------------------------------
def build_merged(general_df, emp_survey_df, mgr_survey_df, badge_df):
    """Build the merged dataset with a given badge feature set."""
    df = general_df.copy()
    df = df.merge(emp_survey_df, on="EmployeeID", how="left")
    df = df.merge(mgr_survey_df, on="EmployeeID", how="left")
    if badge_df is not None:
        df = df.merge(badge_df, on="EmployeeID", how="left")

    # Drop constants
    df = df.drop(columns=["EmployeeCount", "Over18", "StandardHours"], errors="ignore")

    # Target
    df["Attrition"] = (df["Attrition"] == "Yes").astype(int)

    # Feature engineering (same as NB03 minus LongHours)
    if "MonthlyIncome" in df.columns and "JobLevel" in df.columns:
        df["IncomePerJobLevel"] = df["MonthlyIncome"] / df["JobLevel"]
    if "YearsSinceLastPromotion" in df.columns and "YearsAtCompany" in df.columns:
        df["PromotionStagnation"] = df["YearsSinceLastPromotion"] / (df["YearsAtCompany"] + 1)
    survey_items = [c for c in ["EnvironmentSatisfaction", "JobSatisfaction", "WorkLifeBalance"]
                    if c in df.columns]
    if survey_items:
        df["SatisfactionScore"] = df[survey_items].mean(axis=1)
    if "YearsWithCurrManager" in df.columns and "YearsAtCompany" in df.columns:
        df["ManagerStability"] = df["YearsWithCurrManager"] / (df["YearsAtCompany"] + 1)

    # Encode categoricals
    bt_map = {"Non-Travel": 0, "Travel_Rarely": 1, "Travel_Frequently": 2}
    if "BusinessTravel" in df.columns:
        df["BusinessTravel"] = df["BusinessTravel"].map(bt_map)
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)

    return df


# -- Step 3: Preprocess pipeline ------------------------------------------
def preprocess(df, feature_subset=None):
    """Split, impute, filter, scale."""
    if "EmployeeID" in df.columns:
        df = df.drop(columns=["EmployeeID"])

    X = df.drop(columns=["Attrition"])
    y = df["Attrition"]

    if feature_subset is not None:
        available = [c for c in feature_subset if c in X.columns]
        X = X[available]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train = X_train.copy()
    X_test = X_test.copy()

    # KNN imputation
    cols_na = X_train.columns[X_train.isnull().any()].tolist()
    if cols_na:
        imp = KNNImputer(n_neighbors=5, weights="distance")
        X_train[cols_na] = imp.fit_transform(X_train[cols_na])
        X_test[cols_na] = imp.transform(X_test[cols_na])

    # Variance filter
    vt = VarianceThreshold(threshold=0.01)
    vt.fit(X_train)
    low_var = X_train.columns[~vt.get_support()].tolist()
    if low_var:
        X_train = X_train.drop(columns=low_var)
        X_test = X_test.drop(columns=low_var)

    # Correlation filter (0.90)
    corr = X_train.corr().abs()
    tri = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    high_corr = [c for c in tri.columns if any(tri[c] > 0.90)]
    if high_corr:
        X_train = X_train.drop(columns=high_corr)
        X_test = X_test.drop(columns=high_corr)

    # Scale
    sc = StandardScaler()
    X_train_s = pd.DataFrame(sc.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_s = pd.DataFrame(sc.transform(X_test), columns=X_test.columns, index=X_test.index)

    return X_train_s, X_test_s, y_train, y_test


# -- Step 4: Evaluation function ------------------------------------------
def evaluate_scenario(name, X_train, X_test, y_train, y_test):
    """Run RF with class_weight=balanced, 5-fold CV + test evaluation."""
    t0 = time.time()

    model = RandomForestClassifier(
        n_estimators=300, class_weight="balanced",
        random_state=42, n_jobs=-1
    )

    # 5-fold CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_proba_cv = cross_val_predict(model, X_train, y_train, cv=cv,
                                    method="predict_proba", n_jobs=-1)[:, 1]

    # Find optimal threshold via CV
    best_t, best_f1_cv = 0.5, 0
    for t in np.arange(0.10, 0.91, 0.01):
        y_cv = (y_proba_cv >= t).astype(int)
        f1_cv = f1_score(y_train, y_cv, zero_division=0)
        rec_cv = recall_score(y_train, y_cv, zero_division=0)
        if f1_cv > best_f1_cv and rec_cv >= 0.60:
            best_f1_cv = f1_cv
            best_t = t

    # CV metrics at optimal threshold
    y_cv_pred = (y_proba_cv >= best_t).astype(int)
    cv_f1 = f1_score(y_train, y_cv_pred)
    cv_precision = precision_score(y_train, y_cv_pred)
    cv_recall = recall_score(y_train, y_cv_pred)
    cv_auc = roc_auc_score(y_train, y_proba_cv)
    cv_pr_auc = average_precision_score(y_train, y_proba_cv)

    # Train final model and evaluate on test
    model.fit(X_train, y_train)
    y_proba_test = model.predict_proba(X_test)[:, 1]
    y_pred_test = (y_proba_test >= best_t).astype(int)

    test_f1 = f1_score(y_test, y_pred_test)
    test_precision = precision_score(y_test, y_pred_test, zero_division=0)
    test_recall = recall_score(y_test, y_pred_test)
    test_auc = roc_auc_score(y_test, y_proba_test)
    test_pr_auc = average_precision_score(y_test, y_proba_test)

    elapsed = time.time() - t0

    # Feature importance
    importances = pd.Series(model.feature_importances_, index=X_train.columns)
    top10 = importances.nlargest(10)

    result = {
        "scenario": name,
        "n_features": int(X_train.shape[1]),
        "threshold": round(float(best_t), 2),
        "cv_f1": round(float(cv_f1), 4),
        "cv_precision": round(float(cv_precision), 4),
        "cv_recall": round(float(cv_recall), 4),
        "cv_auc": round(float(cv_auc), 4),
        "cv_pr_auc": round(float(cv_pr_auc), 4),
        "test_f1": round(float(test_f1), 4),
        "test_precision": round(float(test_precision), 4),
        "test_recall": round(float(test_recall), 4),
        "test_auc": round(float(test_auc), 4),
        "test_pr_auc": round(float(test_pr_auc), 4),
        "runtime_s": round(elapsed, 1),
        "top10_features": {f: round(float(v), 4) for f, v in top10.items()},
        "features": list(X_train.columns),
    }

    return result


# -- Step 5: Define and run scenarios -------------------------------------

# Top 25 features (by |r| from full audit + manual curation)
TOP25 = [
    "avg_working_hours", "MaritalStatus_Single", "TotalWorkingYears",
    "Age", "YearsWithCurrManager", "SatisfactionScore", "ManagerStability",
    "YearsAtCompany", "BusinessTravel", "JobSatisfaction",
    "MaritalStatus_Married", "EnvironmentSatisfaction", "WorkLifeBalance",
    "TrainingTimesLastYear", "YearsSinceLastPromotion",
    "JobRole_Manufacturing Director", "NumCompaniesWorked",
    "absence_rate", "Gender_Male", "late_arrival_rate",
    "EducationField_Technical Degree", "JobRole_Sales Executive",
    "JobRole_Research Director", "Department_Sales", "StockOptionLevel",
]

# Minimal 15 features (HR-only, no badge, no OHE except MaritalStatus)
MINIMAL15 = [
    "TotalWorkingYears", "Age", "YearsWithCurrManager",
    "SatisfactionScore", "ManagerStability", "YearsAtCompany",
    "BusinessTravel", "JobSatisfaction", "MaritalStatus_Single",
    "MaritalStatus_Married", "EnvironmentSatisfaction",
    "WorkLifeBalance", "TrainingTimesLastYear",
    "YearsSinceLastPromotion", "NumCompaniesWorked",
]

scenarios = {}

# -- S0: Baseline corrected (badge H1, no redundancies) ---
print("\n[3/6] Running S0: Baseline corrected (badge H1, no redundancies)...")
df_s0 = build_merged(general, emp_survey, mgr_survey, badge_h1)
X_tr_s0, X_te_s0, y_tr_s0, y_te_s0 = preprocess(df_s0)
scenarios["S0"] = evaluate_scenario(
    "S0: Baseline corrected (H1 badge)", X_tr_s0, X_te_s0, y_tr_s0, y_te_s0)
print(f"  Features: {scenarios['S0']['n_features']}  "
      f"CV-F1: {scenarios['S0']['cv_f1']}  Test-F1: {scenarios['S0']['test_f1']}")

# -- S1: No badge features at all ---
print("\n[4/6] Running S1: No badge features...")
df_s1 = build_merged(general, emp_survey, mgr_survey, None)
X_tr_s1, X_te_s1, y_tr_s1, y_te_s1 = preprocess(df_s1)
scenarios["S1"] = evaluate_scenario(
    "S1: No badge features", X_tr_s1, X_te_s1, y_tr_s1, y_te_s1)
print(f"  Features: {scenarios['S1']['n_features']}  "
      f"CV-F1: {scenarios['S1']['cv_f1']}  Test-F1: {scenarios['S1']['test_f1']}")

# -- S2: Badge full year (original, for comparison) ---
print("\n[5/6] Running S2: Badge full year 2015...")
df_s2 = build_merged(general, emp_survey, mgr_survey, badge_full)
X_tr_s2, X_te_s2, y_tr_s2, y_te_s2 = preprocess(df_s2)
scenarios["S2"] = evaluate_scenario(
    "S2: Badge full year 2015", X_tr_s2, X_te_s2, y_tr_s2, y_te_s2)
print(f"  Features: {scenarios['S2']['n_features']}  "
      f"CV-F1: {scenarios['S2']['cv_f1']}  Test-F1: {scenarios['S2']['test_f1']}")

# -- S3: Top 25 features ---
print("\n       Running S3: Top 25 features...")
df_s3 = build_merged(general, emp_survey, mgr_survey, badge_h1)
X_tr_s3, X_te_s3, y_tr_s3, y_te_s3 = preprocess(df_s3, feature_subset=TOP25)
scenarios["S3"] = evaluate_scenario(
    "S3: Top 25 features", X_tr_s3, X_te_s3, y_tr_s3, y_te_s3)
print(f"  Features: {scenarios['S3']['n_features']}  "
      f"CV-F1: {scenarios['S3']['cv_f1']}  Test-F1: {scenarios['S3']['test_f1']}")

# -- S4: Minimal 15 features (HR only) ---
print("\n[6/6] Running S4: Minimal 15 (HR only)...")
df_s4 = build_merged(general, emp_survey, mgr_survey, None)
X_tr_s4, X_te_s4, y_tr_s4, y_te_s4 = preprocess(df_s4, feature_subset=MINIMAL15)
scenarios["S4"] = evaluate_scenario(
    "S4: Minimal 15 (HR only)", X_tr_s4, X_te_s4, y_tr_s4, y_te_s4)
print(f"  Features: {scenarios['S4']['n_features']}  "
      f"CV-F1: {scenarios['S4']['cv_f1']}  Test-F1: {scenarios['S4']['test_f1']}")

# -- Step 6: Summary ------------------------------------------------------
print("\n" + "=" * 70)
print("ABLATION TEST RESULTS SUMMARY")
print("=" * 70)
header = (f"{'Scenario':<42s} {'Feat':>4s} {'Thr':>5s} {'CV-F1':>7s} "
          f"{'T-F1':>7s} {'T-Prec':>7s} {'T-Rec':>7s} {'T-AUC':>7s} "
          f"{'T-PRAUC':>8s} {'Time':>6s}")
print(header)
print("-" * len(header))

for k, r in scenarios.items():
    print(f"{r['scenario']:<42s} {r['n_features']:>4d} {r['threshold']:>5.2f} "
          f"{r['cv_f1']:>7.4f} {r['test_f1']:>7.4f} {r['test_precision']:>7.4f} "
          f"{r['test_recall']:>7.4f} {r['test_auc']:>7.4f} {r['test_pr_auc']:>8.4f} "
          f"{r['runtime_s']:>5.1f}s")

# Badge contribution
print(f"\n--- Badge contribution analysis ---")
print(f"S0 (H1 badge) vs S1 (no badge):  Delta CV-F1 = "
      f"{scenarios['S0']['cv_f1'] - scenarios['S1']['cv_f1']:+.4f}")
print(f"S2 (full badge) vs S1 (no badge): Delta CV-F1 = "
      f"{scenarios['S2']['cv_f1'] - scenarios['S1']['cv_f1']:+.4f}")
print(f"S2 (full badge) vs S0 (H1 badge): Delta CV-F1 = "
      f"{scenarios['S2']['cv_f1'] - scenarios['S0']['cv_f1']:+.4f}")
print(f"S3 (top 25) vs S0 (baseline):     Delta CV-F1 = "
      f"{scenarios['S3']['cv_f1'] - scenarios['S0']['cv_f1']:+.4f}")

# Top features per scenario
print("\n--- Top 5 features per scenario ---")
for k, r in scenarios.items():
    top5 = list(r["top10_features"].items())[:5]
    print(f"\n{r['scenario']}:")
    for feat, imp in top5:
        print(f"    {feat:<40s} {imp:.4f}")

# Save results
with open(OUTPUT_DIR / "ablation_results.json", "w") as f:
    json.dump(scenarios, f, indent=2, default=str)
print(f"\nResults saved to outputs/ablation_results.json")

# Save CSV summary
rows = []
for k, r in scenarios.items():
    rows.append({
        "Scenario": r["scenario"], "Features": r["n_features"],
        "Threshold": r["threshold"],
        "CV_F1": r["cv_f1"], "CV_Precision": r["cv_precision"],
        "CV_Recall": r["cv_recall"], "CV_AUC": r["cv_auc"],
        "CV_PR_AUC": r["cv_pr_auc"],
        "Test_F1": r["test_f1"], "Test_Precision": r["test_precision"],
        "Test_Recall": r["test_recall"], "Test_AUC": r["test_auc"],
        "Test_PR_AUC": r["test_pr_auc"], "Runtime_s": r["runtime_s"],
    })
pd.DataFrame(rows).to_csv(OUTPUT_DIR / "ablation_results.csv", index=False)
print("CSV saved to outputs/ablation_results.csv")
print("\nDone.")
