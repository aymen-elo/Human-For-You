import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# =====================================================================
# 1. Rebuild corrected dataset (badge H1, no LongHours)
# =====================================================================
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
late = valid & (in_h >= 10)
n_late = late.sum(axis=1)
badge = pd.DataFrame({
    "EmployeeID": emp_ids.values,
    "avg_working_hours": wh.where(valid).mean(axis=1),
    "absence_rate": n_absent / n_days,
    "late_arrival_rate": (n_late / n_present).where(n_present > 0),
})

# Merge
df = general.copy()
df = df.merge(emp, on="EmployeeID").merge(mgr, on="EmployeeID").merge(badge, on="EmployeeID")
df = df.drop(columns=["EmployeeCount", "Over18", "StandardHours"])
df["Attrition"] = (df["Attrition"] == "Yes").astype(int)

# Feature engineering (no LongHours)
df["IncomePerJobLevel"] = df["MonthlyIncome"] / df["JobLevel"]
df["PromotionStagnation"] = df["YearsSinceLastPromotion"] / (df["YearsAtCompany"] + 1)
df["SatisfactionScore"] = df[["EnvironmentSatisfaction", "JobSatisfaction", "WorkLifeBalance"]].mean(axis=1)
df["ManagerStability"] = df["YearsWithCurrManager"] / (df["YearsAtCompany"] + 1)
df["BusinessTravel"] = df["BusinessTravel"].map({"Non-Travel": 0, "Travel_Rarely": 1, "Travel_Frequently": 2})
cat_cols = df.select_dtypes(include="object").columns.tolist()
df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)

df = df.drop(columns=["EmployeeID"])
X = df.drop(columns=["Attrition"])
y = df["Attrition"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train = X_train.copy()
X_test = X_test.copy()

# Impute
cols_na = X_train.columns[X_train.isnull().any()].tolist()
if cols_na:
    imp = KNNImputer(n_neighbors=5, weights="distance")
    X_train[cols_na] = imp.fit_transform(X_train[cols_na])
    X_test[cols_na] = imp.transform(X_test[cols_na])

# =====================================================================
# 2. Hybrid feature selection
# =====================================================================
print("=" * 70)
print("HYBRID FEATURE SELECTION")
print("=" * 70)

# Correlation with target (unscaled)
corr_target = X_train.corrwith(y_train).abs()

# Gini importance
sc_temp = StandardScaler()
X_tr_s = pd.DataFrame(sc_temp.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
rf_temp = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1)
rf_temp.fit(X_tr_s, y_train)
gini_imp = pd.Series(rf_temp.feature_importances_, index=X_train.columns)

# Build audit table
audit = pd.DataFrame({
    "feature": X_train.columns,
    "abs_r": corr_target.values,
    "gini": gini_imp.values,
}).sort_values("gini", ascending=False).reset_index(drop=True)

# Origin
def get_origin(f):
    if f in ["avg_working_hours", "absence_rate", "late_arrival_rate"]:
        return "Badge H1"
    elif f in ["IncomePerJobLevel", "PromotionStagnation", "SatisfactionScore", "ManagerStability"]:
        return "Derived"
    elif f.startswith(("Department_", "EducationField_", "Gender_", "JobRole_", "MaritalStatus_")):
        return "HR OHE"
    elif f in ["EnvironmentSatisfaction", "JobSatisfaction", "WorkLifeBalance"]:
        return "Survey emp"
    elif f in ["JobInvolvement", "PerformanceRating"]:
        return "Survey mgr"
    else:
        return "HR"

audit["origin"] = audit["feature"].apply(get_origin)
audit["cum_gini"] = audit["gini"].cumsum()
audit["cum_pct"] = audit["cum_gini"] / audit["gini"].sum()

# Selection: |r| >= 0.03 OR top 82% Gini
keep_corr = set(audit[audit["abs_r"] >= 0.03]["feature"])
keep_gini = set(audit[audit["cum_pct"] <= 0.82]["feature"])
keep = keep_corr | keep_gini

# Multicollinearity filter
selected_list = [f for f in audit["feature"] if f in keep]
corr_matrix = X_train[selected_list].corr().abs()
tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = set()
for col in tri.columns:
    partners = tri.index[tri[col] > 0.90].tolist()
    for p in partners:
        if gini_imp[col] >= gini_imp[p]:
            to_drop.add(p)
        else:
            to_drop.add(col)

keep = keep - to_drop
final_features = [f for f in audit["feature"] if f in keep]

print(f"\nSelection results:")
print(f"  Corr filter (|r|>=0.03): {len(keep_corr)} features")
print(f"  Gini filter (top 82%):   {len(keep_gini)} features")
print(f"  Union:                   {len(keep_corr | keep_gini)} features")
print(f"  Multicollinear removed:  {len(to_drop)} {list(to_drop) if to_drop else '(none)'}")
print(f"  FINAL:                   {len(final_features)} features")

# Print full table
print(f"\n{'Rank':>4s} {'Feature':45s} {'Origin':12s} {'|r|':>6s} {'Gini':>6s} {'Cum%':>6s} {'Keep':>5s} Reason")
print("-" * 120)
for i, (_, row) in enumerate(audit.iterrows(), 1):
    f = row["feature"]
    in_final = f in final_features
    if f in to_drop:
        reason = "Multicollinear (r>0.90)"
    elif row["abs_r"] < 0.03 and f not in keep_gini:
        reason = "Noise (|r|<0.03, low Gini)"
    elif in_final and row["abs_r"] >= 0.10:
        reason = "Strong signal"
    elif in_final and row["abs_r"] >= 0.03:
        reason = "Moderate signal"
    elif in_final:
        reason = "High Gini importance"
    else:
        reason = "Below thresholds"
    mark = "YES" if in_final else "no"
    print(f"{i:4d} {f:45s} {row['origin']:12s} {row['abs_r']:6.4f} {row['gini']:6.4f} {row['cum_pct']:6.1%} {mark:>5s}  {reason}")

# Save
audit["keep"] = audit["feature"].isin(final_features)
audit.to_csv(OUTPUT_DIR / "feature_selection_final.csv", index=False)

print(f"\nFinal {len(final_features)} features:")
for i, f in enumerate(final_features, 1):
    print(f"  {i:2d}. {f}")

# Write feature list for downstream use
pd.Series(final_features).to_csv(OUTPUT_DIR / "final_feature_list.csv", index=False, header=False)
print(f"\nSaved to outputs/feature_selection_final.csv")
print(f"Saved to outputs/final_feature_list.csv")
print(f"\nFINAL_FEATURES = {final_features}")
