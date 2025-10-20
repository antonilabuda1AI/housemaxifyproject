import pandas as pd

# Path to your dataset
file_path = "../data/kc_house_data.csv"

# Read the CSV file
df = pd.read_csv(file_path)

# Show the first 5 rows
print(df.head())
# validate_regression.py
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

DATA_PATH = "../data/kc_house_data.csv"
TARGET = "price"        # <-- change if your target is different
DROP_COLS = ["id"]      # identifiers you don't want as features (add more if needed)
PARSE_DATES = ["date"]  # columns to try parsing as datetime (optional)

def header(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

def main():
    # -------------------- Load --------------------
    header("LOAD DATA")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"CSV not found at: {os.path.abspath(DATA_PATH)}")
    df = pd.read_csv(DATA_PATH)
    print(f"Rows: {len(df):,} | Columns: {df.shape[1]}")
    print("Columns:", ", ".join(df.columns))

    # -------------------- Target checks --------------------
    header("TARGET CHECKS")
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found. "
                         f"Available columns: {list(df.columns)}")
    tgt = df[TARGET]
    print(f"Target '{TARGET}': dtype={tgt.dtype}, missing={tgt.isna().sum()}, "
          f"unique={tgt.nunique():,}, min={tgt.min()}, max={tgt.max()}")
    if tgt.isna().any():
        print("WARNING: Target has missing values. Drop or impute before training.")
    if tgt.nunique() < 2:
        raise ValueError("Target has fewer than 2 unique values — regression not possible.")

    # -------------------- Basic sanitation --------------------
    header("BASIC SANITY CHECKS")
    # Parse dates (optional)
    for c in PARSE_DATES:
        if c in df.columns:
            try:
                df[c] = pd.to_datetime(df[c], errors="coerce")
                print(f"Parsed '{c}' to datetime (coerce errors to NaT).")
            except Exception as e:
                print(f"Failed to parse '{c}' as datetime: {e}")

    # Duplicates
    dup_rows = df.duplicated().sum()
    print(f"Duplicate rows: {dup_rows}")
    if "id" in df.columns:
        dup_ids = df["id"].duplicated().sum()
        print(f"Duplicate 'id' values: {dup_ids}")
        if dup_ids > 0:
            print("WARNING: Duplicate IDs found.")

    # Missing values
    na_counts = df.isna().sum().sort_values(ascending=False)
    if na_counts.any():
        print("\nMissing values per column (nonzero only):")
        print(na_counts[na_counts > 0])
    else:
        print("No missing values in any column.")

    # -------------------- Feature set (numeric-only for quick checks) --------------------
    header("FEATURE SET (NUMERIC ONLY FOR QUICK VALIDATION)")
    features = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")
    X_num = features.select_dtypes(include=[np.number]).drop(columns=[TARGET], errors="ignore")
    print(f"Numeric features count: {X_num.shape[1]}")
    print("Numeric feature sample:", ", ".join(X_num.columns[:10]))

    # Non-numeric columns (for you to encode later if you want to use them)
    non_numeric = features.columns.difference(X_num.columns.tolist() + [TARGET])
    if len(non_numeric) > 0:
        print("\nNon-numeric columns (need encoding if you plan to use them):")
        print(", ".join(non_numeric))
    else:
        print("\nNo non-numeric feature columns.")

    # Zero / near-zero variance
    header("ZERO / NEAR-ZERO VARIANCE")
    variances = X_num.var(numeric_only=True)
    zero_var = variances[variances == 0.0].index.tolist()
    low_var = variances[variances > 0.0][variances[variances > 0.0] < 1e-6].index.tolist()
    print(f"Zero-variance features: {len(zero_var)}")
    if zero_var:
        print(", ".join(zero_var))
    print(f"Near-zero variance features (<1e-6): {len(low_var)}")
    if low_var:
        print(", ".join(low_var))

    # -------------------- Outliers (IQR rule) --------------------
    header("OUTLIER SCREEN (IQR RULE)")
    def iqr_outlier_fraction(s: pd.Series):
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr == 0:
            return 0.0
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        return ((s < lower) | (s > upper)).mean()

    tgt_out_frac = iqr_outlier_fraction(tgt)
    print(f"Target '{TARGET}' outlier fraction (IQR rule): {tgt_out_frac:.3f}")
    # Show top 10 most outlier-prone numeric features
    outlier_fracs = X_num.apply(iqr_outlier_fraction).sort_values(ascending=False)
    print("\nTop 10 features by outlier fraction:")
    print(outlier_fracs.head(10))

    # -------------------- Multicollinearity proxy (high correlations) --------------------
    header("HIGH CORRELATION PAIRS (|corr| >= 0.95)")
    corr = X_num.corr(numeric_only=True).abs()
    high_pairs = []
    cols = corr.columns
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            if corr.iat[i, j] >= 0.95:
                high_pairs.append((cols[i], cols[j], corr.iat[i, j]))
    if high_pairs:
        for a, b, c in sorted(high_pairs, key=lambda x: -x[2])[:20]:
            print(f"{a} ~ {b}: {c:.3f}")
        print("NOTE: Consider dropping/combining one of each highly correlated pair.")
    else:
        print("No feature pairs with |corr| ≥ 0.95 found.")

    # -------------------- Quick learnability check --------------------
    header("QUICK LEARNABILITY CHECK (5-fold CV, numeric-only)")
    # Baseline RMSE: predict mean of target
    baseline_rmse = np.sqrt(((tgt - tgt.mean()) ** 2).mean())
    print(f"Baseline RMSE (predict mean): {baseline_rmse:,.2f}")

    # Simple numeric pipeline
    pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", LinearRegression())
    ])

    # Keep only rows where target is not missing
    valid_idx = ~tgt.isna()
    X_use = X_num.loc[valid_idx]
    y_use = tgt.loc[valid_idx]

    if X_use.shape[0] < 10 or X_use.shape[1] == 0:
        print("Not enough numeric data to run CV.")
    else:
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        # negative RMSE -> take abs
        scores = cross_val_score(pipe, X_use, y_use,
                                 scoring="neg_root_mean_squared_error",
                                 cv=cv, n_jobs=None)
        rmse_scores = -scores
        print(f"LinearRegression RMSE (5-fold): mean={rmse_scores.mean():,.2f} "
              f"| std={rmse_scores.std():,.2f}")
        if rmse_scores.mean() < baseline_rmse:
            print("✅ Model beats baseline — data looks learnable (at least with numeric features).")
        else:
            print("⚠️ Model DOES NOT beat baseline — investigate features/leakage/encoding.")

    # -------------------- Summary --------------------
    header("SUMMARY / NEXT STEPS")
    print("- Handle any missing values (especially in target).")
    print("- Drop identifiers (e.g., 'id') from features.")
    print("- Encode non-numeric columns if you plan to use them (One-Hot for small-cardinality).")
    print("- Consider removing zero/near-zero variance and highly correlated features.")
    print("- Review outliers: cap/transform (log), or robust models if necessary.")
    print("- Consider feature engineering from 'date' (e.g., year, month) instead of raw string.")
    print("- After cleaning, rerun this check and try a stronger model (e.g., RandomForest, XGBoost).")

if __name__ == "__main__":
    main()
