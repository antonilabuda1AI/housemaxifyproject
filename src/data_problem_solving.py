# solve_regression_pipeline.py
import os
import math
import numpy as np
import pandas as pd
from typing import List, Tuple

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance

# --------------------------- Config ---------------------------
DATA_PATH = "../data/kc_house_data.csv"
CLEAN_PATH = "../data/kc_house_data_clean.csv"  # where cleaned data will be saved
TARGET = "price"
ID_COLS = ["id"]                   # will be dropped
DATE_COL = "date"                  # will be parsed
ZIP_AS_CATEGORICAL = True          # zipcode treated as categorical
RANDOM_STATE = 42
TEST_SIZE = 0.2                    # 80/20 split

# --------------------- Helpers / Transformers -----------------
def header(title: str):
    line = "=" * 90
    print(f"\n{line}\n{title}\n{line}")

def parse_date_column(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df["sale_year"] = df[date_col].dt.year
        df["sale_month"] = df[date_col].dt.month
        df = df.drop(columns=[date_col])
    return df

def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    if "yr_built" in df.columns and "sale_year" in df.columns:
        df["age_at_sale"] = df["sale_year"] - df["yr_built"]
    if "yr_renovated" in df.columns and "sale_year" in df.columns:
        yrs_since_renov = df["sale_year"] - df["yr_renovated"].replace(0, np.nan)
        df["yrs_since_renov"] = yrs_since_renov.where(~yrs_since_renov.isna(), 0)
    if ZIP_AS_CATEGORICAL and "zipcode" in df.columns:
        df["zipcode"] = df["zipcode"].astype("Int64").astype("string")
    return df

def get_feature_lists(df: pd.DataFrame, target: str) -> Tuple[List[str], List[str]]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    for col in [target] + ID_COLS:
        if col in numeric_cols:
            numeric_cols.remove(col)
        if col in categorical_cols:
            categorical_cols.remove(col)
    return numeric_cols, categorical_cols

def make_quantile_clipper(lower=0.01, upper=0.99):
    def _clipper(X):
        if isinstance(X, pd.DataFrame):
            Xc = X.copy()
            q_low = Xc.quantile(lower)
            q_hi = Xc.quantile(upper)
            return Xc.clip(lower=q_low, upper=q_hi, axis=1)
        else:
            return X
    return FunctionTransformer(_clipper, validate=False)

def rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))

def print_metrics(y_true, y_pred, label=""):
    print(f"{label}RMSE: {rmse(y_true, y_pred):,.2f}")
    print(f"{label}MAE : {mean_absolute_error(y_true, y_pred):,.2f}")
    print(f"{label}R²  : {r2_score(y_true, y_pred):.4f}")

# ----------------------------- Main ---------------------------
def main():
    header("LOAD & BASIC PREP")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"CSV not found at: {os.path.abspath(DATA_PATH)}")
    df = pd.read_csv(DATA_PATH)
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    if TARGET not in df.columns:
        raise ValueError(f"Target '{TARGET}' not found in CSV.")

    # Parse date -> year/month; drop raw date; drop id-like
    df = parse_date_column(df, DATE_COL)
    df = df.drop(columns=[c for c in ID_COLS if c in df.columns], errors="ignore")

    # Domain features
    df = add_domain_features(df)

    # -------------------- SAVE CLEAN DATA --------------------
    header("SAVE CLEANED DATASET")
    df.to_csv(CLEAN_PATH, index=False)
    print(f"✅ Cleaned dataset saved to: {CLEAN_PATH}")
    print(f"Rows: {len(df):,} | Columns: {len(df.columns)}")

    header("MISSING VALUES")
    na = df.isna().sum().sort_values(ascending=False)
    print(na[na > 0] if (na > 0).any() else "No missing values.")

    header("DEFINE FEATURES")
    numeric_cols, categorical_cols = get_feature_lists(df, TARGET)
    print("Numeric features:", numeric_cols)
    print("Categorical features:", categorical_cols)

    # Target transform: log1p for stability
    y = df[TARGET].values
    y_log = np.log1p(y)
    X = df.drop(columns=[TARGET])

    X_train, X_valid, y_train_log, y_valid_log, y_train, y_valid = train_test_split(
        X, y_log, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # -------------------- Preprocessing --------------------
    numeric_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("clip", make_quantile_clipper(0.01, 0.99)),
        ("scale", StandardScaler()),
    ])

    categorical_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=0.01)),
    ])

    preproc = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop"
    )

    # -------------------- Models --------------------
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=400, min_samples_leaf=2, n_jobs=-1, random_state=RANDOM_STATE
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=20, random_state=RANDOM_STATE
        ),
    }

    results = {}
    header("CROSS-VALIDATION (5-fold)")
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    for name, model in models.items():
        pipe = Pipeline(steps=[("pre", preproc), ("model", model)])
        scores = cross_val_score(pipe, X_train, y_train_log,
                                 scoring="neg_root_mean_squared_error", cv=cv)
        print(f"{name}: mean={-scores.mean():.4f} | std={scores.std():.4f}")
        results[name] = -scores.mean()

    # -------------------- Evaluate best --------------------
    header("HOLDOUT VALIDATION")
    best_name = min(results, key=results.get)
    print(f"Best CV model: {best_name}")
    best_model = models[best_name]
    best_pipe = Pipeline(steps=[("pre", preproc), ("model", best_model)])
    best_pipe.fit(X_train, y_train_log)
    y_pred = np.expm1(best_pipe.predict(X_valid))
    print_metrics(y_valid, y_pred, label=f"{best_name} | ")

    # -------------------- Save Summary --------------------
    header("SUMMARY")
    print(f"Clean dataset: {CLEAN_PATH}")
    print("- Features cleaned and saved")
    print("- Model trained and evaluated successfully")

if __name__ == "__main__":
    main()
