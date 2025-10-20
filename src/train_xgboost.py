# train_xgboost.py
import os
import math
import joblib
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from xgboost import XGBRegressor

# ======================= Config =======================
CLEAN_DATA_PATH = "../data/kc_house_data_clean.csv"
MODEL_DIR = "../models"
MODEL_PATH = os.path.join(MODEL_DIR, "house_price_xgboost.joblib")
TARGET = "price"
RANDOM_STATE = 42
TEST_SIZE = 0.2  # 80/20 split

# ======================= Helpers ======================
def header(title: str):
    line = "=" * 90
    print(f"\n{line}\n{title}\n{line}")

def rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))

# ---- Picklable winsorizer (clips to per-column quantiles) ----
class Winsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, lower=0.01, upper=0.99):
        self.lower = float(lower)
        self.upper = float(upper)

    def fit(self, X, y=None):
        # Works with DataFrame or ndarray
        if isinstance(X, pd.DataFrame):
            self._use_df_ = True
            self.columns_ = X.columns
            self.q_low_ = X.quantile(self.lower)
            self.q_hi_ = X.quantile(self.upper)
        else:
            self._use_df_ = False
            X = np.asarray(X)
            # shape: (n_features,)
            self.q_low_ = np.quantile(X, self.lower, axis=0)
            self.q_hi_ = np.quantile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        if getattr(self, "_use_df_", False) and isinstance(X, pd.DataFrame):
            # Pandas path
            return X.clip(lower=self.q_low_, upper=self.q_hi_, axis=1)
        else:
            # NumPy path
            X = np.asarray(X)
            return np.clip(X, self.q_low_, self.q_hi_)

# ======================== Main ========================
def main():
    # -------- Load --------
    header("LOAD CLEAN DATA")
    if not os.path.exists(CLEAN_DATA_PATH):
        raise FileNotFoundError(f"Clean CSV not found at: {os.path.abspath(CLEAN_DATA_PATH)}")

    df = pd.read_csv(CLEAN_DATA_PATH)
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    if TARGET not in df.columns:
        raise ValueError(f"Target '{TARGET}' not found in the dataset.")

    # -------- Features / Target --------
    y = df[TARGET].values
    X = df.drop(columns=[TARGET])

    # Identify column types
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()

    print(f"Numeric features ({len(numeric_cols)}): {numeric_cols[:10]}{'...' if len(numeric_cols)>10 else ''}")
    print(f"Categorical features ({len(categorical_cols)}): {categorical_cols[:10]}{'...' if len(categorical_cols)>10 else ''}")

    # Split (train on log-target, report in original units)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    y_train_log = np.log1p(y_train)  # log1p for stability

    # -------- Preprocessing --------
    numeric_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("winsor", Winsorizer(lower=0.01, upper=0.99)),
        # No scaling for trees
    ])

    # Only create the categorical pipe if there are categorical cols
    transformers = [("num", numeric_pipe, numeric_cols)]
    if len(categorical_cols) > 0:
        categorical_pipe = Pipeline(steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=0.01)),
        ])
        transformers.append(("cat", categorical_pipe, categorical_cols))

    preproc = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False
    )

    # -------- XGBoost Model --------
    xgb = XGBRegressor(
        n_estimators=800,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
        objective="reg:squarederror",
        verbosity=0
    )

    pipe = Pipeline(steps=[
        ("preprocessor", preproc),
        ("model", xgb),
    ])

    # -------- Train --------
    header("TRAIN XGBOOST (log-target)")
    pipe.fit(X_train, y_train_log)
    print("Training complete.")

    # -------- Evaluate (back to original units) --------
    header("EVALUATE ON HOLDOUT (original units)")
    y_pred_log = pipe.predict(X_valid)
    y_pred = np.expm1(y_pred_log)  # invert log1p

    print(f"RMSE: {rmse(y_valid, y_pred):,.2f}")
    print(f"MAE : {mean_absolute_error(y_valid, y_pred):,.2f}")
    print(f"R²  : {r2_score(y_valid, y_pred):.4f}")

    # -------- Save model --------
    header("SAVE MODEL")
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    print(f"✅ Saved XGBoost pipeline to: {MODEL_PATH}")

    # Quick reload test
    loaded = joblib.load(MODEL_PATH)
    _ = loaded.predict(X_valid.iloc[:1])
    print("Reload test: OK")

if __name__ == "__main__":
    main()
