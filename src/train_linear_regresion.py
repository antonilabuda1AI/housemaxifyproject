# train_linear.py
import os
import math
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ---- Config ----
CLEAN_DATA_PATH = "../data/kc_house_data_clean.csv"
MODEL_DIR = "../models"
MODEL_PATH = os.path.join(MODEL_DIR, "house_price_linear.joblib")
TARGET = "price"
RANDOM_STATE = 42
TEST_SIZE = 0.2  # 80/20 split

def header(title: str):
    line = "=" * 80
    print(f"\n{line}\n{title}\n{line}")

def rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))

def main():
    # -------- Load --------
    header("LOAD CLEAN DATA")
    if not os.path.exists(CLEAN_DATA_PATH):
        raise FileNotFoundError(f"Clean CSV not found at: {os.path.abspath(CLEAN_DATA_PATH)}")

    df = pd.read_csv(CLEAN_DATA_PATH)
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    if TARGET not in df.columns:
        raise ValueError(f"Target '{TARGET}' not found.")

    # -------- Features / Target --------
    y = df[TARGET].values
    X = df.drop(columns=[TARGET])

    # Identify column types
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()

    print(f"Numeric features ({len(numeric_cols)}): {numeric_cols[:10]}{'...' if len(numeric_cols)>10 else ''}")
    print(f"Categorical features ({len(categorical_cols)}): {categorical_cols[:10]}{'...' if len(categorical_cols)>10 else ''}")

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # -------- Preprocessing --------
    numeric_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])

    categorical_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preproc = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )

    # -------- Model --------
    model = LinearRegression()

    pipe = Pipeline(steps=[
        ("preprocessor", preproc),
        ("model", model),
    ])

    # -------- Train --------
    header("TRAIN")
    pipe.fit(X_train, y_train)
    print("Training complete.")

    # -------- Evaluate --------
    header("EVALUATE (Holdout)")
    y_pred = pipe.predict(X_valid)
    print(f"RMSE: {rmse(y_valid, y_pred):,.2f}")
    print(f"MAE : {mean_absolute_error(y_valid, y_pred):,.2f}")
    print(f"R²  : {r2_score(y_valid, y_pred):.4f}")

    # -------- Save --------
    header("SAVE MODEL")
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    print(f"✅ Saved trained pipeline to: {MODEL_PATH}")

    # (Optional) quick reload test
    loaded = joblib.load(MODEL_PATH)
    _ = loaded.predict(X_valid.iloc[:1])
    print("Reload test: OK")

if __name__ == "__main__":
    main()
