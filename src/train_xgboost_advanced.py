# train_xgb_tuned.py
import os
import math
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from scipy.stats import randint, uniform
from xgboost import XGBRegressor

# ======================= Config =======================
CLEAN_DATA_PATH = "../data/kc_house_data_clean.csv"
MODEL_DIR = "../models"
MODEL_PATH = os.path.join(MODEL_DIR, "house_price_xgb_tuned.joblib")
TARGET = "price"
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_CLUSTERS = 30  # geo clusters from lat/long
# Approx Seattle downtown for distance feature:
CENTER_LAT, CENTER_LONG = 47.6062, -122.3321

# ======================= Helpers ======================
def header(title: str):
    line = "=" * 90
    print(f"\n{line}\n{title}\n{line}")

def rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))

class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip each numeric column to [q_low, q_hi] quantiles. Picklable."""
    def __init__(self, lower=0.01, upper=0.99):
        self.lower = float(lower)
        self.upper = float(upper)

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self._use_df_ = True
            self.q_low_ = X.quantile(self.lower)
            self.q_hi_ = X.quantile(self.upper)
        else:
            self._use_df_ = False
            X = np.asarray(X)
            self.q_low_ = np.quantile(X, self.lower, axis=0)
            self.q_hi_ = np.quantile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        if getattr(self, "_use_df_", False) and isinstance(X, pd.DataFrame):
            return X.clip(lower=self.q_low_, upper=self.q_hi_, axis=1)
        X = np.asarray(X)
        return np.clip(X, self.q_low_, self.q_hi_)

def safe_div(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(b == 0, np.nan, a / b)
    return out

def dist_to_center(lat, lon, c_lat=CENTER_LAT, c_lon=CENTER_LONG):
    # Euclidean proxy in degrees (good enough for local area)
    return np.sqrt((np.asarray(lat) - c_lat)**2 + (np.asarray(lon) - c_lon)**2)

# ======================== Main ========================
def main():
    # -------- Load --------
    header("LOAD CLEAN DATA")
    if not os.path.exists(CLEAN_DATA_PATH):
        raise FileNotFoundError(f"Clean CSV not found at: {os.path.abspath(CLEAN_DATA_PATH)}")
    df = pd.read_csv(CLEAN_DATA_PATH)
    print(f"Shape: {df.shape}")
    if TARGET not in df.columns:
        raise ValueError(f"Target '{TARGET}' not found.")

    # -------- Split FIRST (avoid leakage) --------
    y = df[TARGET].values
    X = df.drop(columns=[TARGET]).copy()
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # -------- Feature engineering (fit ONLY on train, apply to both) --------
    # Zipcode as string
    if "zipcode" in X_train.columns:
        X_train["zipcode"] = X_train["zipcode"].astype("Int64").astype("string")
        X_valid["zipcode"] = X_valid["zipcode"].astype("Int64").astype("string")

    # Target mean per zipcode (train only)
    if "zipcode" in X_train.columns:
        zip_mean = (
            pd.DataFrame({"zipcode": X_train["zipcode"], "price": y_train})
            .groupby("zipcode")["price"].mean()
        )
        global_mean = float(np.mean(y_train))
        X_train["zipcode_mean_price"] = X_train["zipcode"].map(zip_mean).fillna(global_mean)
        X_valid["zipcode_mean_price"] = X_valid["zipcode"].map(zip_mean).fillna(global_mean)

    # Geo clusters (lat/long)
    kmeans = None
    if {"lat", "long"}.issubset(X_train.columns):
        kmeans = KMeans(n_clusters=N_CLUSTERS, n_init="auto", random_state=RANDOM_STATE)
        kmeans.fit(X_train[["lat", "long"]])
        X_train["loc_cluster"] = pd.Categorical(kmeans.predict(X_train[["lat", "long"]]))
        X_valid["loc_cluster"] = pd.Categorical(kmeans.predict(X_valid[["lat", "long"]]))

        # Distance to city center
        X_train["dist_center"] = dist_to_center(X_train["lat"], X_train["long"])
        X_valid["dist_center"] = dist_to_center(X_valid["lat"], X_valid["long"])

    # Ratios / interactions
    if {"sqft_living", "sqft_lot"}.issubset(X_train.columns):
        X_train["living_to_lot"] = safe_div(X_train["sqft_living"], X_train["sqft_lot"])
        X_valid["living_to_lot"] = safe_div(X_valid["sqft_living"], X_valid["sqft_lot"])

    if {"bathrooms", "bedrooms"}.issubset(X_train.columns):
        X_train["bath_bed_ratio"] = safe_div(X_train["bathrooms"], X_train["bedrooms"])
        X_valid["bath_bed_ratio"] = safe_div(X_valid["bathrooms"], X_valid["bedrooms"])

    if {"sqft_above", "sqft_living"}.issubset(X_train.columns):
        X_train["above_to_living"] = safe_div(X_train["sqft_above"], X_train["sqft_living"])
        X_valid["above_to_living"] = safe_div(X_valid["sqft_above"], X_valid["sqft_living"])

    # Log transforms for skewed features (don’t overwrite originals)
    for col in ["sqft_living", "sqft_lot", "sqft_above", "sqft_basement"]:
        if col in X_train.columns:
            X_train[f"log_{col}"] = np.log1p(X_train[col])
            X_valid[f"log_{col}"] = np.log1p(X_valid[col])

    # -------- Target in log-space --------
    y_train_log = np.log1p(y_train)
    y_valid_log = np.log1p(y_valid)

    # -------- Column types after engineering --------
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=["object", "string", "category"]).columns.tolist()

    # Ensure engineered columns buckets
    for col in ["zipcode_mean_price", "living_to_lot", "bath_bed_ratio", "above_to_living",
                "dist_center", "log_sqft_living", "log_sqft_lot", "log_sqft_above", "log_sqft_basement"]:
        if col in X_train.columns and col not in numeric_cols:
            numeric_cols.append(col)
    if "loc_cluster" in X_train.columns and "loc_cluster" not in categorical_cols:
        categorical_cols.append("loc_cluster")
    if "zipcode" in X_train.columns and "zipcode" not in categorical_cols:
        categorical_cols.append("zipcode")

    header("FEATURE OVERVIEW")
    print(f"Numeric ({len(numeric_cols)}): {numeric_cols[:12]}{'...' if len(numeric_cols)>12 else ''}")
    print(f"Categorical ({len(categorical_cols)}): {categorical_cols[:12]}{'...' if len(categorical_cols)>12 else ''}")

    # -------- Preprocessing --------
    num_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("winsor", Winsorizer(lower=0.01, upper=0.99)),
    ])

    transformers = [("num", num_pipe, numeric_cols)]
    if len(categorical_cols) > 0:
        cat_pipe = Pipeline(steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=0.01)),
        ])
        transformers.append(("cat", cat_pipe, categorical_cols))

    preproc = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )

    # -------- Fit preprocessor & transform --------
    header("FIT PREPROCESSOR & TRANSFORM")
    preproc.fit(X_train)
    X_train_tr = preproc.transform(X_train)
    X_valid_tr = preproc.transform(X_valid)
    print(f"Train transformed shape: {X_train_tr.shape} | Valid: {X_valid_tr.shape}")

    # -------- XGBoost base model --------
    xgb = XGBRegressor(
        n_estimators=3000,      # refit later with best params
        learning_rate=0.03,
        max_depth=8,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
        objective="reg:squarederror",
        eval_metric="rmse",
        verbosity=0,
    )

    # -------- Hyperparameter search (RandomizedSearchCV) --------
    header("HYPERPARAMETER TUNING (RandomizedSearchCV on log-target)")
    param_dist = {
        "n_estimators": randint(1200, 6000),
        "learning_rate": uniform(0.01, 0.05),    # 0.01–0.06
        "max_depth": randint(4, 12),
        "min_child_weight": randint(1, 12),
        "subsample": uniform(0.6, 0.4),          # 0.6–1.0
        "colsample_bytree": uniform(0.6, 0.4),   # 0.6–1.0
        "reg_lambda": uniform(0.3, 2.0),         # 0.3–2.3
        "reg_alpha": uniform(0.0, 0.5),          # 0.0–0.5
        # "gamma": uniform(0.0, 0.3),            # add if your xgboost has gamma
    }

    # scoring on log-target: neg RMSE in log-space
    search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=param_dist,
        n_iter=40,
        scoring="neg_root_mean_squared_error",
        cv=3,
        random_state=RANDOM_STATE,
        verbose=1,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train_tr, y_train_log)
    print("Best params:", search.best_params_)
    best_model = search.best_estimator_

    # -------- Evaluate (original units) --------
    header("EVALUATE ON HOLDOUT (original units)")
    y_pred_log = best_model.predict(X_valid_tr)
    y_pred = np.expm1(y_pred_log)
    print(f"RMSE: {rmse(y_valid, y_pred):,.2f}")
    print(f"MAE : {mean_absolute_error(y_valid, y_pred):,.2f}")
    print(f"R²  : {r2_score(y_valid, y_pred):.4f}")

    # -------- Save bundle --------
    header("SAVE MODEL + AUX OBJECTS")
    os.makedirs(MODEL_DIR, exist_ok=True)
    bundle = {
        "preprocessor": preproc,
        "model": best_model,
        "kmeans": kmeans,  # may be None
        "config": {
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
            "n_clusters": N_CLUSTERS,
            "target": TARGET,
            "center": (CENTER_LAT, CENTER_LONG),
            "best_params": search.best_params_,
        },
    }
    joblib.dump(bundle, MODEL_PATH)
    print(f"✅ Saved to: {MODEL_PATH}")

    # Reload smoke test
    loaded = joblib.load(MODEL_PATH)
    Xv_tr = loaded["preprocessor"].transform(X_valid.iloc[:1])
    _ = loaded["model"].predict(Xv_tr)
    print("Reload test: OK")

if __name__ == "__main__":
    main()
