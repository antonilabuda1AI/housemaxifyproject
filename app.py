import os
from datetime import datetime
from typing import List, Dict, Any, Callable, Optional, Tuple

import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash
from werkzeug.exceptions import NotFound
import joblib

# Compatibility shim for pickled pipelines that reference a `Winsorizer`
# defined in training notebooks/scripts as `__main__.Winsorizer`.
# If `feature_engine` is available, we use the real implementation;
# otherwise we provide a passthrough to allow unpickling and prediction.
try:
    from feature_engine.outliers import Winsorizer as _FEWinsorizer  # type: ignore

    class Winsorizer(_FEWinsorizer):  # type: ignore
        pass
except Exception:
    class Winsorizer:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            return X

        def fit_transform(self, X, y=None):
            return X


# Flask setup
# - Templates served from ./html
# - Static-like folders mapped via explicit routes: /css, /js, /assets
app = Flask(__name__, template_folder='html', static_folder=None)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-house-maxify")


# Paths
DATA_PATH = os.path.join("data", "kc_house_data_clean.csv")
MODEL_PATH = os.path.join("model", "house_price_xgb_advanced.joblib")
LEADS_PATH = os.path.join("data", "leads.csv")


# Globals (loaded at startup)
model = None
data_df: pd.DataFrame | None = None
feature_names: List[str] = []
# Output calibration (fitted at startup if possible)
_calibrator: Optional[Tuple[str, float, float]] = None  # (mode, a, b)
_metrics_cache: Optional[Dict[str, Any]] = None


def _is_pipeline_like(obj: Any) -> bool:
    try:
        name = type(obj).__name__.lower()
        return hasattr(obj, "predict") and (hasattr(obj, "named_steps") or "pipeline" in name)
    except Exception:
        return False


def _is_ttr_like(obj: Any) -> bool:
    try:
        name = type(obj).__name__.lower()
        return hasattr(obj, "predict") and ("transformedtargetregressor" in name or hasattr(obj, "regressor_"))
    except Exception:
        return False


def _resolve_predictor(obj: Any):
    # If it already predicts, return it
    if hasattr(obj, "predict"):
        return obj

    # If dict, look for common keys or recurse values
    visited: set[int] = set()
    stack: List[Any] = [obj]
    common_keys = [
        "model",
        "estimator",
        "best_estimator_",
        "regressor",
        "pipeline",
        "xgb_model",
    ]
    pipeline_candidate = None
    ttr_candidate = None
    simple_candidate = None

    while stack:
        cur = stack.pop()
        ident = id(cur)
        if ident in visited:
            continue
        visited.add(ident)
        if _is_pipeline_like(cur):
            pipeline_candidate = pipeline_candidate or cur
        elif _is_ttr_like(cur):
            ttr_candidate = ttr_candidate or cur
        elif hasattr(cur, "predict"):
            simple_candidate = simple_candidate or cur
        # Try common attributes
        for k in common_keys:
            if hasattr(cur, k):
                stack.append(getattr(cur, k))
        # Dive into containers
        if isinstance(cur, dict):
            # Prioritize common keys when present
            for k in common_keys:
                if k in cur:
                    stack.append(cur[k])
            # Then explore remaining values
            for key, v in cur.items():
                if key not in common_keys:
                    stack.append(v)
        elif isinstance(cur, (list, tuple, set)):
            for v in cur:
                stack.append(v)
    # Prefer full pipeline, then TTR, then any predictor, else original
    return pipeline_candidate or ttr_candidate or simple_candidate or obj


def _detect_feature_names(predictor: Any, df: pd.DataFrame | None) -> List[str]:
    fn: List[str] = []
    if hasattr(predictor, "feature_names_in_"):
        try:
            fn = list(getattr(predictor, "feature_names_in_"))
        except Exception:
            fn = []
    elif hasattr(predictor, "get_booster"):
        try:
            booster = predictor.get_booster()
            if booster is not None and hasattr(booster, "feature_names") and booster.feature_names:
                fn = list(booster.feature_names)
        except Exception:
            fn = []
    if not fn and df is not None and not df.empty:
        exclude = {"price", "id", "date"}
        fn = [c for c in df.columns if c.lower() not in exclude]
    return fn


def _fit_output_calibrator(predictor: Any, df: pd.DataFrame | None, fn: List[str]) -> Optional[Tuple[str, float, float]]:
    # Attempt to learn a simple mapping from model raw output to actual price
    # using the dataset. Returns (mode, a, b) where price ≈ a * g(y_raw) + b.
    try:
        if df is None or df.empty or "price" not in df.columns or not fn:
            return None
        available = [c for c in fn if c in df.columns]
        if not available:
            return None

        # Sample up to N rows for efficiency
        N = min(400, len(df))
        sample = df.sample(N, random_state=42) if len(df) > N else df.copy()
        Xs = sample[available].copy()
        # Coerce numeric
        for c in Xs.columns:
            Xs[c] = pd.to_numeric(Xs[c], errors="coerce")
        Xs = Xs.fillna(Xs.median(numeric_only=True)).fillna(0.0)

        # Predict raw
        y_raw = predictor.predict(Xs)
        y_raw = pd.to_numeric(pd.Series(y_raw), errors="coerce").astype(float).values
        y_true = pd.to_numeric(sample["price"], errors="coerce").astype(float).values

        # If y_true seems to be in thousands, upscale to dollars
        if np.nanmedian(y_true) < 25000:
            y_true = y_true * 1000.0

        # Candidate transforms g(y)
        def safe_expm1(v):
            try:
                return np.expm1(v)
            except Exception:
                return np.full_like(v, np.nan)

        def safe_pow10(v):
            try:
                return np.power(10.0, v)
            except Exception:
                return np.full_like(v, np.nan)

        candidates: List[Tuple[str, np.ndarray]] = [
            ("identity", y_raw),
            ("expm1", safe_expm1(y_raw)),
            ("pow10", safe_pow10(y_raw)),
            ("x1e3", y_raw * 1e3),
            ("x1e4", y_raw * 1e4),
            ("x1e5", y_raw * 1e5),
        ]

        best_mode = None
        best_score = -np.inf
        best_a, best_b = 1.0, 0.0

        for mode, gy in candidates:
            gy = pd.to_numeric(pd.Series(gy), errors="coerce").astype(float).values
            msk = np.isfinite(gy) & np.isfinite(y_true)
            if msk.sum() < 20:
                continue
            G = np.vstack([gy[msk], np.ones(msk.sum())]).T
            # Solve least squares for a, b: y_true ≈ a * gy + b
            a, b = np.linalg.lstsq(G, y_true[msk], rcond=None)[0]
            pred = a * gy[msk] + b
            # Compute R^2 as score
            ss_res = np.sum((y_true[msk] - pred) ** 2)
            ss_tot = np.sum((y_true[msk] - np.mean(y_true[msk])) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else -np.inf

            # Prefer higher R^2; break ties by lower MAE
            mae = np.mean(np.abs(y_true[msk] - pred))
            score = r2 - 1e-9 * mae  # tiny tie-breaker
            if score > best_score:
                best_score = score
                best_mode = mode
                best_a, best_b = float(a), float(b)

        # Require a minimum R^2 improvement over identity
        if best_mode is None:
            return None

        # If best is identity but R^2 is reasonable, we still keep it
        return (best_mode, best_a, best_b)
    except Exception:
        return None


def load_resources():
    global model, data_df, feature_names

    # Load model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    loaded = joblib.load(MODEL_PATH)
    model = _resolve_predictor(loaded)

    # Load dataset (for feature reference and example properties)
    if os.path.exists(DATA_PATH):
        data_df = pd.read_csv(DATA_PATH)
    else:
        data_df = pd.DataFrame()

    # Determine feature names expected by the model
    feature_names = _detect_feature_names(model, data_df)

    # Fit output calibrator if possible
    global _calibrator
    _calibrator = _fit_output_calibrator(model, data_df, feature_names)


def _predict_with_calibration(X: pd.DataFrame) -> np.ndarray:
    # Raw predictions
    y_raw = model.predict(X)
    y_raw = np.asarray(y_raw, dtype=float)
    y_pred = y_raw.copy()

    # Apply learned calibrator when available
    if _calibrator is not None:
        mode, a, b = _calibrator
        if mode == "identity":
            y_pred = a * y_raw + b
        elif mode == "expm1":
            y_pred = a * np.expm1(y_raw) + b
        elif mode == "pow10":
            y_pred = a * np.power(10.0, y_raw) + b
        elif mode == "x1e3":
            y_pred = a * (y_raw * 1e3) + b
        elif mode == "x1e4":
            y_pred = a * (y_raw * 1e4) + b
        elif mode == "x1e5":
            y_pred = a * (y_raw * 1e5) + b
    else:
        # Fallback heuristic only if not a full pipeline/TTR
        if not (_is_pipeline_like(model) or _is_ttr_like(model)):
            y_pred = np.array([_calibrate_price(float(v)) for v in y_raw], dtype=float)
    return y_pred


def _build_features_from_df(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    use_cols = [c for c in cols if c in df.columns]
    X = df[use_cols].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    # Fill missing with dataset medians, then zeros
    for c in X.columns:
        med = pd.to_numeric(df[c], errors='coerce').median() if c in df.columns else np.nan
        if pd.notna(med):
            X[c] = X[c].fillna(float(med))
        X[c] = X[c].fillna(0.0)
    return X


def compute_metrics() -> Optional[Dict[str, Any]]:
    global _metrics_cache
    if _metrics_cache is not None:
        return _metrics_cache

    try:
        if data_df is None or data_df.empty or model is None or not feature_names:
            return None
        if "price" not in data_df.columns:
            return None

        df = data_df.copy()
        # Sample for speed
        N = min(1000, len(df))
        if len(df) > N:
            df = df.sample(N, random_state=42)

        X = _build_features_from_df(df, feature_names)
        y_true = pd.to_numeric(df["price"], errors="coerce").astype(float).values
        # Upscale if prices are likely in thousands
        if np.nanmedian(y_true) < 25000:
            y_true = y_true * 1000.0

        y_pred = _predict_with_calibration(X)
        msk = np.isfinite(y_true) & np.isfinite(y_pred)
        if msk.sum() < 20:
            return None

        yt = y_true[msk]
        yp = y_pred[msk]
        mae = float(np.mean(np.abs(yt - yp)))
        rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
        # R^2
        ss_res = float(np.sum((yt - yp) ** 2))
        ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
        # MAPE (clip true to avoid div-by-zero blowups)
        yt_clip = np.clip(np.abs(yt), 1e-6, None)
        mape = float(np.mean(np.abs((yt - yp) / yt_clip)))

        _metrics_cache = {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "mape": mape,
            "n_samples": int(msk.sum()),
        }
        return _metrics_cache
    except Exception:
        return None


def ensure_leads_csv_header():
    # Create leads CSV with header if missing
    if not os.path.exists(LEADS_PATH):
        os.makedirs(os.path.dirname(LEADS_PATH), exist_ok=True)
        header = [
            "timestamp",
            "name",
            "email",
            "phone",
            "intent",
            "timeline_months",
            "prediction",
            "prediction_low",
            "prediction_high",
            "zipcode",
            "bedrooms",
            "bathrooms",
            "sqft_living",
        ]
        pd.DataFrame(columns=header).to_csv(LEADS_PATH, index=False)


def _coerce_float(val, default=np.nan):
    if val is None or val == "":
        return default
    try:
        return float(val)
    except Exception:
        return default


def _extract_features_from_form(form) -> pd.DataFrame:
    # Map known inputs from the form to numeric values
    # Include a broad set of common KC housing features; missing ones default to median
    candidate_inputs = {
        "bedrooms": _coerce_float(form.get("bedrooms")),
        "bathrooms": _coerce_float(form.get("bathrooms")),
        "sqft_living": _coerce_float(form.get("sqft_living")),
        "sqft_lot": _coerce_float(form.get("sqft_lot")),
        "floors": _coerce_float(form.get("floors")),
        "waterfront": _coerce_float(form.get("waterfront")),
        "view": _coerce_float(form.get("view")),
        "condition": _coerce_float(form.get("condition")),
        "grade": _coerce_float(form.get("grade")),
        "sqft_above": _coerce_float(form.get("sqft_above")),
        "sqft_basement": _coerce_float(form.get("sqft_basement")),
        "yr_built": _coerce_float(form.get("yr_built")),
        "yr_renovated": _coerce_float(form.get("yr_renovated")),
        "zipcode": _coerce_float(form.get("zipcode")),
        "lat": _coerce_float(form.get("lat")),
        "long": _coerce_float(form.get("long")),
        "sqft_living15": _coerce_float(form.get("sqft_living15")),
        "sqft_lot15": _coerce_float(form.get("sqft_lot15")),
    }

    # Start with inferred feature list if available; otherwise use provided candidates
    if feature_names:
        cols = feature_names
    else:
        cols = list(candidate_inputs.keys())

    # Create single-row DataFrame
    X = pd.DataFrame([{c: candidate_inputs.get(c, np.nan) for c in cols}], columns=cols)

    # Integer-like rounding for typical integer features
    int_like_cols = [
        "bedrooms", "zipcode", "view", "condition", "grade",
        "sqft_living", "sqft_lot", "sqft_above", "sqft_basement",
        "sqft_living15", "sqft_lot15", "yr_built", "yr_renovated", "waterfront"
    ]

    for c in X.columns:
        if c in int_like_cols and X[c].notna().any():
            try:
                X[c] = np.round(pd.to_numeric(X[c], errors='coerce'))
            except Exception:
                pass

    # Fill missing with dataset medians if available, else with column medians from X
    if data_df is not None and not data_df.empty:
        for c in X.columns:
            if X[c].isna().any():
                if c in data_df.columns:
                    try:
                        med = float(pd.to_numeric(data_df[c], errors='coerce').median())
                        X[c] = X[c].fillna(med)
                    except Exception:
                        X[c] = X[c].fillna(0.0)
                else:
                    X[c] = X[c].fillna(0.0)
    else:
        X = X.fillna(X.median(numeric_only=True))
        X = X.fillna(0.0)

    return X


def _format_currency(val: float) -> str:
    try:
        return f"${val:,.0f}"
    except Exception:
        return str(val)


def _calibrate_price(y_raw: float) -> float:
    # If already in a plausible dollar range, keep it
    if 50_000 <= y_raw <= 10_000_000:
        return y_raw

    # Target anchor: dataset median price if available, else a typical value
    anchor = 450_000.0
    if data_df is not None and not data_df.empty and "price" in data_df.columns:
        try:
            med = pd.to_numeric(data_df["price"], errors="coerce").median()
            if pd.notna(med) and med > 0:
                anchor = float(med)
                # If the dataset uses thousands (e.g., 450 instead of 450,000), upscale
                if anchor < 25_000:
                    anchor *= 1_000.0
        except Exception:
            pass

    cands = []
    y = float(y_raw)
    # Raw
    if y > 0:
        cands.append(y)
    # Exponential transforms (common when training on log targets)
    try:
        cands.append(float(np.expm1(y)))
    except Exception:
        pass
    try:
        cands.append(float(10 ** y))
    except Exception:
        pass
    # Linear scalings (common when normalizing to units of 1e3/1e5/1e6)
    for scale in (1e3, 1e4, 1e5, 1e6):
        cands.append(float(y * scale))

    # Filter to positive candidates
    cands = [c for c in cands if c > 0]
    if not cands:
        return max(y_raw, 0.0)

    # Choose candidate closest to anchor on log scale, constrained to a plausible range
    plausible = [c for c in cands if 50_000 <= c <= 10_000_000]
    if not plausible:
        plausible = cands

    def log_dist(c):
        try:
            return abs(np.log10(c) - np.log10(anchor))
        except Exception:
            return float("inf")

    best = min(plausible, key=log_dist)
    return float(best)


def _find_similar_properties_v2(user_features: pd.DataFrame, predicted_price: float) -> List[Dict[str, Any]]:
    # Progressive constraint selection for highly relevant “similar properties”.
    out: List[Dict[str, Any]] = []
    if data_df is None or data_df.empty:
        # Fallback to simple echoes around prediction
        base = float(predicted_price)
        def _fv(col, default):
            try:
                return (float(user_features[col].iloc[0]) if isinstance(user_features, pd.DataFrame) else float(user_features.get(col, default)))
            except Exception:
                return default
        for i in range(3):
            out.append({
                "bedrooms": int(_fv("bedrooms", 3)),
                "bathrooms": float(_fv("bathrooms", 2.0)),
                "sqft_living": int(_fv("sqft_living", 1800)),
                "zipcode": int(_fv("zipcode", 98103)),
                "price": base * (0.9 + 0.05 * i),
            })
        return out

    df = data_df.copy()
    for c in ["bedrooms", "bathrooms", "sqft_living", "grade", "zipcode", "price"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    def uf(col, default=np.nan):
        try:
            if isinstance(user_features, pd.DataFrame):
                return float(user_features[col].iloc[0])
            return float(user_features.get(col, default))
        except Exception:
            return default

    zp = None
    try:
        zp = int(float(uf("zipcode")))
    except Exception:
        pass
    b = uf("bedrooms")
    ba = uf("bathrooms")
    s = uf("sqft_living")
    g = uf("grade")

    price_available = ("price" in df.columns and df["price"].notna().any())
    def plausible(p):
        try:
            return 50_000 <= float(p) <= 5_000_000
        except Exception:
            return False
    use_price = price_available and plausible(predicted_price)

    def select(th_price=0.15, th_sqft=0.20, th_bed=1.0, th_bath=1.0, th_grade=1.0, same_zip=True):
        c = df.copy()
        if same_zip and zp is not None and "zipcode" in c.columns:
            c = c[c["zipcode"] == zp]
        # Drop obvious NaNs
        for col in ["bedrooms", "bathrooms", "sqft_living"]:
            if col in c.columns:
                c = c[c[col].notna()]

        if not np.isnan(b):
            c = c[np.abs(c.get("bedrooms", b) - b) <= th_bed]
        if not np.isnan(ba):
            c = c[np.abs(c.get("bathrooms", ba) - ba) <= th_bath]
        if not np.isnan(g):
            c = c[np.abs(c.get("grade", g) - g) <= th_grade]
        if not np.isnan(s) and s > 0:
            band = max(th_sqft * s, 300.0)
            c = c[np.abs(c.get("sqft_living", s) - s) <= band]
        if use_price and "price" in c.columns:
            pband = max(th_price * predicted_price, 25_000.0)
            c = c[np.abs(c["price"] - predicted_price) <= pband]

        if c.empty:
            return c

        s_norm = max(s if s and s > 0 else 1.0, 1.0)
        c = c.assign(
            _b=np.abs(pd.to_numeric(c.get("bedrooms", b), errors="coerce") - b),
            _ba=np.abs(pd.to_numeric(c.get("bathrooms", ba), errors="coerce") - ba),
            _s=(np.abs(pd.to_numeric(c.get("sqft_living", s), errors="coerce") - s) / s_norm),
            _g=np.abs(pd.to_numeric(c.get("grade", g), errors="coerce") - g),
            _p=(np.abs(pd.to_numeric(c.get("price", predicted_price), errors="coerce") - predicted_price) / max(predicted_price, 1.0)) if use_price else 0.0,
        )
        c["_dist"] = c["_s"] * 2.0 + c["_b"] * 1.2 + c["_ba"] * 0.9 + c["_g"] * 0.5 + (c["_p"] * 0.6 if use_price else 0.0)
        return c.sort_values("_dist").head(3)

    # Progressive strategies
    strategies = [
        dict(th_price=0.12, th_sqft=0.18, th_bed=1.0, th_bath=1.0, th_grade=1.0, same_zip=True),
        dict(th_price=0.20, th_sqft=0.25, th_bed=1.0, th_bath=1.0, th_grade=1.0, same_zip=True),
        dict(th_price=0.25, th_sqft=0.35, th_bed=1.5, th_bath=1.25, th_grade=1.5, same_zip=True),
        dict(th_price=0.20, th_sqft=0.25, th_bed=1.0, th_bath=1.0, th_grade=1.0, same_zip=False),
        dict(th_price=0.30, th_sqft=0.40, th_bed=2.0, th_bath=1.5, th_grade=2.0, same_zip=False),
    ]

    chosen = None
    for sdef in strategies:
        chosen = select(**sdef)
        if chosen is not None and not chosen.empty:
            break

    if chosen is None or chosen.empty:
        # Fallback: nearest by sqft+bed across dataset
        c = df.assign(
            _b=np.abs(pd.to_numeric(df.get("bedrooms", b), errors="coerce") - b),
            _s=np.abs(pd.to_numeric(df.get("sqft_living", s), errors="coerce") - s),
        ).sort_values(["_b", "_s"]).head(3)
        chosen = c

    for _, r in chosen.iterrows():
        out.append({
            "bedrooms": int(r.get("bedrooms", np.nan)) if not pd.isna(r.get("bedrooms", np.nan)) else None,
            "bathrooms": float(r.get("bathrooms", np.nan)) if not pd.isna(r.get("bathrooms", np.nan)) else None,
            "sqft_living": int(r.get("sqft_living", np.nan)) if not pd.isna(r.get("sqft_living", np.nan)) else None,
            "zipcode": int(r.get("zipcode", np.nan)) if not pd.isna(r.get("zipcode", np.nan)) else None,
            "price": float(r.get("price", predicted_price)) if price_available else predicted_price,
        })

    return out
def _find_similar_properties(user_features: pd.DataFrame, predicted_price: float) -> List[Dict[str, Any]]:
    # Robust similarity: prioritize same zipcode and feature proximity.
    # Use price as a weak signal only if the predicted price is plausible.
    examples: List[Dict[str, Any]] = []
    if data_df is None or data_df.empty:
        # Fallback dummy examples
        base = predicted_price
        for i in range(3):
            examples.append({
                "bedrooms": int((user_features["bedrooms"].iloc[0] if isinstance(user_features, pd.DataFrame) else user_features.get("bedrooms", 3)) or 3),
                "bathrooms": float((user_features["bathrooms"].iloc[0] if isinstance(user_features, pd.DataFrame) else user_features.get("bathrooms", 2.0)) or 2.0),
                "sqft_living": int((user_features["sqft_living"].iloc[0] if isinstance(user_features, pd.DataFrame) else user_features.get("sqft_living", 1800)) or 1800),
                "zipcode": int((user_features["zipcode"].iloc[0] if isinstance(user_features, pd.DataFrame) else user_features.get("zipcode", 98103)) or 98103),
                "price": base * (0.9 + 0.1 * i),
            })
        return examples

    df = data_df.copy()
    # Use numeric only
    for col in [
        "bedrooms",
        "bathrooms",
        "sqft_living",
        "zipcode",
        "price",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Filter by same zipcode if provided
    def _uf(col, default=np.nan):
        try:
            if isinstance(user_features, pd.DataFrame):
                return float(user_features[col].iloc[0])
            return float(user_features.get(col, default))
        except Exception:
            return default

    try:
        zp = int(float(_uf("zipcode")))
    except Exception:
        zp = None
    if zp is not None and "zipcode" in df.columns:
        df = df[df["zipcode"] == zp]

    # Composite proximity metric using key features; price is optional and weakly weighted
    price_available = ("price" in df.columns and df["price"].notna().any())
    b = _uf("bedrooms")
    s = _uf("sqft_living")
    ba = _uf("bathrooms")
    gr = _uf("grade")

    # Determine if predicted price is plausible; otherwise ignore price distance
    def _plausible_price(p: float) -> bool:
        try:
            return 50000 <= float(p) <= 5_000_000
        except Exception:
            return False

    use_price = price_available and _plausible_price(predicted_price)
    if use_price:
        df["_p"] = np.abs(pd.to_numeric(df["price"], errors="coerce") - predicted_price)
    else:
        df["_p"] = 0.0

    df["_b"] = np.abs(pd.to_numeric(df.get("bedrooms", 0), errors="coerce") - b)
    df["_s"] = np.abs(pd.to_numeric(df.get("sqft_living", 0), errors="coerce") - s)
    df["_ba"] = np.abs(pd.to_numeric(df.get("bathrooms", 0), errors="coerce") - ba)
    df["_g"] = np.abs(pd.to_numeric(df.get("grade", 0), errors="coerce") - gr)

    # Weighted distance: prioritize sqft and bedrooms; price influences only when plausible
    s_norm = max(s if s and s > 0 else 1.0, 1.0)
    df["_dist"] = (
        (df["_s"] / s_norm) * 2.0 +
        (df["_b"] * 1.2) +
        (df["_ba"] * 0.7) +
        (df["_g"] * 0.4) +
        ((df["_p"] / max(predicted_price, 1.0)) * 0.5 if use_price else 0.0)
    )

    sorted_df = df.sort_values("_dist")
    # Enforce tight similarity on bedrooms and sqft if user provided them
    try:
        b_val = float(b) if b == b else None
    except Exception:
        b_val = None
    try:
        s_val = float(s) if s == s else None
    except Exception:
        s_val = None

    if b_val is not None and s_val is not None and s_val > 0:
        tight = sorted_df[
            (sorted_df["_b"] <= 1.0) &
            (sorted_df["_s"] <= max(400.0, 0.25 * s_val))
        ]
        take = (tight.head(3) if len(tight) >= 3 else sorted_df.head(3))
    else:
        take = sorted_df.head(3)
    for _, r in take.iterrows():
        examples.append({
            "bedrooms": int(r.get("bedrooms", np.nan)) if not pd.isna(r.get("bedrooms", np.nan)) else None,
            "bathrooms": float(r.get("bathrooms", np.nan)) if not pd.isna(r.get("bathrooms", np.nan)) else None,
            "sqft_living": int(r.get("sqft_living", np.nan)) if not pd.isna(r.get("sqft_living", np.nan)) else None,
            "zipcode": int(r.get("zipcode", np.nan)) if not pd.isna(r.get("zipcode", np.nan)) else None,
            "price": float(r.get("price", predicted_price)) if price_available else predicted_price,
        })

    # Ensure 3 examples
    while len(examples) < 3:
        examples.append(examples[-1] if examples else {
            "bedrooms": 3, "bathrooms": 2.0, "sqft_living": 1800, "zipcode": zp or 98103, "price": predicted_price
        })
    return examples[:3]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/form", methods=["GET", "POST"])
def form():
    if request.method == "GET":
        return render_template("form.html", feature_set=set(feature_names or []))

    # Validate basic user fields
    name = (request.form.get("name") or "").strip()
    email = (request.form.get("email") or "").strip()
    phone = (request.form.get("phone") or "").strip()
    intent = (request.form.get("intent") or "").strip()  # buy or sell
    timeline = (request.form.get("timeline") or "").strip()

    missing = [f for f in ["name", "email", "phone"] if not (request.form.get(f) or "").strip()]
    if missing:
        flash("Please fill in your name, email, and phone number.", "error")
        return render_template("form.html", preserved=request.form, feature_set=set(feature_names or []))

    # Extract model features
    X = _extract_features_from_form(request.form)

    # Predict
    try:
        y_pred = float(_predict_with_calibration(X)[0])
    except Exception as e:
        flash(f"Prediction failed: {e}", "error")
        return render_template("form.html", preserved=request.form, feature_set=set(feature_names or []))

    # Construct range (+/- 10%)
    low = max(0.0, y_pred * 0.9)
    high = y_pred * 1.1

    # Find example similar properties
    similars = _find_similar_properties_v2(X, y_pred)

    # Save lead info to CSV
    ensure_leads_csv_header()
    lead_row = {
        "timestamp": datetime.utcnow().isoformat(),
        "name": name,
        "email": email,
        "phone": phone,
        "intent": intent,
        "timeline_months": timeline,
        "prediction": y_pred,
        "prediction_low": low,
        "prediction_high": high,
        "zipcode": request.form.get("zipcode"),
        "bedrooms": request.form.get("bedrooms"),
        "bathrooms": request.form.get("bathrooms"),
        "sqft_living": request.form.get("sqft_living"),
    }
    try:
        pd.DataFrame([lead_row]).to_csv(LEADS_PATH, mode="a", header=False, index=False)
    except Exception:
        # Non-fatal if writing fails
        pass

    # Render result
    return render_template(
        "result.html",
        name=name,
        intent=intent,
        timeline=timeline,
        y_pred=y_pred,
        y_low=low,
        y_high=high,
        fmt=_format_currency,
        similars=similars,
    )


@app.route("/debug/model")
def debug_model():
    try:
        loaded_type = type(model).__name__ if model is not None else "None"
        info = {
            "resolved_model_type": loaded_type,
            "has_named_steps": hasattr(model, "named_steps"),
            "has_feature_names_in": hasattr(model, "feature_names_in_"),
            "feature_names_count": len(feature_names or []),
            "first_10_features": (feature_names or [])[:10],
            "calibrator": {
                "mode": _calibrator[0] if _calibrator else None,
                "a": _calibrator[1] if _calibrator else None,
                "b": _calibrator[2] if _calibrator else None,
            }
        }
        if isinstance(model, dict):
            info["dict_keys"] = list(model.keys())
        # Include metrics snapshot if available
        metrics = compute_metrics()
        if metrics:
            info["metrics"] = metrics
        return {"ok": True, "info": info}
    except Exception as e:
        return {"ok": False, "error": str(e)}, 500


@app.route("/about")
def about():
    metrics = compute_metrics()
    return render_template("about.html", metrics=metrics, fmt=_format_currency)


# Static serving for required folders
@app.route('/css/<path:filename>')
def css(filename: str):
    return send_from_directory('css', filename)


@app.route('/js/<path:filename>')
def js(filename: str):
    return send_from_directory('js', filename)


@app.route('/assets/<path:filename>')
def assets(filename: str):
    # Serve assets or raise 404 with friendly message
    try:
        return send_from_directory('assets', filename)
    except NotFound:
        return "Asset not found", 404


if __name__ == "__main__":
    # Load resources once and start the app
    load_resources()
    try:
        print("[House Maxify] Resolved model:", type(model).__name__)
        print("[House Maxify] Feature names count:", len(feature_names or []))
        if feature_names:
            print("[House Maxify] First features:", (feature_names or [])[:10])
    except Exception:
        pass
    # How to run:
    #   python app.py
    # Access in browser:
    #   http://localhost:5000 (local) or http://<your-ip>:5000 (LAN)
    app.run(host="0.0.0.0", port=5000, debug=True)
