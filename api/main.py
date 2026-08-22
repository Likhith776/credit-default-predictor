"""Thin FastAPI service exposing the trained credit-default model.

Endpoints
---------
GET  /health   — warm-up / liveness ping (also reports model metadata)
POST /predict  — score one application, return probability, risk tier,
                 and the top SHAP factors for that specific prediction

Run from the repo root:
    pip install -r api/requirements.txt
    uvicorn api.main:app --host 0.0.0.0 --port 8000

The feature-vector logic mirrors app/streamlit_app.py (median-filled vector,
user overrides, sanitized feature names) but is standalone so the frontend
never depends on Streamlit.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"

EDUCATION_LEVELS = [
    "Secondary/secondary special",
    "Higher education",
    "Incomplete higher",
    "Lower secondary",
    "Academic degree",
]
CONTRACT_TYPES = ["Cash loans", "Revolving loans"]

# Frontend origins allowed to call the API. The deployed Vercel/Netlify URL
# should be added here (or set via the API_ORIGINS env var, comma-separated).
ALLOWED_ORIGINS = [
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    "http://localhost:3000",
    "http://localhost:8080",
    "http://localhost:8000",
]


def clean_name(name: str) -> str:
    """Same sanitization train.py applies (LightGBM rejects JSON special chars)."""
    return re.sub(r"[^A-Za-z0-9_]+", "_", name)


def _load_artifact(path: Path):
    """joblib-load a model artifact, with a clear error if Git LFS didn't smudge it.

    Render/Railway clones sometimes fetch LFS files as ~130-byte text pointers;
    joblib would fail with a confusing "invalid load key" — catch it here with
    an actionable message instead.
    """
    if path.exists() and path.stat().st_size < 1024:
        head = path.read_bytes()[:64]
        if head.startswith(b"version https://git-lfs"):
            raise RuntimeError(
                f"{path.name} is a Git LFS pointer, not the real artifact. "
                f"Make sure the deploy environment fetches LFS objects "
                f"(e.g. run `git lfs install && git lfs pull` in the build, "
                f"or download the file from the GitHub release/raw media URL)."
            )
    return joblib.load(path)


# ------------------------------------------------------------------ startup

app = FastAPI(title="Credit Default Predictor API", version="1.0.0")

_origins = ALLOWED_ORIGINS + [o for o in __import__("os")
                              .environ.get("API_ORIGINS", "").split(",") if o]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

model = _load_artifact(MODEL_DIR / "lgb_model.pkl")
shap_explainer = _load_artifact(MODEL_DIR / "shap_explainer.pkl")
FEATURE_NAMES: list[str] = json.loads((MODEL_DIR / "feature_names.json").read_text())
MEDIANS: dict[str, float] = json.loads((MODEL_DIR / "training_medians.json").read_text())
METRICS: dict = json.loads((MODEL_DIR / "model_metrics.json").read_text())


# ------------------------------------------------------------------ schemas

class ApplicantInput(BaseModel):
    income: float = Field(135_000, ge=25_000, le=2_000_000,
                          description="Annual income")
    credit: float = Field(600_000, ge=50_000, le=3_000_000,
                          description="Loan / credit amount")
    annuity: float = Field(27_000, ge=5_000, le=200_000,
                           description="Loan annuity (yearly payment)")
    employed_years: float = Field(5.0, ge=0.0, le=40.0,
                                  description="Employment duration in years")
    age: int = Field(35, ge=18, le=70, description="Age in years")
    family_members: int = Field(2, ge=1, le=10, description="Family size")
    education: Literal[tuple(EDUCATION_LEVELS)] = Field(  # type: ignore[valid-type]
        "Secondary/secondary special")
    contract_type: Literal[tuple(CONTRACT_TYPES)] = Field(  # type: ignore[valid-type]
        "Cash loans")


# ------------------------------------------------------------------ helpers

def build_feature_vector(inp: ApplicantInput) -> pd.DataFrame:
    """Median-filled vector with the applicant's values overridden.

    Same construction as the Streamlit app: everything defaults to the
    training medians, then the raw fields, domain ratios, and one-hot
    columns for education / contract type are set from the input.
    """
    vec = pd.Series(MEDIANS, dtype=float).reindex(FEATURE_NAMES).fillna(0.0)

    vec["AMT_INCOME_TOTAL"] = float(inp.income)
    vec["AMT_CREDIT"] = float(inp.credit)
    vec["AMT_ANNUITY"] = float(inp.annuity)
    vec["DAYS_BIRTH"] = float(-inp.age * 365)
    vec["DAYS_EMPLOYED"] = float(-inp.employed_years * 365)
    vec["CNT_FAM_MEMBERS"] = float(inp.family_members)

    # Formulas must match _app_domain_features in src/features/feature_engineering.py
    vec["CREDIT_INCOME_RATIO"] = inp.credit / (inp.income + 1)
    vec["ANNUITY_INCOME_RATIO"] = inp.annuity / (inp.income + 1)
    vec["CREDIT_TERM"] = inp.credit / (inp.annuity + 1)
    vec["DAYS_EMPLOYED_RATIO"] = (inp.employed_years * 365) / (inp.age * 365)
    vec["INCOME_PER_PERSON"] = inp.income / (inp.family_members + 1)
    vec["EXT_SOURCE_MEAN"] = 0.5
    if "EXT_SOURCE_MIN" in vec:
        vec["EXT_SOURCE_MIN"] = 0.5

    for col in FEATURE_NAMES:
        if col.startswith(("NAME_EDUCATION_TYPE_", "NAME_CONTRACT_TYPE_")):
            vec[col] = 0.0
    edu_col = "NAME_EDUCATION_TYPE_" + clean_name(inp.education)
    if edu_col in vec:
        vec[edu_col] = 1.0
    ct_col = "NAME_CONTRACT_TYPE_" + clean_name(inp.contract_type)
    if ct_col in vec:
        vec[ct_col] = 1.0

    return vec.to_frame().T


def top_shap_factors(explainer, X: pd.DataFrame, k: int = 8) -> list[dict]:
    sv = explainer.shap_values(X)
    if isinstance(sv, list):  # older shap: [class0, class1]
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:  # newer shap: (rows, features, classes)
        sv = sv[0, :, 1]
    sv = sv.reshape(-1)
    order = np.abs(sv).argsort()[::-1][:k]
    return [
        {
            "feature": FEATURE_NAMES[i],
            "value": float(X.iloc[0, i]),
            "impact": float(sv[i]),
        }
        for i in order
    ]


def risk_tier(probability: float) -> str:
    if probability < 0.20:
        return "LOW"
    if probability < 0.50:
        return "MODERATE"
    return "HIGH"


# ------------------------------------------------------------------ routes

@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model": "LightGBM",
        "n_features": len(FEATURE_NAMES),
        "cv_auc": METRICS.get("cv_auc_mean"),
    }


@app.post("/predict")
def predict(inp: ApplicantInput) -> dict:
    X = build_feature_vector(inp)
    probability = float(model.predict_proba(X)[0, 1])
    return {
        "probability": probability,
        "risk_tier": risk_tier(probability),
        "shap_top_factors": top_shap_factors(shap_explainer, X),
    }
