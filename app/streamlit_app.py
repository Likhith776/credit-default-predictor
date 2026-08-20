"""Streamlit app: Credit Default Predictor.

Serves default-risk predictions for a single application. Uses the
trained LightGBM model from models/ when available; otherwise falls
back to app/mock_predict.py so the UI works before training.

Run:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project root

from app.mock_predict import mock_predict_proba  # noqa: E402

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "lgbm_model.pkl"
FEATURE_NAMES_PATH = MODEL_PATH.parent / "feature_names.json"


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    import joblib

    return joblib.load(MODEL_PATH)


def main() -> None:
    st.set_page_config(page_title="Credit Default Predictor", page_icon="🏦", layout="wide")
    st.title("🏦 Credit Default Predictor")
    st.caption("Home Credit Default Risk — will this applicant repay their loan?")

    model = load_model()
    if model is None:
        st.info("No trained model found — showing **mock predictions**. "
                "Run `python -m src.models.train` to train the real model.")

    with st.form("application_form"):
        st.subheader("Applicant details")
        col1, col2 = st.columns(2)
        with col1:
            income = st.number_input("Annual income", 10_000, 5_000_000, 150_000, step=10_000)
            credit = st.number_input("Credit amount", 10_000, 5_000_000, 500_000, step=10_000)
            annuity = st.number_input("Annuity", 1_000, 500_000, 25_000, step=1_000)
        with col2:
            age = st.slider("Age (years)", 18, 90, 40)
            employed = st.slider("Years employed", 0, 50, 5)
        submitted = st.form_submit_button("Predict", type="primary")

    if submitted:
        row = pd.Series({
            "AMT_INCOME_TOTAL": income,
            "AMT_CREDIT": credit,
            "AMT_ANNUITY": annuity,
            "DAYS_BIRTH": -age * 365,
            "DAYS_EMPLOYED": -employed * 365,
        })

        if model is not None:
            import json

            feature_names = json.loads(FEATURE_NAMES_PATH.read_text())
            # Fill the full feature vector the model expects; simple demo
            # inputs mean most engineered features default to 0/NaN.
            X = pd.DataFrame([row.reindex(feature_names).fillna(0)]).astype(float)
            proba = float(model.predict_proba(X)[0, 1])
            source = "trained LightGBM model"
        else:
            proba = mock_predict_proba(row)
            source = "mock predictor"

        risk = "🔴 HIGH" if proba >= 0.5 else "🟢 LOW" if proba < 0.2 else "🟡 MEDIUM"
        st.metric("Probability of default", f"{proba:.1%}")
        st.write(f"Risk band: **{risk}**  ·  scored by: {source}")

        st.progress(min(proba, 1.0))


if __name__ == "__main__":
    main()
