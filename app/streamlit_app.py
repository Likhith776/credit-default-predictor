"""Streamlit app: Credit Default Predictor (portfolio demo).

Serves single-applicant default-risk predictions from the trained LightGBM
model, with per-applicant SHAP explanations and global model insights.

Run from the project root:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

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


# ---------------------------------------------------------------- loading

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_DIR / "lgb_model.pkl")
    explainer = joblib.load(MODEL_DIR / "shap_explainer.pkl")
    feature_names = json.loads((MODEL_DIR / "feature_names.json").read_text())
    metrics = json.loads((MODEL_DIR / "model_metrics.json").read_text())
    medians = json.loads((MODEL_DIR / "training_medians.json").read_text())
    return model, explainer, feature_names, metrics, medians


def clean_name(name: str) -> str:
    """Same sanitization train.py applies (LightGBM rejects JSON special chars)."""
    return re.sub(r"[^A-Za-z0-9_]+", "_", name)


# ---------------------------------------------------------------- helpers

def build_feature_vector(inputs, feature_names, medians):
    """Median-filled vector, overridden with the user's application inputs."""
    vec = pd.Series(medians, dtype=float).reindex(feature_names).fillna(0.0)

    income = inputs["income"]
    credit = inputs["credit"]
    annuity = inputs["annuity"]
    age = inputs["age"]
    employed = inputs["employed"]
    family = inputs["family"]

    # Raw application fields (stored as negative days in the dataset)
    vec["AMT_INCOME_TOTAL"] = float(income)
    vec["AMT_CREDIT"] = float(credit)
    vec["AMT_ANNUITY"] = float(annuity)
    vec["DAYS_BIRTH"] = float(-age * 365)
    vec["DAYS_EMPLOYED"] = float(-employed * 365)
    vec["CNT_FAM_MEMBERS"] = float(family)

    # Engineered domain features — formulas must match _app_domain_features
    # in src/features/feature_engineering.py exactly
    vec["CREDIT_INCOME_RATIO"] = credit / (income + 1)
    vec["ANNUITY_INCOME_RATIO"] = annuity / (income + 1)
    vec["CREDIT_TERM"] = credit / (annuity + 1)
    vec["DAYS_EMPLOYED_RATIO"] = (employed * 365) / (age * 365)
    vec["INCOME_PER_PERSON"] = income / (family + 1)
    vec["EXT_SOURCE_MEAN"] = 0.5
    if "EXT_SOURCE_MIN" in vec:
        vec["EXT_SOURCE_MIN"] = 0.5

    # One-hot columns (drop_first=True in training: Cash loans and
    # Academic degree are the dropped baselines -> all-zero for those)
    for col in feature_names:
        if col.startswith("NAME_EDUCATION_TYPE_"):
            vec[col] = 0.0
        if col.startswith("NAME_CONTRACT_TYPE_"):
            vec[col] = 0.0
    edu_col = "NAME_EDUCATION_TYPE_" + clean_name(inputs["education"])
    if edu_col in vec:
        vec[edu_col] = 1.0
    ct_col = "NAME_CONTRACT_TYPE_" + clean_name(inputs["contract"])
    if ct_col in vec:
        vec[ct_col] = 1.0

    return vec


def risk_gauge(prob: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={"suffix": "%", "font": {"size": 60}},
        title={"text": "Probability of Default", "font": {"size": 22}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "white"},
            "bar": {"color": "white", "thickness": 0.25},
            "steps": [
                {"range": [0, 20], "color": "#2ecc71"},
                {"range": [20, 50], "color": "#f1c40f"},
                {"range": [50, 100], "color": "#e74c3c"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 4},
                "thickness": 0.9,
                "value": prob * 100,
            },
        },
    ))
    fig.update_layout(height=320, margin=dict(t=60, b=10, l=30, r=30))
    return fig


def shap_waterfall(explainer, vec: pd.DataFrame) -> go.Figure:
    """Top-8 |SHAP| features for THIS applicant, red = pushes risk up."""
    sv = explainer.shap_values(vec)
    if isinstance(sv, list):        # older shap: [class0, class1]
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:                # newer shap: (rows, features, classes)
        sv = sv[0, :, 1]
    sv = sv.reshape(-1)

    s = pd.Series(sv, index=vec.columns)
    top = s.abs().sort_values(ascending=False).head(8).index
    top = s.loc[top].sort_values()

    colors = ["#e74c3c" if v > 0 else "#3498db" for v in top.values]
    labels = [
        f"{feat}<br><span style='font-size:11px;color:#95a5a6'>"
        f"value: {vec[feat].iloc[0]:,.3g}</span>"
        for feat in top.index
    ]
    fig = go.Figure(go.Bar(
        x=top.values, y=labels, orientation="h",
        marker_color=colors,
        text=[f"{v:+.3f}" for v in top.values],
        textposition="outside",
    ))
    fig.update_layout(
        title="Why this score — top 8 factors for this applicant",
        height=420, margin=dict(l=20, r=20, t=50, b=20),
        xaxis_title="SHAP value (impact on default risk)",
        yaxis=dict(autorange="reversed"),
    )
    return fig


# ---------------------------------------------------------------- pages

def tab_predict(model, explainer, feature_names, medians):
    col_in, col_out = st.columns([1, 1.4])

    with col_in:
        st.subheader("📝 Applicant Details")
        income = st.number_input("Annual Income ($)", 25_000, 2_000_000, 135_000, step=1_000)
        credit = st.number_input("Loan Amount ($)", 50_000, 3_000_000, 600_000, step=10_000)
        annuity = st.number_input("Loan Annuity ($/yr)", 5_000, 200_000, 27_000, step=1_000)
        employed = st.slider("Employment Duration (years)", 0.0, 40.0, 5.0, 0.5)
        age = st.slider("Age (years)", 18, 70, 35)
        family = st.selectbox("Family Members", list(range(1, 11)), index=1)
        education = st.selectbox("Education", EDUCATION_LEVELS)
        contract = st.selectbox("Contract Type", CONTRACT_TYPES)
        assess = st.button("Assess Default Risk", type="primary", use_container_width=True)

    if not assess:
        col_out.info("Fill in the applicant details and click **Assess Default Risk**.")
        return

    inputs = dict(income=income, credit=credit, annuity=annuity, age=age,
                  employed=employed, family=family, education=education,
                  contract=contract)
    vec = build_feature_vector(inputs, feature_names, medians)
    X = vec.to_frame().T
    prob = float(model.predict_proba(X)[0, 1])

    with col_out:
        st.plotly_chart(risk_gauge(prob), use_container_width=True)

        if prob < 0.20:
            st.success("✓ **LOW RISK** — Strong repayment likelihood")
        elif prob < 0.50:
            st.warning("⚠ **MODERATE RISK** — Requires further assessment")
        else:
            st.error("✗ **HIGH RISK** — Likely default")

        st.plotly_chart(shap_waterfall(explainer, X), use_container_width=True)
        st.caption("These are the factors that moved this specific applicant's "
                   "score, not just the model's overall top features.")


def tab_insights(metrics):
    st.subheader("📊 Model Performance")
    c1, c2, c3 = st.columns(3)
    c1.metric("CV AUC (5-fold)", f"{metrics['cv_auc_mean']:.4f}",
              f"± {metrics['cv_auc_std']:.4f}")
    c2.metric("Gini Coefficient", f"{metrics['cv_gini_mean']:.4f}")
    c3.metric("Validation AUC", f"{metrics['val_auc']:.4f}")
    st.caption(f"Trained with LightGBM on {metrics['n_features']:,} features; "
               f"best iteration ≈ {metrics['best_iteration']} boosting rounds "
               f"(early stopping over 5-fold stratified CV).")

    st.subheader("What Drives the Model")
    left, right = st.columns(2)
    with left:
        st.markdown("**Feature importance (gain)** — which features the model "
                    "split on most, weighted by improvement.")
        st.image(str(MODEL_DIR / "feature_importance.png"))
    with right:
        st.markdown("**SHAP summary** — magnitude **and** direction of each "
                    "feature's effect across many applicants (e.g. low "
                    "EXT_SOURCE_MEAN pushes risk up), unlike the gain chart.")
        st.image(str(MODEL_DIR / "shap_summary.png"))

    st.subheader("ROC Curve — Reading the AUC")
    st.markdown(
        f"""
        An AUC of **{metrics['val_auc']:.3f}** means that for a randomly chosen
        defaulting applicant and a randomly chosen repaying applicant, the model
        ranks the defaulter as riskier **{metrics['val_auc']:.1%}** of the time
        (50% = random guessing, 100% = perfect). The Gini coefficient
        (**{metrics['val_gini']:.3f}** = 2 × AUC − 1) is the same signal on the
        scale lenders conventionally use — values above 0.5 are considered
        strong for application-scorecard style models.
        """
    )


def tab_about():
    st.subheader("About This Project")
    st.markdown(
        """
        This app demonstrates an end-to-end credit default risk pipeline built
        on the Kaggle **Home Credit Default Risk** dataset. Seven relational
        tables (applications, bureau histories, previous applications,
        installments, POS and credit-card balances) are aggregated into
        engineered features per applicant, and a LightGBM classifier is trained
        with 5-fold cross-validation and early stopping.

        The focus is **explainability**: global insights come from gain-based
        feature importance and SHAP summary plots, while the Predict tab
        computes per-applicant SHAP values so every individual score shows
        exactly which factors pushed it up or down.
        """
    )
    st.markdown(
        """
        **Dataset:** [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) (Kaggle)

        **Tech stack:** Python, LightGBM, Streamlit, Pandas, NumPy,
        Scikit-learn, Plotly, SHAP

        **GitHub:** [Likhith776 — credit-default-predictor](https://github.com/Likhith776/credit-default-predictor)

        Model trained on **307,511 loan applications** with **190+ engineered
        features**.
        """
    )
    st.markdown("---")
    st.markdown(
        """
        **Limitations.** The 20% / 50% risk thresholds shown here are
        illustrative, not calibrated to any real business cost of default vs.
        rejected revenue. The model has not been evaluated for disparate
        impact across protected attributes, and it should not be treated as a
        real lending decision tool — this is a portfolio demonstration of the
        modeling and explainability pipeline only.
        """
    )


# ---------------------------------------------------------------- main

def main() -> None:
    st.set_page_config(page_title="Credit Default Predictor",
                       page_icon="🏦", layout="wide")

    st.markdown("""
        <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            .stTabs [data-baseweb="tab-list"] {gap: 8px;}
            .stTabs [data-baseweb="tab"] {
                font-size: 17px; font-weight: 600; padding: 10px 24px;
                border-radius: 8px 8px 0 0;
            }
            h1, h2, h3, .stMetricLabel {
                font-family: 'Segoe UI', 'Inter', sans-serif;
            }
        </style>
    """, unsafe_allow_html=True)

    st.title("🏦 Credit Default Predictor")
    st.caption("Home Credit Default Risk · LightGBM + SHAP · portfolio demo "
               "by Likhith776")

    model, explainer, feature_names, metrics, medians = load_artifacts()

    t1, t2, t3 = st.tabs(["🎯 Predict", "📊 Model Insights", "ℹ️ About"])
    with t1:
        tab_predict(model, explainer, feature_names, medians)
    with t2:
        tab_insights(metrics)
    with t3:
        tab_about()


if __name__ == "__main__":
    main()
