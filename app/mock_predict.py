"""Mock predictor — lets the Streamlit app run before a model is trained.

Generates a plausible-looking default probability from a handful of
application fields with fixed coefficients and noise, so the UI can be
developed and demoed without models/lgbm_model.pkl present.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def mock_predict_proba(app_row: pd.Series | dict) -> float:
    """Return a pseudo-probability of default for one application row."""
    row = pd.Series(app_row)

    credit_income = row.get("AMT_CREDIT", 5e5) / (row.get("AMT_INCOME_TOTAL", 1.5e5) + 1e-6)
    annuity_income = row.get("AMT_ANNUITY", 2.5e4) / (row.get("AMT_INCOME_TOTAL", 1.5e5) + 1e-6)
    days_birth = row.get("DAYS_BIRTH", -40 * 365)
    days_employed = row.get("DAYS_EMPLOYED", -5 * 365)
    if days_employed == 365243:  # pensioner sentinel
        days_employed = -10 * 365
    age_years = -days_birth / 365
    employed_years = -days_employed / 365

    z = (
        0.35 * credit_income
        + 2.0 * annuity_income
        - 0.02 * age_years
        - 0.03 * employed_years
        - 1.5
        + np.random.default_rng(42).normal(0, 0.15)
    )
    return float(1 / (1 + np.exp(-z)))  # sigmoid


if __name__ == "__main__":
    sample = {"AMT_CREDIT": 500_000, "AMT_INCOME_TOTAL": 150_000,
              "AMT_ANNUITY": 25_000, "DAYS_BIRTH": -14_600, "DAYS_EMPLOYED": -1_825}
    print(f"Mock default probability: {mock_predict_proba(sample):.4f}")
