"""Feature engineering for the Home Credit Default Risk dataset.

Builds a single modeling table keyed by SK_ID_CURR:
  1. Domain features on the application table (ratios, aggregates).
  2. Grouped aggregates of the auxiliary tables (bureau, POS, credit
     card, installments, previous applications).

Usage
-----
    from src.data_loader import load_all_data
    from src.features.feature_engineering import build_features

    data = load_all_data()
    train_df = build_features(data.pop("application"), data)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

AGG_FUNCS = ["mean", "max", "min", "sum", "std"]


def add_domain_features(app: pd.DataFrame) -> pd.DataFrame:
    """Add ratio/aggregate features derived from application columns."""
    df = app.copy()
    df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / (df["AMT_INCOME_TOTAL"] + 1e-6)
    df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / (df["AMT_INCOME_TOTAL"] + 1e-6)
    df["CREDIT_TERM"] = df["AMT_ANNUITY"] / (df["AMT_CREDIT"] + 1e-6)
    df["DAYS_EMPLOYED_RATIO"] = df["DAYS_EMPLOYED"] / (df["DAYS_BIRTH"] + 1e-6)
    df["INCOME_PER_PERSON"] = df["AMT_INCOME_TOTAL"] / df["CNT_FAM_MEMBERS"].clip(lower=1)
    # Flag the sentinel value 365243 used for pensioners in DAYS_EMPLOYED.
    df["DAYS_EMPLOYED_ANOMALY"] = (df["DAYS_EMPLOYED"] == 365243).astype(int)
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)
    return df


def aggregate_auxiliary(aux: pd.DataFrame, key: str, prefix: str) -> pd.DataFrame:
    """Collapse an auxiliary table to one row per key via groupby.agg."""
    numeric_cols = aux.select_dtypes("number").columns.drop(key, errors="ignore")
    grouped = aux.groupby(key)[list(numeric_cols)].agg(AGG_FUNCS)
    grouped.columns = [f"{prefix}_{col.upper()}_{stat.upper()}" for col, stat in grouped.columns]
    return grouped.reset_index()


def build_features(app: pd.DataFrame, aux_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Combine the application table with auxiliary aggregates.

    Parameters
    ----------
    app : main application table (train or test).
    aux_tables : mapping of table name -> DataFrame (bureau, pos_cash,
        installments, credit_card, previous_application). Missing tables
        are skipped so the same code works with partial inputs.
    """
    df = add_domain_features(app)

    merge_specs = {
        "bureau": ("SK_ID_CURR", "BUR"),
        "previous_application": ("SK_ID_CURR", "PREV"),
        "pos_cash": ("SK_ID_CURR", "POS"),
        "installments": ("SK_ID_CURR", "INST"),
        "credit_card": ("SK_ID_CURR", "CC"),
    }
    for name, (key, prefix) in merge_specs.items():
        aux = aux_tables.get(name)
        if aux is None or key not in aux.columns:
            print(f"[features] skipping '{name}' (not provided)")
            continue
        df = df.merge(aggregate_auxiliary(aux, key, prefix), on=key, how="left")

    return df
