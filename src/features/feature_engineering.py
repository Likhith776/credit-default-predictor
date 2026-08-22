"""Feature engineering for the Home Credit Default Risk dataset.

Five aggregation functions collapse the auxiliary tables to one row per
SK_ID_CURR, and ``build_features`` merges everything onto the application
table, adds domain ratios, one-hot encodes categoricals, and cleans
infinite/missing values.

Run directly to build the training table:

    python -m src.features.feature_engineering
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# bureau_balance.STATUS: C=closed, X=no debt, 0..5 = days past due buckets.
STATUS_TO_DPD = {"C": 0, "X": 0, "0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5}


def agg_bureau(bureau_df: pd.DataFrame, bureau_balance_df: pd.DataFrame) -> pd.DataFrame:
    """Credit Bureau aggregates per SK_ID_CURR, including monthly DPD stats."""
    bur = bureau_df.copy()

    feats = pd.DataFrame(index=bur.groupby("SK_ID_CURR").size().index)
    feats["BUREAU_CREDIT_COUNT"] = bur.groupby("SK_ID_CURR").size()
    feats["BUREAU_ACTIVE_COUNT"] = (
        bur.assign(_a=(bur["CREDIT_ACTIVE"] == "Active").astype(int))
        .groupby("SK_ID_CURR")["_a"].sum()
    )
    feats["BUREAU_CLOSED_COUNT"] = (
        bur.assign(_c=(bur["CREDIT_ACTIVE"] == "Closed").astype(int))
        .groupby("SK_ID_CURR")["_c"].sum()
    )
    feats["BUREAU_AMT_CREDIT_SUM_MEAN"] = bur.groupby("SK_ID_CURR")["AMT_CREDIT_SUM"].mean()
    feats["BUREAU_AMT_CREDIT_SUM_MAX"] = bur.groupby("SK_ID_CURR")["AMT_CREDIT_SUM"].max()
    feats["BUREAU_DAYS_CREDIT_MEAN"] = bur.groupby("SK_ID_CURR")["DAYS_CREDIT"].mean()

    # Monthly balance history: map STATUS -> dpd int, then per-loan -> per-client
    bb = bureau_balance_df.copy()
    bb["DPD"] = bb["STATUS"].map(STATUS_TO_DPD)
    bb["OVERDUE"] = (~bb["STATUS"].isin(["C", "X", "0"])).astype(int)
    loan_dpd = bb.groupby("SK_ID_BUREAU").agg(DPD_MEAN=("DPD", "mean"), DPD_MAX=("DPD", "max"))
    loan_overdue = bb.groupby("SK_ID_BUREAU")["OVERDUE"].sum().rename("OVERDUE_COUNT")

    bur_bb = bur[["SK_ID_CURR", "SK_ID_BUREAU"]].merge(
        loan_dpd.join(loan_overdue), on="SK_ID_BUREAU", how="left"
    )
    feats["BUREAU_DPD_MEAN"] = bur_bb.groupby("SK_ID_CURR")["DPD_MEAN"].mean()
    feats["BUREAU_DPD_MAX"] = bur_bb.groupby("SK_ID_CURR")["DPD_MAX"].max()
    feats["BUREAU_OVERDUE_COUNT"] = bur_bb.groupby("SK_ID_CURR")["OVERDUE_COUNT"].sum()

    return feats


def agg_previous_applications(prev_df: pd.DataFrame) -> pd.DataFrame:
    """Previous Home Credit application aggregates per SK_ID_CURR."""
    prev = prev_df.copy()
    g = prev.groupby("SK_ID_CURR")

    feats = pd.DataFrame(index=g.size().index)
    feats["PREV_APP_COUNT"] = g.size()
    feats["PREV_APPROVED_COUNT"] = (
        prev.assign(_a=(prev["NAME_CONTRACT_STATUS"] == "Approved").astype(int))
        .groupby("SK_ID_CURR")["_a"].sum()
    )
    feats["PREV_REFUSED_COUNT"] = (
        prev.assign(_r=(prev["NAME_CONTRACT_STATUS"] == "Refused").astype(int))
        .groupby("SK_ID_CURR")["_r"].sum()
    )
    feats["PREV_APPROVAL_RATE"] = feats["PREV_APPROVED_COUNT"] / feats["PREV_APP_COUNT"]
    feats["PREV_AMT_APPLICATION_MEAN"] = g["AMT_APPLICATION"].mean()
    feats["PREV_AMT_CREDIT_MEAN"] = g["AMT_CREDIT"].mean()
    feats["PREV_AMT_DOWN_PAYMENT_MEAN"] = g["AMT_DOWN_PAYMENT"].mean()
    feats["PREV_DAYS_LAST_DUE_MEAN"] = g["DAYS_LAST_DUE"].mean()

    return feats


def agg_installments(install_df: pd.DataFrame) -> pd.DataFrame:
    """Installment payment history aggregates per SK_ID_CURR."""
    inst = install_df.copy()
    inst["PAYMENT_RATIO"] = (
        inst["AMT_PAYMENT"] / inst["AMT_INSTALMENT"]
    ).clip(0, 2)
    inst["LATE"] = (inst["DAYS_ENTRY_PAYMENT"] > inst["DAYS_INSTALMENT"] + 1).astype(int)
    inst["DPD"] = (inst["DAYS_ENTRY_PAYMENT"] - inst["DAYS_INSTALMENT"]).clip(lower=0)

    g = inst.groupby("SK_ID_CURR")
    feats = pd.DataFrame(index=g.size().index)
    feats["INSTAL_PAYMENT_RATIO_MEAN"] = g["PAYMENT_RATIO"].mean()
    feats["INSTAL_LATE_COUNT"] = g["LATE"].sum()
    feats["INSTAL_LATE_RATIO"] = g["LATE"].mean()
    feats["INSTAL_DPD_MEAN"] = g["DPD"].mean()
    feats["INSTAL_DPD_MAX"] = g["DPD"].max()

    return feats


def agg_pos_cash(pos_df: pd.DataFrame) -> pd.DataFrame:
    """POS/cash loan monthly balance aggregates per SK_ID_CURR."""
    pos = pos_df.copy()
    g = pos.groupby("SK_ID_CURR")

    feats = pd.DataFrame(index=g.size().index)
    feats["POS_MONTHS_BALANCE_MEAN"] = g["MONTHS_BALANCE"].mean()
    feats["POS_CNT_INSTALMENT_FUTURE_MEAN"] = g["CNT_INSTALMENT_FUTURE"].mean()
    feats["POS_SK_DPD_MEAN"] = g["SK_DPD"].mean()
    feats["POS_SK_DPD_DEF_MAX"] = g["SK_DPD_DEF"].max()
    feats["POS_COMPLETED_COUNT"] = (
        pos.assign(_c=(pos["NAME_CONTRACT_STATUS"] == "Completed").astype(int))
        .groupby("SK_ID_CURR")["_c"].sum()
    )

    return feats


def agg_credit_card(cc_df: pd.DataFrame) -> pd.DataFrame:
    """Credit card monthly balance aggregates per SK_ID_CURR."""
    cc = cc_df.copy()
    cc["UTILIZATION"] = cc["AMT_BALANCE"] / (cc["AMT_CREDIT_LIMIT_ACTUAL"] + 1)
    g = cc.groupby("SK_ID_CURR")

    feats = pd.DataFrame(index=g.size().index)
    feats["CC_UTILIZATION_MEAN"] = g["UTILIZATION"].mean()
    feats["CC_AMT_BALANCE_MAX"] = g["AMT_BALANCE"].max()
    # AMT_DRAWINGS_CURRENT is this dataset's monthly total drawing amount
    # (there is no AMT_DRAWINGS_TOTAL column).
    feats["CC_AMT_DRAWINGS_TOTAL_MEAN"] = g["AMT_DRAWINGS_CURRENT"].mean()
    feats["CC_SK_DPD_MEAN"] = g["SK_DPD"].mean()

    return feats


def _app_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """Domain ratios and EXT_SOURCE combos derived from application columns."""
    df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / (df["AMT_INCOME_TOTAL"] + 1)
    df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / (df["AMT_INCOME_TOTAL"] + 1)
    df["CREDIT_TERM"] = df["AMT_CREDIT"] / (df["AMT_ANNUITY"] + 1)
    df["DAYS_EMPLOYED_RATIO"] = df["DAYS_EMPLOYED"] / (df["DAYS_BIRTH"] + 1)
    df["INCOME_PER_PERSON"] = df["AMT_INCOME_TOTAL"] / (df["CNT_FAM_MEMBERS"] + 1)

    ext = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]]
    ext = ext.fillna(ext.mean())  # column means, then row-wise stats
    df["EXT_SOURCE_MEAN"] = ext.mean(axis=1)
    df["EXT_SOURCE_MIN"] = ext.min(axis=1)
    return df


def build_features(
    app_df: pd.DataFrame,
    bureau_df: pd.DataFrame,
    bb_df: pd.DataFrame,
    prev_df: pd.DataFrame,
    inst_df: pd.DataFrame,
    cc_df: pd.DataFrame,
    pos_df: pd.DataFrame,
) -> pd.DataFrame:
    """Assemble the final modeling table keyed by SK_ID_CURR."""
    df = app_df.copy()

    agg_tables = {
        "bureau": lambda: agg_bureau(bureau_df, bb_df),
        "prev": lambda: agg_previous_applications(prev_df),
        "inst": lambda: agg_installments(inst_df),
        "pos": lambda: agg_pos_cash(pos_df),
        "cc": lambda: agg_credit_card(cc_df),
    }
    for name, build in agg_tables.items():
        print(f"[features] aggregating {name}...", flush=True)
        feats = build()
        df = df.merge(feats.reset_index(), on="SK_ID_CURR", how="left")
    print("[features] merging done", flush=True)

    df = _app_domain_features(df)
    obj_cols = df.select_dtypes(include=["object", "str"]).columns.tolist()
    df = pd.get_dummies(df, columns=obj_cols, drop_first=True)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(-999)

    return df


if __name__ == "__main__":
    import argparse

    from src.data_loader import load_all_data

    parser = argparse.ArgumentParser(description="Build the engineered feature table")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    args = parser.parse_args()

    data = load_all_data()
    app_key = "application" if args.split == "train" else "application_test"
    features = build_features(
        app_df=data[app_key],
        bureau_df=data["bureau"],
        bb_df=data["bureau_balance"],
        prev_df=data["previous_application"],
        inst_df=data["installments"],
        cc_df=data["credit_card"],
        pos_df=data["pos_cash"],
    )
    out = f"data/processed/{args.split}_features.parquet"
    features.to_parquet(out, index=False)
    n_id = 2 if "TARGET" in features.columns else 1  # SK_ID_CURR (+ TARGET)
    n_features = features.shape[1] - n_id
    print(f"\nSaved {out}")
    print(f"shape: {features.shape[0]:,} rows x {features.shape[1]} columns ({n_features} features)")
