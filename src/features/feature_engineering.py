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


# ---------------------------------------------------------------------------
# Trend / velocity aggregators — capture direction of change, not just level
# (MONTHS_BALANCE is negative and relative to application date, so the
# largest values are the most recent months).
# ---------------------------------------------------------------------------

def _recent_vs_full(sorted_df: pd.DataFrame, group_col: str, col: str,
                    prefix: str) -> pd.DataFrame:
    """Per-group last-3-month mean, full-history mean, trend, and MoM diff.

    trend = last3_mean - full_mean  (positive => worsening for DPD/util)
    diff  = month-over-month change of the series, aggregated mean/max
    """
    g = sorted_df.groupby(group_col)
    feats = pd.DataFrame(index=g.size().index)
    feats[f"{prefix}_MEAN"] = g[col].mean()
    recent = g.tail(3).groupby(group_col)[col].mean()
    feats[f"{prefix}_LAST3_MEAN"] = recent
    feats[f"{prefix}_TREND"] = feats[f"{prefix}_LAST3_MEAN"] - feats[f"{prefix}_MEAN"]
    diffs = g[col].diff()
    feats[f"{prefix}_DIFF_MEAN"] = diffs.groupby(sorted_df[group_col]).mean()
    feats[f"{prefix}_DIFF_MAX"] = diffs.groupby(sorted_df[group_col]).max()
    return feats


def agg_bureau_trends(bureau_df: pd.DataFrame, bureau_balance_df: pd.DataFrame) -> pd.DataFrame:
    """Bureau monthly DPD trend features, lifted from loan level to client.

    A client whose bureau DPD is climbing month over month is a different
    risk than one who has been flat at the same average.
    """
    bb = bureau_balance_df.copy()
    bb["DPD"] = bb["STATUS"].map(STATUS_TO_DPD)
    bb = bb.sort_values(["SK_ID_BUREAU", "MONTHS_BALANCE"])

    loan_feats = _recent_vs_full(bb, "SK_ID_BUREAU", "DPD", "BB_DPD")
    bur = bureau_df[["SK_ID_CURR", "SK_ID_BUREAU"]].merge(
        loan_feats, on="SK_ID_BUREAU", how="inner"
    )
    client = bur.groupby("SK_ID_CURR")[loan_feats.columns].mean()
    client = client.rename(columns={
        "BB_DPD_LAST3_MEAN": "BUREAU_DPD_LAST3_MEAN",
        "BB_DPD_TREND": "BUREAU_DPD_TREND",
        "BB_DPD_DIFF_MEAN": "BUREAU_DPD_DIFF_MEAN",
        "BB_DPD_DIFF_MAX": "BUREAU_DPD_DIFF_MAX",
    })
    return client[["BUREAU_DPD_LAST3_MEAN", "BUREAU_DPD_TREND",
                   "BUREAU_DPD_DIFF_MEAN", "BUREAU_DPD_DIFF_MAX"]]


def agg_pos_trends(pos_df: pd.DataFrame) -> pd.DataFrame:
    """POS/cash DPD direction-of-change features per SK_ID_CURR."""
    pos = pos_df.sort_values(["SK_ID_CURR", "MONTHS_BALANCE"])
    feats = _recent_vs_full(pos, "SK_ID_CURR", "SK_DPD", "POS_SK_DPD")
    feats = feats.join(_recent_vs_full(pos, "SK_ID_CURR", "SK_DPD_DEF", "POS_SK_DPD_DEF"))
    return feats


def agg_credit_card_trends(cc_df: pd.DataFrame) -> pd.DataFrame:
    """Credit-card utilization and balance velocity per SK_ID_CURR.

    Rising utilization into the application is the classic distress signal.
    """
    cc = cc_df.copy()
    cc["UTILIZATION"] = cc["AMT_BALANCE"] / (cc["AMT_CREDIT_LIMIT_ACTUAL"] + 1)
    cc = cc.sort_values(["SK_ID_CURR", "MONTHS_BALANCE"])
    feats = _recent_vs_full(cc, "SK_ID_CURR", "UTILIZATION", "CC_UTILIZATION")
    feats = feats.join(_recent_vs_full(cc, "SK_ID_CURR", "AMT_BALANCE", "CC_AMT_BALANCE"))
    return feats


# ---------------------------------------------------------------------------
# Peer-rank (cross-sectional percentile) features
# ---------------------------------------------------------------------------

RANKED_FEATURES = [
    "CREDIT_INCOME_RATIO",
    "ANNUITY_INCOME_RATIO",
    "CREDIT_TERM",
    "DAYS_EMPLOYED_RATIO",
    "INCOME_PER_PERSON",
    "EXT_SOURCE_MEAN",
]

# value feature -> list of cohort columns to rank within
GROUP_RANKS = {
    "CREDIT_INCOME_RATIO": ["NAME_INCOME_TYPE", "ORGANIZATION_TYPE"],
    "EXT_SOURCE_MEAN": ["NAME_INCOME_TYPE"],
}

RANK_EDGES_PATH = "data/processed/rank_edges.json"


def _edges_from_series(s: pd.Series) -> list[float]:
    """101 quantile cut points (0..1) approximating the full rank distribution."""
    return [float(v) for v in s.quantile(np.linspace(0, 1, 101))]


def _pct_rank(values: pd.Series, edges: list[float]) -> pd.Series:
    """Percentile of each value against fixed cut points (no refitting)."""
    return pd.Series(
        np.searchsorted(np.asarray(edges), values.to_numpy(), side="right") / 100.0,
        index=values.index,
    )


def _add_rank_features(df: pd.DataFrame, fit_ranks: bool = True,
                       edges_path: str = RANK_EDGES_PATH) -> pd.DataFrame:
    """Add RANK_* columns, fit on train / apply frozen cut points to test.

    On the training split the quantile edges (global and per cohort) are
    computed from train rows only and saved to ``edges_path``; on any later
    split those same edges are loaded and reused, so test rows never leak
    into the ranking. Unseen cohort values fall back to the global edges.
    """
    import json as _json
    from pathlib import Path as _Path

    fill = {c: df[c].median() for c in RANKED_FEATURES}

    if fit_ranks:
        edges: dict = {"__global__": {}, "__groups__": {}}
        for col in RANKED_FEATURES:
            filled = df[col].fillna(fill[col])
            edges["__global__"][col] = _edges_from_series(filled)
            df[f"RANK_{col}"] = filled.rank(pct=True)
            for group_col in GROUP_RANKS.get(col, []):
                df[f"RANK_{col}_BY_{group_col.replace('NAME_', '')}"] = (
                    filled.groupby(df[group_col]).rank(pct=True)
                )
                edges["__groups__"][f"{col}|{group_col}"] = {
                    str(k): _edges_from_series(v)
                    for k, v in filled.groupby(df[group_col])
                }
        _Path(edges_path).parent.mkdir(parents=True, exist_ok=True)
        _Path(edges_path).write_text(_json.dumps(edges))
        print(f"[features] fit rank edges on train -> {edges_path}")
        return df

    # Apply mode: reuse the training cut points exactly
    edges = _json.loads(_Path(edges_path).read_text())
    for col in RANKED_FEATURES:
        filled = df[col].fillna(fill[col])
        df[f"RANK_{col}"] = _pct_rank(filled, edges["__global__"][col])
        for group_col in GROUP_RANKS.get(col, []):
            group_edges = edges["__groups__"][f"{col}|{group_col}"]
            out = pd.Series(index=df.index, dtype=float)
            for name, idx in df.groupby(group_col).groups.items():
                out.loc[idx] = _pct_rank(
                    filled.loc[idx], group_edges.get(str(name), edges["__global__"][col])
                )
            df[f"RANK_{col}_BY_{group_col.replace('NAME_', '')}"] = out
    print(f"[features] applied train rank edges (frozen) from {edges_path}")
    return df


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
    fit_ranks: bool = True,
) -> pd.DataFrame:
    """Assemble the final modeling table keyed by SK_ID_CURR."""
    df = app_df.copy()

    agg_tables = {
        "bureau": lambda: agg_bureau(bureau_df, bb_df),
        "prev": lambda: agg_previous_applications(prev_df),
        "inst": lambda: agg_installments(inst_df),
        "pos": lambda: agg_pos_cash(pos_df),
        "cc": lambda: agg_credit_card(cc_df),
        "bureau_trends": lambda: agg_bureau_trends(bureau_df, bb_df),
        "pos_trends": lambda: agg_pos_trends(pos_df),
        "cc_trends": lambda: agg_credit_card_trends(cc_df),
    }
    for name, build in agg_tables.items():
        print(f"[features] aggregating {name}...", flush=True)
        feats = build()
        df = df.merge(feats.reset_index(), on="SK_ID_CURR", how="left")
    print("[features] merging done", flush=True)

    df = _app_domain_features(df)
    # Peer ranks run before one-hot encoding so cohort columns still exist;
    # fit_ranks=True computes the quantile edges, False reuses them frozen.
    df = _add_rank_features(df, fit_ranks=fit_ranks)
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
        fit_ranks=(args.split == "train"),
    )
    out = f"data/processed/{args.split}_features.parquet"
    features.to_parquet(out, index=False)
    n_id = 2 if "TARGET" in features.columns else 1  # SK_ID_CURR (+ TARGET)
    n_features = features.shape[1] - n_id
    print(f"\nSaved {out}")
    print(f"shape: {features.shape[0]:,} rows x {features.shape[1]} columns ({n_features} features)")
