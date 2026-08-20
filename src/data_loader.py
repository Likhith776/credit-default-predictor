"""Data loading utilities for the Home Credit Default Risk dataset.

Each loader reads one raw CSV from ``data/raw/`` and prints a compact
summary (shape, memory, and — for the application table — the TARGET
distribution and missing-cell percentage) so sanity checks happen
automatically at load time.

Usage
-----
    from src.data_loader import load_all_data

    data = load_all_data()                      # dict of all 7 tables
    app = load_application("data/raw/application_train.csv")

    python -m src.data_loader                   # quick CLI health check
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_DATA_DIR = "data/raw"

# Canonical file names in the Kaggle competition download.
FILE_NAMES = {
    "application": "application_train.csv",
    "bureau": "bureau.csv",
    "bureau_balance": "bureau_balance.csv",
    "previous_application": "previous_application.csv",
    "pos_cash": "POS_CASH_balance.csv",
    "installments": "installments_payments.csv",
    "credit_card": "credit_card_balance.csv",
}


def _print_summary(df: pd.DataFrame, name: str) -> None:
    """Print a one-line summary shared by every loader."""
    memory_mb = df.memory_usage(deep=True).sum() / 1024**2
    missing_pct = df.isna().to_numpy().mean() * 100
    print(
        f"[{name}] {len(df):,} rows x {df.shape[1]} cols | "
        f"missing cells: {missing_pct:.1f}% | memory: {memory_mb:.1f} MB"
    )


def _read_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Expected data file not found: {path}. "
            f"Download it from https://kaggle.com/competitions/"
            f"home-credit-default-risk/data into {DEFAULT_DATA_DIR}/."
        )
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Individual table loaders
# ---------------------------------------------------------------------------

def load_application(path: str | Path = f"{DEFAULT_DATA_DIR}/application_train.csv") -> pd.DataFrame:
    """Load application_train.csv — the main table (one row per loan).

    Prints shape, TARGET class distribution, and overall % missing.
    """
    df = _read_csv(path)
    print("=" * 72)
    print(f"application_train ({Path(path).name})")
    print(f"  shape: {len(df):,} rows x {df.shape[1]} columns")

    if "TARGET" in df.columns:
        counts = df["TARGET"].value_counts().sort_index()
        share = df["TARGET"].value_counts(normalize=True).sort_index() * 100
        print("  TARGET distribution:")
        for label in counts.index:
            tag = "repaid" if label == 0 else "DEFAULT"
            print(f"    TARGET={label} ({tag}): {counts[label]:,} ({share[label]:.2f}%)")

    missing_pct = df.isna().to_numpy().mean() * 100
    print(f"  missing cells: {missing_pct:.1f}% "
          f"({df.isna().to_numpy().sum():,} of {df.size:,})")
    _print_summary(df, "application_train")
    print("=" * 72)
    return df


def load_bureau(path: str | Path = f"{DEFAULT_DATA_DIR}/bureau.csv") -> pd.DataFrame:
    """Load bureau.csv — client's credit records reported to the Credit Bureau."""
    df = _read_csv(path)
    _print_summary(df, "bureau")
    return df


def load_bureau_balance(path: str | Path = f"{DEFAULT_DATA_DIR}/bureau_balance.csv") -> pd.DataFrame:
    """Load bureau_balance.csv — monthly balances per Credit Bureau loan."""
    df = _read_csv(path)
    _print_summary(df, "bureau_balance")
    return df


def load_previous_applications(
    path: str | Path = f"{DEFAULT_DATA_DIR}/previous_application.csv",
) -> pd.DataFrame:
    """Load previous_application.csv — client's prior Home Credit applications."""
    df = _read_csv(path)
    _print_summary(df, "previous_application")
    return df


def load_pos_cash(path: str | Path = f"{DEFAULT_DATA_DIR}/POS_CASH_balance.csv") -> pd.DataFrame:
    """Load POS_CASH_balance.csv — monthly balance snapshots of POS/cash loans."""
    df = _read_csv(path)
    _print_summary(df, "pos_cash")
    return df


def load_installments(path: str | Path = f"{DEFAULT_DATA_DIR}/installments_payments.csv") -> pd.DataFrame:
    """Load installments_payments.csv — payment history for previous loans."""
    df = _read_csv(path)
    _print_summary(df, "installments")
    return df


def load_credit_card(path: str | Path = f"{DEFAULT_DATA_DIR}/credit_card_balance.csv") -> pd.DataFrame:
    """Load credit_card_balance.csv — monthly credit card balance snapshots."""
    df = _read_csv(path)
    _print_summary(df, "credit_card")
    return df


# ---------------------------------------------------------------------------
# Batch loader
# ---------------------------------------------------------------------------

_LOADERS = {
    "application": load_application,
    "bureau": load_bureau,
    "bureau_balance": load_bureau_balance,
    "previous_application": load_previous_applications,
    "pos_cash": load_pos_cash,
    "installments": load_installments,
    "credit_card": load_credit_card,
}


def load_all_data(data_dir: str | Path = DEFAULT_DATA_DIR) -> dict[str, pd.DataFrame]:
    """Load all 7 raw tables from ``data_dir`` into a dict of DataFrames.

    Returns
    -------
    dict with keys:
        application, bureau, bureau_balance, previous_application,
        pos_cash, installments, credit_card
    """
    data_dir = Path(data_dir)
    print(f"\nLoading Home Credit dataset from: {data_dir.resolve()}\n")
    data: dict[str, pd.DataFrame] = {}
    for key, loader in _LOADERS.items():
        data[key] = loader(data_dir / FILE_NAMES[key])
    print(f"\nDone — loaded {len(data)} tables: {', '.join(data)}")
    return data


if __name__ == "__main__":
    load_all_data()
