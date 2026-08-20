"""Batch inference with the trained model.

Usage
-----
    python -m src.models.predict --input data/processed/test_features.parquet \
        --output data/processed/test_predictions.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

MODEL_PATH = Path("models/lgbm_model.pkl")
FEATURE_NAMES_PATH = Path("models/feature_names.json")


def load_model(model_path: str | Path = MODEL_PATH):
    """Load the fitted model and the feature order it was trained on."""
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"No trained model at {model_path}. Run `python -m src.models.train` "
            f"first, or use app/mock_predict.py for a mock inference."
        )
    model = joblib.load(model_path)
    feature_names = None
    if FEATURE_NAMES_PATH.exists():
        feature_names = json.loads(FEATURE_NAMES_PATH.read_text())
    return model, feature_names


def predict(df: pd.DataFrame, model_path: str | Path = MODEL_PATH) -> pd.DataFrame:
    """Return a DataFrame with SK_ID_CURR, probability of default, and label."""
    model, feature_names = load_model(model_path)
    X = df[feature_names] if feature_names else df.drop(columns=["SK_ID_CURR"], errors="ignore")
    proba = model.predict_proba(X)[:, 1]
    return pd.DataFrame(
        {
            "SK_ID_CURR": df["SK_ID_CURR"] if "SK_ID_CURR" in df else np.arange(len(df)),
            "SCORE": proba,
            "PREDICTION": (proba >= 0.5).astype(int),
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score a feature table")
    parser.add_argument("--input", required=True, help="Parquet with engineered features")
    parser.add_argument("--output", default="data/processed/test_predictions.csv")
    parser.add_argument("--model", default=str(MODEL_PATH))
    args = parser.parse_args()

    result = predict(pd.read_parquet(args.input), args.model)
    result.to_csv(args.output, index=False)
    print(f"Wrote {len(result):,} predictions -> {args.output}")
