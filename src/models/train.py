"""Train a LightGBM classifier on the engineered feature table.

Usage
-----
    python -m src.models.train --input data/processed/train_features.parquet

Outputs
-------
    models/lgbm_model.pkl     — fitted pipeline (joblib)
    models/feature_names.json — column order used at fit time
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

MODEL_DIR = Path("models")
TARGET = "TARGET"
SEED = 42

# Slightly imbalanced target -> scale_pos_weight ≈ n_negative / n_positive.
LGBM_PARAMS = dict(
    objective="binary",
    n_estimators=2000,
    learning_rate=0.05,
    num_leaves=34,
    colsample_bytree=0.9,
    subsample=0.9,
    subsample_freq=1,
    max_depth=8,
    reg_alpha=0.04,
    reg_lambda=0.07,
    min_child_samples=60,
    random_state=SEED,
    n_jobs=-1,
    verbose=-1,
)


def train(input_path: str = "data/processed/train_features.parquet",
          model_path: str | None = None) -> lgb.LGBMClassifier:
    df = pd.read_parquet(input_path)
    y = df[TARGET]
    X = df.drop(columns=[TARGET, "SK_ID_CURR"], errors="ignore")

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    scale_pos_weight = (y_tr == 0).sum() / (y_tr == 1).sum()

    model = lgb.LGBMClassifier(**LGBM_PARAMS, scale_pos_weight=scale_pos_weight)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(200)],
    )

    val_auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
    print(f"Validation AUC: {val_auc:.5f} (best iter: {model.best_iteration_})")

    MODEL_DIR.mkdir(exist_ok=True)
    model_path = model_path or str(MODEL_DIR / "lgbm_model.pkl")
    joblib.dump(model, model_path)
    (MODEL_DIR / "feature_names.json").write_text(json.dumps(list(X.columns)))
    print(f"Saved model -> {model_path}")
    print(f"Saved feature names -> {MODEL_DIR / 'feature_names.json'}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the credit default model")
    parser.add_argument("--input", default="data/processed/train_features.parquet",
                        help="Path to the engineered training parquet")
    args = parser.parse_args()
    train(args.input)
