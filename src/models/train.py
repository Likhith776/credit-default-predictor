"""Train a LightGBM classifier on the engineered feature table.

Usage
-----
    python -m src.models.train --input data/processed/train_features.parquet

Outputs (all under models/)
-------
    lgb_model.pkl          — final model trained on ALL data (joblib)
    feature_names.json     — column order used at fit time
    model_metrics.json     — CV/holdout AUC, Gini, best iteration, n_features
    training_medians.json  — per-feature medians (Streamlit fills missing inputs)
    feature_importance.png — top 25 features by gain
    shap_explainer.pkl     — shap.TreeExplainer for per-applicant explanations
    shap_summary.png       — global SHAP beeswarm (direction of effect)

# Tip: run this in Google Colab for fastest training:
#   Runtime -> Change runtime type -> T4 GPU
#   Upload data/processed/train_features.parquet, run training,
#   then download the models/ folder and commit it to GitHub.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

MODEL_DIR = Path("models")
TARGET = "TARGET"
SEED = 42

LGBM_PARAMS = dict(
    objective="binary",
    metric="auc",
    boosting_type="gbdt",
    num_leaves=63,
    max_depth=-1,
    learning_rate=0.05,
    n_estimators=5000,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    min_child_samples=20,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=SEED,
    n_jobs=-1,
    verbose=-1,
)


def clean_feature_names(columns):
    # LightGBM rejects feature names containing special JSON characters
    # (brackets/braces appear in one-hot dummy names like
    # "ORGANIZATION_TYPE_[Bank...]"); replace them so training works.
    import re
    return [re.sub(r'[^A-Za-z0-9_]+', '_', c) for c in columns]


def load_data(input_path):
    df = pd.read_parquet(input_path)
    y = df[TARGET]
    X = df.drop(columns=[TARGET, "SK_ID_CURR"], errors="ignore")
    X.columns = clean_feature_names(X.columns)

    n_pos = int(y.sum())
    n_neg = int((y == 0).sum())
    print(f"Data: {X.shape[0]:,} rows x {X.shape[1]} features")
    print(f"Class balance: {n_pos:,} defaults / {n_neg:,} repaid "
          f"-> default rate {y.mean():.2%} (expected ~8%)")
    return X, y


def run_cv(X, y, scale_pos_weight):
    """5-fold stratified CV with early stopping; returns per-fold stats."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    aucs, ginis, best_iters = [], [], []

    for i, (tr_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        model = lgb.LGBMClassifier(**LGBM_PARAMS, scale_pos_weight=scale_pos_weight)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(100, verbose=False),
                lgb.log_evaluation(200),
            ],
        )

        proba = model.predict_proba(X_val)[:, 1]
        fold_auc = roc_auc_score(y_val, proba)
        fold_gini = 2 * fold_auc - 1
        aucs.append(fold_auc)
        ginis.append(fold_gini)
        best_iters.append(model.best_iteration_)
        print(f"Fold {i} — AUC: {fold_auc:.4f}, Gini: {fold_gini:.4f}, "
              f"Best iter: {model.best_iteration_}")

    print(f"CV AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    return aucs, ginis, best_iters


def evaluate_holdout(model, X_val, y_val):
    proba = model.predict_proba(X_val)[:, 1]
    preds = (proba >= 0.5).astype(int)
    val_auc = roc_auc_score(y_val, proba)
    val_gini = 2 * val_auc - 1
    print(f"\nHoldout evaluation (80/20):")
    print(f"AUC: {val_auc:.4f} | Gini: {val_gini:.4f}")
    print("Confusion matrix (threshold 0.5):")
    print(confusion_matrix(y_val, preds))
    print(f"Precision (class 1): {precision_score(y_val, preds):.4f}")
    print(f"Recall    (class 1): {recall_score(y_val, preds):.4f}")
    print(f"F1        (class 1): {f1_score(y_val, preds):.4f}")
    return val_auc, val_gini


def plot_feature_importance(model, path):
    imp = pd.Series(model.booster_.feature_importance("gain"),
                    index=model.booster_.feature_name())
    top = imp.sort_values(ascending=False).head(25)
    fig, ax = plt.subplots(figsize=(10, 8))
    top.sort_values().plot(kind="barh", ax=ax)
    ax.set_title("Top 25 Features by Gain")
    ax.set_xlabel("Total gain")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved feature importance plot -> {path}")


def build_shap_artifacts(model, X_val, path_explainer, path_plot):
    # TreeExplainer is fast (no retraining; it works directly on the saved
    # LightGBM booster) so this adds negligible time to the training run.
    explainer = shap.TreeExplainer(model)
    sample = X_val.sample(n=min(2000, len(X_val)), random_state=SEED)
    shap_values = explainer.shap_values(sample)
    if isinstance(shap_values, list):  # older shap: [class0, class1] lists
        shap_values = shap_values[1]

    joblib.dump(explainer, path_explainer)
    print(f"Saved SHAP explainer -> {path_explainer}")

    shap.summary_plot(shap_values, sample, show=False)
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    fig.tight_layout()
    fig.savefig(path_plot, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"Saved SHAP summary plot -> {path_plot}")


def train(input_path: str = "data/processed/train_features.parquet") -> lgb.LGBMClassifier:
    MODEL_DIR.mkdir(exist_ok=True)
    X, y = load_data(input_path)
    scale_pos_weight = (y == 0).sum() / (y == 1).sum()
    print(f"scale_pos_weight: {scale_pos_weight:.4f}")

    # Holdout split (also reused later as the SHAP sampling pool)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    # --- 5-fold stratified CV with early stopping ---
    aucs, ginis, best_iters = run_cv(X, y, scale_pos_weight)

    # --- Final model: train on ALL data, n_estimators from CV best iters ---
    final_n_estimators = max(1, int(np.mean(best_iters) * 1.1))
    print(f"\nFinal model: n_estimators={final_n_estimators} "
          f"(mean CV best iter {np.mean(best_iters):.1f} x 1.1), trained on ALL data")
    final_params = {**LGBM_PARAMS, "n_estimators": final_n_estimators}
    final_model = lgb.LGBMClassifier(**final_params, scale_pos_weight=scale_pos_weight)
    final_model.fit(X, y)

    # --- Holdout evaluation (80/20) ---
    # The final model saw all rows, so we score an identically-configured model
    # trained on the 80% split for an honest holdout estimate.
    holdout_model = lgb.LGBMClassifier(**final_params, scale_pos_weight=scale_pos_weight)
    holdout_model.fit(X_tr, y_tr)
    val_auc, val_gini = evaluate_holdout(holdout_model, X_val, y_val)

    # --- Save artifacts ---
    joblib.dump(final_model, MODEL_DIR / "lgb_model.pkl")
    (MODEL_DIR / "feature_names.json").write_text(json.dumps(list(X.columns)))
    (MODEL_DIR / "training_medians.json").write_text(
        json.dumps({c: float(v) for c, v in X.median(numeric_only=True).items()})
    )
    metrics = {
        "cv_auc_mean": float(np.mean(aucs)),
        "cv_auc_std": float(np.std(aucs)),
        "cv_gini_mean": float(np.mean(ginis)),
        "val_auc": float(val_auc),
        "val_gini": float(val_gini),
        "best_iteration": int(np.mean(best_iters)),
        "n_features": int(X.shape[1]),
    }
    (MODEL_DIR / "model_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Saved model -> {MODEL_DIR / 'lgb_model.pkl'}")
    print(f"Saved metrics -> {MODEL_DIR / 'model_metrics.json'}")

    # --- Plots & SHAP ---
    plot_feature_importance(final_model, MODEL_DIR / "feature_importance.png")
    build_shap_artifacts(final_model, X_val,
                         MODEL_DIR / "shap_explainer.pkl",
                         MODEL_DIR / "shap_summary.png")

    print(f"\nModel saved. CV AUC: {metrics['cv_auc_mean']:.4f} ± "
          f"{metrics['cv_auc_std']:.4f} | Val AUC: {metrics['val_auc']:.4f}")
    return final_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the credit default model")
    parser.add_argument("--input", default="data/processed/train_features.parquet",
                        help="Path to the engineered training parquet")
    args = parser.parse_args()
    train(args.input)
