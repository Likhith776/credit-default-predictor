# Credit Default Predictor

End-to-end machine learning project on the [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk)
Kaggle competition: predict whether an applicant will repay a loan,
using LightGBM over the full 7-table relational dataset, served through
a Streamlit app.

## Project structure

```
credit-default-predictor/
├── data/
│   ├── raw/          # competition CSVs (gitignored)
│   └── processed/    # engineered parquet files (gitignored)
├── models/           # trained .pkl / .json artifacts (Git LFS, deployable)
├── notebooks/        # EDA & experiments
├── src/
│   ├── data_loader.py            # loaders for all 7 tables + load_all_data()
│   ├── features/
│   │   └── feature_engineering.py # domain ratios + auxiliary aggregates
│   └── models/
│       ├── train.py               # LightGBM training with early stopping
│       └── predict.py             # batch scoring
├── app/
│   ├── streamlit_app.py           # prediction UI
│   ├── mock_predict.py            # mock scorer (no model needed)
│   └── requirements.txt           # deployment-only dependencies
├── requirements.txt
├── .gitignore
└── .gitattributes                 # Git LFS routes for models/*
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> Requires Python 3.10–3.12 (pinned numpy 1.26 / lightgbm 4.1 do not
> build on 3.13+).

## Data

Join the competition (free) at
[kaggle.com/competitions/home-credit-default-risk/data](https://www.kaggle.com/competitions/home-credit-default-risk/data)
and place the CSVs in `data/raw/`:

| File | Grain |
|---|---|
| `application_train.csv` | one row per loan (main table, has `TARGET`) |
| `application_test.csv` | same, without `TARGET` |
| `bureau.csv` | client's credits reported to the Credit Bureau |
| `bureau_balance.csv` | monthly balances per bureau loan |
| `previous_application.csv` | prior Home Credit applications |
| `POS_CASH_balance.csv` | monthly POS/cash loan balances |
| `installments_payments.csv` | installment payment history |
| `credit_card_balance.csv` | monthly credit card balances |

Or use the Kaggle CLI (needs `~/.kaggle/kaggle.json` — see
[kaggle.com/settings](https://www.kaggle.com/settings) → Create New Token):

```bash
kaggle competitions download -c home-credit-default-risk -p data/raw --unzip
```

Sanity-check the data loads:

```bash
python -m src.data_loader
```

## Usage

```bash
# 1. Build the feature table
python - <<'PY'
from src.data_loader import load_all_data
from src.features.feature_engineering import build_features

data = load_all_data()
train = build_features(data.pop("application"), data)
train.to_parquet("data/processed/train_features.parquet", index=False)
PY

# 2. Train (writes models/lgbm_model.pkl + models/feature_names.json)
python -m src.models.train

# 3. Score
python -m src.models.predict --input data/processed/test_features.parquet

# 4. Explore the app (falls back to mock predictions pre-training)
streamlit run app/streamlit_app.py
```

## Tracking models with Git LFS

Model artifacts are committed through LFS so the Streamlit app deploys
with its weights (routing lives in `.gitattributes`):

```bash
git lfs install
git add models/lgbm_model.pkl models/feature_names.json
git commit -m "Add trained LightGBM model"
```
