# Stock Movement Predictor - OPPE 1 (MLOps)

**Student:** Ripusudan Kumar Jha  
**Roll Number:** 23F2004705

## Overview

This project implements an end-to-end MLOps pipeline for predicting whether a stock price will increase in the next 5 minutes using historical minute-level stock data.

The pipeline includes:

- Data Versioning with DVC
- Feature Engineering with Feast
- Model Training using Scikit-learn
- Experiment Tracking with MLflow
- Model Registration using MLflow Model Registry
- Continuous Integration using GitHub Actions
- CML Report Generation

---

## Project Structure

```
.
├── StockAnalyticaData/
│   ├── v0/
│   ├── v1/
│   ├── v0.dvc
│   └── v1.dvc
├── feature_repo/
├── merged_data/
├── preprocess.py
├── merge_data.py
├── train.py
├── register_model.py
├── tests/
├── .github/workflows/
├── requirements.txt
└── README.md
```

---

## Features Used

- rolling_avg_10
- volume_sum_10
- stock_name

Target:

- Predict whether the stock closes higher after 5 minutes.

---

## DVC

Data versions:

- v0
- v1

Remote Storage:

- Google Cloud Storage (GCS)

Useful Commands

```bash
dvc pull
dvc push
```

---

## Feast

Feature Store contains:

Entity:

- stock_name

Feature View:

- stock_features

Features:

- rolling_avg_10
- volume_sum_10

Commands:

```bash
feast apply
feast materialize-incremental $(date -u +"%Y-%m-%dT%H:%M:%S")
```

---

## Model Training

Iteration 1

```bash
python train.py \
--data feature_repo/stock_feature_store/feature_repo/data/stock_features_v0.parquet
```

Iteration 2

```bash
python train.py \
--data merged_data/merged_stock_data.csv
```

Hyperparameter tuning example

```bash
python train.py \
--data feature_repo/stock_feature_store/feature_repo/data/stock_features_v0.parquet \
--n_estimators 200 \
--max_depth 20
```

---

## MLflow

Start UI

```bash
mlflow ui --host 0.0.0.0 --port 5000
```

Register Best Model

```bash
python register_model.py
```

---

## Tests

Run Feature Tests

```bash
pytest tests/test_features.py -v
```

---

## GitHub Actions CI

Pipeline performs:

- Install dependencies
- Authenticate to Google Cloud
- Pull DVC data
- Run feature sanity tests
- Generate CML report

---

## Technologies Used

- Python
- Scikit-learn
- Pandas
- DVC
- Feast
- MLflow
- GitHub Actions
- CML
- Google Cloud Platform
