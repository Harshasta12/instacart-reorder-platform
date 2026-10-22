# Instacart Reorder Platform

An end-to-end data engineering and machine learning project built on the real [Instacart Market Basket Analysis dataset](https://www.kaggle.com/c/instacart-market-basket-analysis). The project transforms raw CSVs into a tested analytical warehouse, trains and tracks a LightGBM reorder prediction model, and serves recommendations through a REST API.

## Overview

Given a user ID, the platform predicts which previously purchased products the user is likely to reorder in their next order.

### Architecture

```text
Kaggle CSVs
    |
    v
CSV-to-Parquet Ingestion (DuckDB / PyArrow)
    |
    v
Parquet Data Lake
    |
    v
dbt Data Warehouse
    |
    +-- Staging Models
    +-- Intermediate Models
    +-- Mart Models
    +-- Data Quality Tests
    |
    v
Feature Engineering
    |
    v
LightGBM Model (MLflow Tracking)
    |
    v
FastAPI Recommendation Service
```

An Airflow DAG for ingestion and dbt orchestration has been written but is not yet deployed.

## Tech Stack

- **Data Engineering:** Python, DuckDB, Parquet, PyArrow
- **Data Transformation:** dbt Core, dbt-duckdb
- **Machine Learning:** LightGBM, scikit-learn
- **Experiment Tracking:** MLflow
- **API Development:** FastAPI, Uvicorn
- **Orchestration:** Apache Airflow (DAG implemented; deployment pending)

## Data Engineering

The pipeline processes the six CSV tables provided by the Instacart dataset.

### Ingestion

`ingestion/csv_to_parquet.py`

- Converts raw CSV files into typed Parquet files.
- Uses Zstandard (ZSTD) compression.
- Produces six Parquet tables for downstream processing.

### Data Warehouse

`dbt_project/`

The dbt project organizes transformations into three layers:

- **Staging:** Cleaned and standardized source data.
- **Intermediate:** User-product purchase history and supporting transformations.
- **Marts:** Analytical tables used for feature engineering and model serving.

### Data Quality

The project includes dbt tests for uniqueness, non-null values, referential integrity, accepted values, and accepted ranges.

- **21 declared data tests**
- **28/28 dbt build checks passing** on the full dataset

## Machine Learning

### Prediction Task

For each `(user, previously purchased product)` pair, predict whether the product will appear in that user's next order.

The models use historical purchase behavior and user/product statistics to rank likely reorders.

### Leakage Prevention and Validation

- Features are derived from prior purchase history to avoid using future information.
- Validation is split by user.
- Model experiments are tracked in MLflow.
- Evaluation uses a 20% user sample, comprising approximately 1.7 million candidate rows.

### Model Performance

| Experiment | Features | ROC-AUC | Average Precision | Best F1 |
|---|---:|---:|---:|---:|
| `lgbm-baseline` | 16 | 0.8321 | 0.4200 | 0.4400 |
| `lgbm-v2` | 26 | 0.8340 | 0.4228 | 0.4417 |

**Feature expansion:** Version 2 adds recency, basket size, aisle/department information, and ranking-related features.

The second model achieves a small improvement over the baseline. Although the additional recency features are used by the model, their incremental predictive value is limited in these experiments.

**Evaluation note:** F1 is calculated at the product level using the best validation threshold. It is not directly comparable to the per-order F1 metric used on the Kaggle leaderboard.

### MLflow Experiment Tracking

The training runs and evaluation metrics are tracked using MLflow.

![MLflow baseline run](docs/mlflow_baseline.png)

## Recommendation API

The trained model is exposed through a FastAPI REST endpoint.

### Endpoint

```http
GET /recommend/{user_id}?k=10
```

Returns the top-k products that a user is predicted to reorder, along with their predicted probabilities.

### Example

```http
GET /recommend/12345?k=10
```

The response contains ranked product recommendations and their associated scores. The exact response schema depends on the serving implementation.

### API Documentation

Run the service locally and open the interactive Swagger documentation.

![FastAPI Swagger documentation](docs/api_swagger.png)

- Swagger UI: `http://localhost:8000/docs`

**Demo limitation:** The current serving demo covers approximately 26,000 users in the 20% sample used for training. It demonstrates model serving and API integration rather than generalization to the entire user population.

## Getting Started

### Prerequisites

- Python 3.10 or a compatible version supported by the selected dependencies
- The six CSV files from the [Instacart Market Basket Analysis dataset](https://www.kaggle.com/c/instacart-market-basket-analysis)

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd <your-repository-directory>
```

### 2. Create a virtual environment

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install duckdb pyarrow pandas dbt-core dbt-duckdb lightgbm scikit-learn mlflow fastapi uvicorn
```

### 4. Prepare the data

Download the six dataset CSVs and place them in `data/raw/`.

Run the ingestion script:

```bash
python ingestion/csv_to_parquet.py
```

### 5. Build the dbt warehouse

From the repository root:

```bash
cd dbt_project
dbt deps --profiles-dir .
dbt build --profiles-dir .
cd ..
```

### 6. Train the models

Train the baseline model:

```bash
python ml/train_baseline.py --sample 0.2
```

Train version 2:

```bash
python ml/train_v2.py --sample 0.2
```

### 7. Export the serving artifacts

```bash
python ml/export_serving.py
```

### 8. Start the API

```bash
python -m uvicorn ml.serving.app:app --port 8000
```

Open `http://localhost:8000/docs` to explore the API.

> **Note:** Run commands from the repository root unless otherwise specified. The exact Python version, file paths, and dependency versions should match the project's implementation and configuration.

## Roadmap

- [ ] Deploy the Airflow DAG using Docker to orchestrate ingestion and dbt transformations.
- [ ] Extend orchestration to include model training and artifact export.
- [ ] Add data and prediction drift monitoring with Evidently.
- [ ] Build a Streamlit dashboard for exploring reorder recommendations and model performance.
- [ ] Evaluate performance across additional user samples and temporal validation splits.
- [ ] Improve recommendation quality and investigate the impact of additional behavioral features.

## Project Highlights

- End-to-end pipeline from raw data ingestion to model serving.
- Columnar Parquet storage with compressed files.
- Layered dbt transformations and automated data quality checks.
- LightGBM experiments with MLflow tracking.
- REST API for serving ranked product recommendations.
- Explicit documentation of model limitations and future improvements.

## Dataset

[Instacart Market Basket Analysis — Kaggle](https://www.kaggle.com/c/instacart-market-basket-analysis)
