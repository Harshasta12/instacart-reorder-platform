\# Instacart Reorder Platform



End-to-end data + ML project on the real \[Instacart Market Basket](https://www.kaggle.com/c/instacart-market-basket-analysis) dataset (3.4M orders, 32M order-product rows): raw CSVs to a tested dbt warehouse, a tracked LightGBM reorder model, and a REST API.



\## What it does



Given a user, predict which products they previously bought will be in their next order.



```

Kaggle CSVs -> Parquet lake (DuckDB) -> dbt (staging -> intermediate -> marts, 28 tests)

&#x20;                                             |

&#x20;                             features -> LightGBM (MLflow tracking) -> FastAPI

```



\## Stack



DuckDB, Parquet, dbt (dbt-duckdb), LightGBM, MLflow, FastAPI, Python. An Airflow DAG (`airflow/dags/`) for ingest -> dbt is written but not deployed yet.



\## Data engineering



\- `ingestion/csv\_to\_parquet.py`: typed CSV -> Parquet (ZSTD) conversion, 6 tables.

\- `dbt\_project/`: staging views, an intermediate user-product history table, and mart tables.

\- 21 dbt data tests (unique, not\_null, relationships, accepted\_values, accepted\_range). `dbt build`: 28/28 pass on the full dataset.



\## Machine learning



Task: for each (user, previously bought product), predict whether it appears in the user's next order. Features come only from prior orders (no leakage); validation split is by user.



| Run | Features | AUC | Avg precision | Best F1 |

|---|---|---|---|---|

| lgbm-baseline | 16 (purchase history, user/product stats) | 0.8321 | 0.4200 | 0.4400 |

| lgbm-v2 | 26 (+ recency, basket size, aisle/department, rank) | 0.8340 | 0.4228 | 0.4417 |



Evaluated on a 20% user sample (1.7M rows). The v2 gain is small; the recency features are used by the model but add limited signal beyond the baseline. F1 is product-level at the best threshold, not the per-order F1 used on the Kaggle leaderboard. Experiments are tracked in MLflow.



!\[MLflow baseline run](docs/mlflow\_baseline.png)



\## API



`GET /recommend/{user\_id}?k=10` returns the top-k products a user is likely to reorder, with probabilities. The demo covers the \~26K users in the 20% sample, which the model was trained on, so it demonstrates serving, not accuracy.



!\[API docs](docs/api\_swagger.png)



\## Run it



```bash

python -m venv .venv \&\& .venv\\Scripts\\activate      # Windows

pip install duckdb pyarrow pandas dbt-core dbt-duckdb lightgbm scikit-learn mlflow fastapi uvicorn

\# put the 6 Kaggle CSVs in data/raw/

python ingestion/csv\_to\_parquet.py

cd dbt\_project \&\& dbt deps --profiles-dir . \&\& dbt build --profiles-dir . \&\& cd ..

python ml/train\_baseline.py --sample 0.2

python ml/train\_v2.py --sample 0.2

python ml/export\_serving.py

python -m uvicorn ml.serving.app:app --port 8000     # http://localhost:8000/docs

```



\## Roadmap



\- Airflow deployment (Docker) for the ingest -> dbt -> train flow

\- Evidently drift monitoring

\- Streamlit dashboard

