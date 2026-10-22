"""Daily pipeline: ingest -> dbt build."""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

PROJECT = "/opt/airflow/project"

with DAG(
    dag_id="instacart_pipeline",
    start_date=datetime(2026, 10, 1),
    schedule="@daily",
    catchup=False,
    default_args={"owner": "harsh", "retries": 2, "retry_delay": timedelta(minutes=5)},
    tags=["instacart", "dbt"],
) as dag:
    ingest = BashOperator(
        task_id="ingest_csv_to_parquet",
        bash_command=f"cd {PROJECT} && python ingestion/csv_to_parquet.py",
    )
    dbt_build = BashOperator(
        task_id="dbt_build",
        bash_command=f"cd {PROJECT}/dbt_project && dbt deps && dbt build --profiles-dir .",
    )
    ingest >> dbt_build
