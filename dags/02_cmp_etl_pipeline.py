"""
DAG 02: CMP ETL Pipeline
Flow: Read raw logs → Schema validation → Anomaly detection → Write to DB → RAG analysis → Slack alert

Each step is handled by an independent module:
- schema_validator: Schema validation
- anomaly_detector: Anomaly detection
- rag_agent: RAG analysis + LLM recommendations
- notifier: Slack notifications
- config: Centralized settings
"""
import os
import shutil
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
import polars as pl
import pymongo
import certifi
from sqlalchemy import create_engine

sys.path.append('/opt/airflow/plugins')

from config import (
    MONGO_URI, GEMINI_API_KEY, POSTGRES_CONN_STR,
    SLACK_ANOMALY_WEBHOOK, SLACK_MISMATCH_WEBHOOK,
    RAW_DIR, PROCESSED_DIR, ARCHIVE_DIR, ANOMALY_DIR, QUARANTINE_DIR,
    KNOWLEDGE_DIR
)
from schema_validator import get_golden_schema, verify_schema
from anomaly_detector import flatten_metrics, detect_anomalies
from rag_agent import consult_llm_rag
from notifier import send_schema_alert, send_anomaly_alert


def process_cmp_logs(**context):
    """Main ETL flow: orchestrates all modules."""
    print("Starting ETL Process...")

    for d in [PROCESSED_DIR, ARCHIVE_DIR, ANOMALY_DIR, QUARANTINE_DIR]:
        os.makedirs(d, exist_ok=True)

    files = [f for f in os.listdir(RAW_DIR) if f.endswith('.json')]
    if not files:
        print("No files found. Job finished.")
        return

    golden_schema = get_golden_schema()
    pg_engine = create_engine(POSTGRES_CONN_STR)
    print(f"Found {len(files)} files. Processing...")

    for filename in files:
        file_path = os.path.join(RAW_DIR, filename)
        if not os.path.exists(file_path):
            continue

        print(f"\n--- Processing: {filename} ---")
        try:
            # Step 1: Read
            df = pl.read_json(file_path, schema_overrides={"error_code": pl.Utf8})
            df = df.with_columns(pl.col("timestamp").str.to_datetime())

            # Step 2: Schema validation
            errors = verify_schema(golden_schema, df.schema)
            if errors:
                print(f"FAIL: Schema Error: {filename}")
                send_schema_alert(SLACK_MISMATCH_WEBHOOK, filename, errors)
                try:
                    shutil.move(file_path, os.path.join(QUARANTINE_DIR, filename))
                    report_path = os.path.join(QUARANTINE_DIR, filename + ".report.txt")
                    with open(report_path, "w") as f:
                        f.write("\n".join(errors))
                except FileNotFoundError:
                    pass
                continue

            print(f"OK: Schema passed")

            # Step 3: Flatten + Anomaly detection
            flat_df = flatten_metrics(df)
            anomaly_df = detect_anomalies(flat_df)

            # Step 4: Write to PostgreSQL
            print(f" -> Writing {len(flat_df)} records to PostgreSQL...")
            try:
                flat_df.to_pandas().to_sql(
                    name="cmp_sensor_logs",
                    con=pg_engine,
                    if_exists="append",
                    index=False
                )
                print(" -> OK: PostgreSQL write complete")
            except Exception as e:
                print(f" -> FAIL: PostgreSQL write: {e}")

            # Step 5: Handle anomalies
            if not anomaly_df.is_empty():
                print(f"WARN: Found {anomaly_df.height} anomalies")

                # 5a. Save CSV
                save_name = f"anomaly_{filename.replace('.json', '.csv')}"
                anomaly_df.write_csv(os.path.join(ANOMALY_DIR, save_name))

                # 5b. Write to MongoDB
                if MONGO_URI:
                    try:
                        records = anomaly_df.to_dicts()
                        for r in records:
                            r['etl_processed_at'] = datetime.utcnow()

                        client = pymongo.MongoClient(MONGO_URI, tlsCAFile=certifi.where())
                        db = client["manufacturing_db"]
                        col = db["cmp_anomalies"]
                        col.insert_many(records)
                        print(f" -> OK: MongoDB write ({len(records)} records)")
                    except Exception as e:
                        print(f" -> FAIL: MongoDB write: {e}")

                # 5c. RAG analysis + Slack alert
                first_record = anomaly_df.row(0, named=True)
                ai_suggestion = consult_llm_rag(first_record, KNOWLEDGE_DIR, GEMINI_API_KEY)
                send_anomaly_alert(SLACK_ANOMALY_WEBHOOK, save_name, anomaly_df, ai_suggestion)

            # Step 6: Archive
            parquet_name = filename.replace('.json', '.parquet')
            df.write_parquet(os.path.join(PROCESSED_DIR, parquet_name))
            try:
                shutil.move(file_path, os.path.join(ARCHIVE_DIR, filename))
            except FileNotFoundError:
                pass

            print(f"OK: Done: {filename}")

        except Exception as e:
            print(f"CRITICAL: Error on {filename}: {e}")


default_args = {
    'owner': 'data_engineer',
    'depends_on_past': False,
    'retries': 0,
}

with DAG(
    '02_cmp_etl_polars_v2',
    default_args=default_args,
    description='CMP ETL Pipeline with RAG & Slack Alert',
    schedule_interval=None,
    start_date=datetime(2023, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=['CMP', 'ETL', 'RAG'],
) as dag:

    etl_task = PythonOperator(
        task_id='polars_process_logs_with_validation',
        python_callable=process_cmp_logs,
    )
