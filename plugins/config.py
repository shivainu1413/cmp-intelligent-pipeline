"""
Centralized configuration module.
All secrets and settings are read from environment variables.
"""
import os

# External services
MONGO_URI = os.getenv("MONGO_URI", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
SLACK_ANOMALY_WEBHOOK = os.getenv("SLACK_ANOMALY_WEBHOOK", "")
SLACK_MISMATCH_WEBHOOK = os.getenv("SLACK_MISMATCH_WEBHOOK", "")
POSTGRES_CONN_STR = os.getenv(
    "POSTGRES_CONN_STR",
    "postgresql+psycopg2://airflow:airflow@postgres/airflow"
)

# MongoDB
MONGO_DB_NAME = "manufacturing_db"
MONGO_ANOMALY_COLLECTION = "cmp_anomalies"
MONGO_REPAIR_COLLECTION = "repair_history"

# Paths
BASE_DIR = "/opt/airflow/data"
KNOWLEDGE_DIR = "/opt/airflow/knowledge"
RAW_DIR = os.path.join(BASE_DIR, "raw_logs")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed_data")
ARCHIVE_DIR = os.path.join(BASE_DIR, "archive_logs")
ANOMALY_DIR = os.path.join(BASE_DIR, "anomalies")
QUARANTINE_DIR = os.path.join(BASE_DIR, "quarantine_logs")
