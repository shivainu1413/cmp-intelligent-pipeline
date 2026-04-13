"""
MongoDB Knowledge Base Initialization Script
Populates the repair_history collection with historical cases and their embeddings.
Run locally: python init_mongo.py (requires .env or environment variables)
"""
import os
import requests
import time
import pymongo
import certifi
from datetime import datetime

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
MONGO_URI = os.getenv("MONGO_URI", "")
DB_NAME = "manufacturing_db"
COLLECTION_NAME = "repair_history"


def get_embedding(text_input):
    """Call Gemini API to generate a text embedding."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "models/text-embedding-004",
        "content": {"parts": [{"text": text_input}]}
    }
    try:
        res = requests.post(url, headers=headers, json=payload, timeout=10)
        if res.status_code == 200:
            return res.json()['embedding']['values']
    except Exception as e:
        print(f"Embedding Error: {e}")
    return None


def init_mongo_db():
    if not MONGO_URI:
        print("ERROR: MONGO_URI is not set. Check your .env file.")
        return
    if not GEMINI_API_KEY:
        print("ERROR: GEMINI_API_KEY is not set. Check your .env file.")
        return

    print("Connecting to MongoDB...")
    client = pymongo.MongoClient(MONGO_URI, tlsCAFile=certifi.where())
    db = client[DB_NAME]
    col = db[COLLECTION_NAME]

    col.delete_many({})
    print("Cleared old data.")

    # Historical repair records
    raw_data = [
        {"code": "ALM-3050", "symptom": "Slurry flow rate unstable (150-180ml) and pump vibration high.", "cause": "Filter Clog", "action": "Replace POU Filter immediately."},
        {"code": "ALM-3050", "symptom": "Flow rate is 0, but pump is running. No leaks found.", "cause": "Air Lock in Pump", "action": "Purge air from the pump line manually."},
        {"code": "ALM-3050", "symptom": "High motor current (28A) and squeaking sound from head.", "cause": "Slurry Dry-out / High Friction", "action": "Clean the nozzle and perform pad conditioning."},
        {"code": "ERR-9999", "symptom": "Sensor heartbeat lost. Signal intermittent.", "cause": "RS232 Cable Loose", "action": "Reconnect and screw tight the RS232 connector."},
        {"code": "Unknown", "symptom": "Strange grinding noise coming from the platen area.", "cause": "Mechanical Bearing Wear", "action": "Schedule PM to replace Platen Bearings."},
        {"code": "Unknown", "symptom": "Wafer slipped out during polishing process.", "cause": "Retaining Ring Worn out", "action": "Replace Retaining Ring and check head alignment."}
    ]

    print("Generating embeddings and indexing data...")

    docs_to_insert = []
    for row in raw_data:
        vector = get_embedding(row["symptom"])

        if vector:
            doc = {
                "error_code": row["code"],
                "symptom_desc": row["symptom"],
                "root_cause": row["cause"],
                "solution_action": row["action"],
                "embedding": vector,
                "created_at": datetime.utcnow()
            }
            docs_to_insert.append(doc)
            print(f"  -> Processed: [{row['code']}] {row['symptom'][:40]}...")
            time.sleep(0.5)

    if docs_to_insert:
        col.insert_many(docs_to_insert)
        print(f"\nSuccessfully inserted {len(docs_to_insert)} documents into MongoDB.")
    else:
        print("ERROR: No documents inserted. Check your GEMINI_API_KEY.")


if __name__ == "__main__":
    init_mongo_db()
