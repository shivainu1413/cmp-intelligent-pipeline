"""
RAG Agent Module
Retrieves similar historical repair cases via vector search,
then generates maintenance recommendations using an LLM.
"""
import os
import sys
import requests
import json
import pymongo
import certifi
import numpy as np
import polars as pl
from datetime import datetime

sys.path.append('/opt/airflow/plugins')
from config import MONGO_URI, MONGO_DB_NAME, MONGO_REPAIR_COLLECTION


def get_embedding(text_input, api_key):
    """Call Gemini API to generate a text embedding vector."""
    if not api_key:
        return None

    url = f"https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "models/text-embedding-004",
        "content": {"parts": [{"text": text_input}]}
    }
    try:
        res = requests.post(url, headers=headers, json=payload, timeout=10)
        if res.status_code == 200:
            return res.json()['embedding']['values']
    except Exception:
        pass
    return None


def cosine_similarity(v1, v2):
    """Compute cosine similarity between two vectors."""
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def retrieve_similar_cases_mongo(query_text: str, api_key: str):
    """
    Search MongoDB for similar historical repair cases.
    Uses client-side cosine similarity (suitable for < 100k documents).
    """
    query_vector = get_embedding(query_text, api_key)
    if not query_vector:
        return "Vector embedding failed (Check API Key)."

    if not MONGO_URI:
        return "MONGO_URI not configured."

    try:
        client = pymongo.MongoClient(MONGO_URI, tlsCAFile=certifi.where())
        col = client[MONGO_DB_NAME][MONGO_REPAIR_COLLECTION]

        cursor = col.find({"embedding": {"$exists": True}})

        results = []
        for doc in cursor:
            db_vector = doc['embedding']
            score = cosine_similarity(query_vector, db_vector)

            if score > 0.6:
                results.append({
                    "score": score,
                    "doc": doc
                })

        results.sort(key=lambda x: x['score'], reverse=True)
        top_k = results[:3]

        if not top_k:
            return "No similar historical cases found (Similarity too low)."

        context_text = ""
        for i, item in enumerate(top_k, 1):
            doc = item['doc']
            context_text += f"Similar Case #{i} (Score: {item['score']:.2f}):\n"
            context_text += f"  - Symptom: {doc['symptom_desc']}\n  - Cause: {doc['root_cause']}\n  - Fix: {doc['solution_action']}\n"

        return context_text

    except Exception as e:
        return f"MongoDB Connection Error: {e}"


def consult_llm_rag(anomaly_record: dict, base_dir: str, api_key: str = None) -> str:
    """
    RAG Agent: query historical cases and generate maintenance recommendations.
    Flow: Build query → Retrieve similar cases → LLM generates diagnosis
    """
    query_text = (
        f"Error code {anomaly_record.get('error_code')} "
        f"with slurry flow {anomaly_record.get('slurry_flow_rate_ml_min')} "
        f"and motor current {anomaly_record.get('motor_current_amps')}"
    )

    # Retrieve: find similar historical cases from MongoDB
    retrieved_context = retrieve_similar_cases_mongo(query_text, api_key)

    # Generate: use Gemini LLM to produce recommendations
    prompt = f"""
    Context: I am a semiconductor equipment engineer.
    Current Anomaly: {query_text}
    Historical Repair Records: {retrieved_context}
    Task: Based on the history, what is the root cause and fix? Provide a concise answer.
    """

    if api_key:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
            headers = {"Content-Type": "application/json"}
            payload = {
                "contents": [{"parts": [{"text": prompt}]}]
            }

            res = requests.post(url, headers=headers, json=payload, timeout=60)

            if res.status_code == 200:
                response_json = res.json()
                if "candidates" in response_json:
                    return response_json['candidates'][0]['content']['parts'][0]['text']
                else:
                    return f"WARNING: Unexpected response format: {str(response_json)[:200]}..."
            else:
                error_msg = res.json().get('error', {}).get('message', res.text)
                return f"WARNING: API Error ({res.status_code}): {error_msg}\nContext: {retrieved_context}"

        except Exception as e:
            return f"WARNING: Connection Error: {str(e)}\nContext: {retrieved_context}"

    return f"[Simulation] No API Key.\nContext: {retrieved_context}"
