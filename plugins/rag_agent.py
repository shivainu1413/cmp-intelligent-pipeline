"""
RAG Agent Module
Retrieves similar historical repair cases via vector search,
then generates structured maintenance recommendations using an LLM
with Chain-of-Thought reasoning and confidence scoring.
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


# ----------------------------------------------------------------
# Embedding
# ----------------------------------------------------------------
def get_embedding(text_input, api_key):
    """Call Gemini API to generate a text embedding vector."""
    if not api_key:
        return None

    url = (
        "https://generativelanguage.googleapis.com/v1beta/"
        "models/text-embedding-004:embedContent"
        f"?key={api_key}"
    )
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


# ----------------------------------------------------------------
# Vector similarity
# ----------------------------------------------------------------
def cosine_similarity(v1, v2):
    """Compute cosine similarity between two vectors."""
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


# ----------------------------------------------------------------
# Retrieval
# ----------------------------------------------------------------
def retrieve_similar_cases_mongo(query_text: str, api_key: str):
    """
    Search MongoDB for similar historical repair cases.
    Uses client-side cosine similarity (suitable for < 100k documents).
    Returns a list of dicts with 'score' and 'doc' keys.
    """
    query_vector = get_embedding(query_text, api_key)
    if not query_vector:
        return []

    if not MONGO_URI:
        return []

    try:
        client = pymongo.MongoClient(MONGO_URI, tlsCAFile=certifi.where())
        col = client[MONGO_DB_NAME][MONGO_REPAIR_COLLECTION]

        cursor = col.find({"embedding": {"$exists": True}})

        results = []
        for doc in cursor:
            db_vector = doc['embedding']
            score = cosine_similarity(query_vector, db_vector)
            if score > 0.6:
                results.append({"score": score, "doc": doc})

        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:3]

    except Exception:
        return []


def format_retrieved_context(cases: list) -> str:
    """Format retrieved cases into a readable context string."""
    if not cases:
        return "No similar historical cases found."

    lines = []
    for i, item in enumerate(cases, 1):
        doc = item['doc']
        lines.append(
            f"Case #{i} (Similarity: {item['score']:.2f}):\n"
            f"  Symptom: {doc.get('symptom_desc', 'N/A')}\n"
            f"  Root Cause: {doc.get('root_cause', 'N/A')}\n"
            f"  Solution: {doc.get('solution_action', 'N/A')}"
        )
    return "\n".join(lines)


# ----------------------------------------------------------------
# Structured prompt with CoT + confidence scoring
# ----------------------------------------------------------------
DIAGNOSIS_PROMPT = """You are a semiconductor CMP (Chemical Mechanical Polishing) equipment diagnostics expert.

## Current Anomaly
- Error Code: {error_code}
- Slurry Flow Rate: {slurry_flow} ml/min
- Motor Current: {motor_current} A
- Pad Temperature: {pad_temp} °C
- Downforce Pressure: {downforce} psi

## Historical Repair Records
{retrieved_context}

## Instructions
Analyze the anomaly using Chain-of-Thought reasoning. Respond ONLY with valid JSON in this exact format:

{{
  "reasoning": "Step-by-step analysis of what the sensor readings indicate and how they correlate with historical cases",
  "root_cause": "Most likely root cause in one sentence",
  "recommended_action": "Specific maintenance action to take",
  "urgency": "critical | high | medium | low",
  "confidence": 0.85,
  "similar_case_used": 1
}}

Rules:
- "confidence" is a float between 0.0 and 1.0 reflecting how certain you are
- "similar_case_used" is the case number (1, 2, or 3) that most influenced your diagnosis, or null if none were relevant
- "urgency" must be one of: critical, high, medium, low
- Do NOT include any text outside the JSON object
"""


def build_diagnosis_prompt(anomaly_record: dict, retrieved_context: str) -> str:
    """Build the structured diagnosis prompt with anomaly data."""
    return DIAGNOSIS_PROMPT.format(
        error_code=anomaly_record.get('error_code', 'UNKNOWN'),
        slurry_flow=anomaly_record.get('slurry_flow_rate_ml_min', 'N/A'),
        motor_current=anomaly_record.get('motor_current_amps', 'N/A'),
        pad_temp=anomaly_record.get('pad_temperature_celsius', 'N/A'),
        downforce=anomaly_record.get('downforce_pressure_psi', 'N/A'),
        retrieved_context=retrieved_context,
    )


# ----------------------------------------------------------------
# LLM call + response parsing
# ----------------------------------------------------------------
def call_gemini(prompt: str, api_key: str) -> str:
    """Call Gemini generateContent API. Returns raw text response."""
    url = (
        "https://generativelanguage.googleapis.com/v1beta/"
        "models/gemini-2.0-flash:generateContent"
        f"?key={api_key}"
    )
    headers = {"Content-Type": "application/json"}
    payload = {"contents": [{"parts": [{"text": prompt}]}]}

    res = requests.post(url, headers=headers, json=payload, timeout=60)

    if res.status_code != 200:
        error_msg = res.json().get('error', {}).get('message', res.text)
        raise RuntimeError(f"Gemini API error ({res.status_code}): {error_msg}")

    response_json = res.json()
    if "candidates" not in response_json:
        raise RuntimeError(f"Unexpected response: {str(response_json)[:200]}")

    return response_json['candidates'][0]['content']['parts'][0]['text']


def parse_llm_response(raw_text: str) -> dict:
    """
    Parse the structured JSON response from the LLM.
    Returns a validated dict with diagnosis fields.
    Falls back to a default structure if parsing fails.
    """
    # Strip markdown code fences if present
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        # Remove first and last lines (``` markers)
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines)

    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        return {
            "reasoning": raw_text,
            "root_cause": "Unable to parse structured response",
            "recommended_action": "Manual inspection required",
            "urgency": "medium",
            "confidence": 0.0,
            "similar_case_used": None,
            "parse_error": True,
        }

    # Validate and normalize fields
    valid_urgency = {"critical", "high", "medium", "low"}
    if result.get("urgency") not in valid_urgency:
        result["urgency"] = "medium"

    confidence = result.get("confidence", 0.5)
    if not isinstance(confidence, (int, float)):
        confidence = 0.5
    result["confidence"] = max(0.0, min(1.0, float(confidence)))

    result["parse_error"] = False
    return result


# ----------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------
def consult_llm_rag(anomaly_record: dict, base_dir: str, api_key: str = None) -> dict:
    """
    RAG Agent: query historical cases and generate structured diagnosis.

    Returns a dict with keys:
      - reasoning, root_cause, recommended_action, urgency, confidence
      - retrieved_cases (formatted string of similar cases)
      - similar_case_used (which case number influenced the diagnosis)

    If API key is missing or call fails, returns a fallback dict.
    """
    # Build query from anomaly data
    query_text = (
        f"Error code {anomaly_record.get('error_code')} "
        f"with slurry flow {anomaly_record.get('slurry_flow_rate_ml_min')} "
        f"and motor current {anomaly_record.get('motor_current_amps')}"
    )

    # Retrieve similar cases
    cases = retrieve_similar_cases_mongo(query_text, api_key)
    retrieved_context = format_retrieved_context(cases)

    # Build structured prompt
    prompt = build_diagnosis_prompt(anomaly_record, retrieved_context)

    # Call LLM and parse response
    if not api_key:
        return {
            "reasoning": "No API key provided, skipping LLM diagnosis.",
            "root_cause": "N/A",
            "recommended_action": "Configure GEMINI_API_KEY to enable AI diagnosis.",
            "urgency": "low",
            "confidence": 0.0,
            "similar_case_used": None,
            "retrieved_cases": retrieved_context,
            "parse_error": False,
        }

    try:
        raw_text = call_gemini(prompt, api_key)
        result = parse_llm_response(raw_text)
        result["retrieved_cases"] = retrieved_context
        return result

    except Exception as e:
        return {
            "reasoning": f"LLM call failed: {str(e)}",
            "root_cause": "Unable to generate diagnosis",
            "recommended_action": "Check API connectivity and retry.",
            "urgency": "medium",
            "confidence": 0.0,
            "similar_case_used": None,
            "retrieved_cases": retrieved_context,
            "parse_error": True,
        }
