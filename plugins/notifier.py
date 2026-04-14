"""
Notification Module
Handles all Slack alert logic for schema violations and anomalies.
Supports structured AI diagnosis output with urgency and confidence.
"""
import requests
import polars as pl
from typing import List


URGENCY_EMOJI = {
    "critical": "🔴",
    "high": "🟠",
    "medium": "🟡",
    "low": "🟢",
}


def send_schema_alert(webhook_url: str, filename: str, error_reasons: List[str]):
    """Send a Slack notification when schema validation fails."""
    if not webhook_url:
        print("WARN: SLACK_MISMATCH_WEBHOOK not set, skipping schema alert")
        return

    error_text = "\n".join([f"• {e}" for e in error_reasons])
    payload = {
        "blocks": [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": "Schema Violation Detected", "emoji": True}
            },
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"*File: `{filename}`*\n*Errors:*\n```{error_text}```"}
            },
            {
                "type": "context",
                "elements": [{"type": "mrkdwn", "text": "File moved to quarantine."}]
            }
        ]
    }
    try:
        requests.post(webhook_url, json=payload, timeout=10)
    except Exception as e:
        print(f"WARN: Failed to send schema alert: {e}")


def format_diagnosis_text(diagnosis: dict) -> str:
    """Format a structured diagnosis dict into readable text for Slack."""
    if isinstance(diagnosis, str):
        # Backward compatibility: old-style string response
        return diagnosis

    urgency = diagnosis.get("urgency", "medium")
    emoji = URGENCY_EMOJI.get(urgency, "⚪")
    confidence = diagnosis.get("confidence", 0.0)
    confidence_pct = f"{confidence * 100:.0f}%"

    parts = [
        f"{emoji} *Urgency:* {urgency.upper()}  |  *Confidence:* {confidence_pct}",
        "",
        f"*Root Cause:* {diagnosis.get('root_cause', 'N/A')}",
        f"*Action:* {diagnosis.get('recommended_action', 'N/A')}",
    ]

    reasoning = diagnosis.get("reasoning", "")
    if reasoning:
        # Truncate reasoning to keep Slack message under limit
        if len(reasoning) > 500:
            reasoning = reasoning[:500] + "..."
        parts.append(f"\n*Reasoning:*\n```{reasoning}```")

    case_num = diagnosis.get("similar_case_used")
    if case_num:
        parts.append(f"_Based on historical case #{case_num}_")

    return "\n".join(parts)


def send_anomaly_alert(
    webhook_url: str,
    filename: str,
    anomaly_df: pl.DataFrame,
    ai_suggestion,
):
    """Send a Slack notification with anomaly details and AI diagnosis."""
    if not webhook_url:
        print("WARN: SLACK_ANOMALY_WEBHOOK not set, skipping anomaly alert")
        return

    preview_df = anomaly_df.head(5)
    cols = ["timestamp", "slurry_flow_rate_ml_min", "motor_current_amps", "error_code"]
    valid_cols = [c for c in cols if c in preview_df.columns]
    csv_preview = preview_df.select(valid_cols).write_csv(separator=",")

    diagnosis_text = format_diagnosis_text(ai_suggestion)

    rich_payload = {
        "blocks": [
            {"type": "header", "text": {"type": "plain_text", "text": "Process Anomaly & AI Diagnosis", "emoji": True}},
            {"type": "section", "fields": [{"type": "mrkdwn", "text": f"*File:*\n`{filename}`"}]},
            {"type": "divider"},
            {"type": "section", "text": {"type": "mrkdwn", "text": diagnosis_text[:2900]}},
            {"type": "section", "text": {"type": "mrkdwn", "text": f"*Data Snapshot:*\n```csv\n{csv_preview}```"}}
        ]
    }

    fallback_payload = {
        "text": f"Anomaly Alert\n\nFile: {filename}\n\n{diagnosis_text}\n\nData:\n{csv_preview}"
    }

    try:
        r = requests.post(webhook_url, json=rich_payload, timeout=10)
        if r.status_code == 200:
            print(f"OK: Anomaly Slack alert sent for {filename}")
        else:
            print("WARN: Rich payload failed, trying fallback...")
            r2 = requests.post(webhook_url, json=fallback_payload, timeout=10)
            if r2.status_code == 200:
                print(f"OK: Anomaly Slack alert sent (Fallback) for {filename}")
            else:
                print(f"ERROR: Slack refused: {r2.status_code} - {r2.text}")
    except Exception as e:
        print(f"WARN: Failed to send anomaly alert: {e}")
