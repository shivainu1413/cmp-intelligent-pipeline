"""
Tests for the RAG agent module.
Covers: prompt building, response parsing, embedding, and main entry point.
All external calls (Gemini API, MongoDB) are mocked.
"""
import json
from unittest.mock import patch, MagicMock
from rag_agent import (
    build_diagnosis_prompt,
    parse_llm_response,
    format_retrieved_context,
    consult_llm_rag,
    call_gemini,
)


# ----------------------------------------------------------------
# Test data fixtures
# ----------------------------------------------------------------
SAMPLE_ANOMALY = {
    "error_code": "ALM-3050",
    "slurry_flow_rate_ml_min": 155.0,
    "motor_current_amps": 22.5,
    "pad_temperature_celsius": 48.2,
    "downforce_pressure_psi": 3.8,
}

VALID_LLM_JSON = json.dumps({
    "reasoning": "High slurry flow with elevated motor current suggests pad clogging.",
    "root_cause": "Slurry residue buildup on polishing pad",
    "recommended_action": "Replace polishing pad and flush slurry lines",
    "urgency": "high",
    "confidence": 0.87,
    "similar_case_used": 1,
})


# ----------------------------------------------------------------
# Tests: build_diagnosis_prompt
# ----------------------------------------------------------------
class TestBuildDiagnosisPrompt:
    """Verify the prompt is assembled correctly from anomaly data."""

    def test_includes_error_code(self):
        prompt = build_diagnosis_prompt(SAMPLE_ANOMALY, "No cases.")
        assert "ALM-3050" in prompt

    def test_includes_sensor_values(self):
        prompt = build_diagnosis_prompt(SAMPLE_ANOMALY, "No cases.")
        assert "155.0" in prompt
        assert "22.5" in prompt
        assert "48.2" in prompt

    def test_includes_retrieved_context(self):
        context = "Case #1: Pad wear detected"
        prompt = build_diagnosis_prompt(SAMPLE_ANOMALY, context)
        assert "Pad wear detected" in prompt

    def test_handles_missing_fields(self):
        """Should not crash when anomaly record has missing keys."""
        prompt = build_diagnosis_prompt({}, "No cases.")
        assert "UNKNOWN" in prompt
        assert "N/A" in prompt


# ----------------------------------------------------------------
# Tests: parse_llm_response
# ----------------------------------------------------------------
class TestParseLlmResponse:
    """Verify parsing and validation of LLM JSON output."""

    def test_valid_json(self):
        result = parse_llm_response(VALID_LLM_JSON)
        assert result["root_cause"] == "Slurry residue buildup on polishing pad"
        assert result["urgency"] == "high"
        assert result["confidence"] == 0.87
        assert result["parse_error"] is False

    def test_json_with_markdown_fences(self):
        """LLMs sometimes wrap JSON in ```json ... ``` fences."""
        wrapped = f"```json\n{VALID_LLM_JSON}\n```"
        result = parse_llm_response(wrapped)
        assert result["root_cause"] == "Slurry residue buildup on polishing pad"
        assert result["parse_error"] is False

    def test_invalid_json_returns_fallback(self):
        result = parse_llm_response("This is not JSON at all")
        assert result["parse_error"] is True
        assert result["urgency"] == "medium"
        assert result["confidence"] == 0.0
        assert "This is not JSON" in result["reasoning"]

    def test_invalid_urgency_normalized(self):
        bad_urgency = json.dumps({
            "reasoning": "test",
            "root_cause": "test",
            "recommended_action": "test",
            "urgency": "super_critical",
            "confidence": 0.5,
            "similar_case_used": None,
        })
        result = parse_llm_response(bad_urgency)
        assert result["urgency"] == "medium"  # normalized to default

    def test_confidence_clamped(self):
        """Confidence should be clamped between 0.0 and 1.0."""
        over = json.dumps({
            "reasoning": "t", "root_cause": "t", "recommended_action": "t",
            "urgency": "low", "confidence": 5.0, "similar_case_used": None,
        })
        result = parse_llm_response(over)
        assert result["confidence"] == 1.0

        under = json.dumps({
            "reasoning": "t", "root_cause": "t", "recommended_action": "t",
            "urgency": "low", "confidence": -0.5, "similar_case_used": None,
        })
        result = parse_llm_response(under)
        assert result["confidence"] == 0.0

    def test_non_numeric_confidence(self):
        bad = json.dumps({
            "reasoning": "t", "root_cause": "t", "recommended_action": "t",
            "urgency": "low", "confidence": "very high", "similar_case_used": None,
        })
        result = parse_llm_response(bad)
        assert result["confidence"] == 0.5  # default fallback


# ----------------------------------------------------------------
# Tests: format_retrieved_context
# ----------------------------------------------------------------
class TestFormatRetrievedContext:

    def test_empty_cases(self):
        assert format_retrieved_context([]) == "No similar historical cases found."

    def test_formats_cases(self):
        cases = [{
            "score": 0.92,
            "doc": {
                "symptom_desc": "High vibration",
                "root_cause": "Bearing wear",
                "solution_action": "Replace bearing",
            }
        }]
        text = format_retrieved_context(cases)
        assert "Case #1" in text
        assert "0.92" in text
        assert "High vibration" in text
        assert "Replace bearing" in text


# ----------------------------------------------------------------
# Tests: consult_llm_rag (main entry point)
# ----------------------------------------------------------------
class TestConsultLlmRag:

    def test_no_api_key_returns_fallback(self):
        """Without an API key, should return a structured fallback dict."""
        result = consult_llm_rag(SAMPLE_ANOMALY, "/tmp", api_key=None)
        assert isinstance(result, dict)
        assert result["confidence"] == 0.0
        assert result["parse_error"] is False
        assert "No API key" in result["reasoning"]

    @patch("rag_agent.retrieve_similar_cases_mongo", return_value=[])
    @patch("rag_agent.call_gemini", return_value=VALID_LLM_JSON)
    def test_successful_rag_flow(self, mock_gemini, mock_retrieve):
        """Full flow with mocked API: retrieve → prompt → parse."""
        result = consult_llm_rag(SAMPLE_ANOMALY, "/tmp", api_key="fake-key")
        assert result["urgency"] == "high"
        assert result["confidence"] == 0.87
        assert "retrieved_cases" in result
        mock_gemini.assert_called_once()

    @patch("rag_agent.retrieve_similar_cases_mongo", return_value=[])
    @patch("rag_agent.call_gemini", side_effect=RuntimeError("API down"))
    def test_api_failure_returns_error_dict(self, mock_gemini, mock_retrieve):
        """API failure should return error dict, not raise."""
        result = consult_llm_rag(SAMPLE_ANOMALY, "/tmp", api_key="fake-key")
        assert result["parse_error"] is True
        assert "API down" in result["reasoning"]
        assert result["confidence"] == 0.0
