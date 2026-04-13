"""
Tests for notifier module.
Verifies that Slack notification functions handle all edge cases without crashing.
Note: These are unit tests using mocks — no real Slack messages are sent.
"""
import polars as pl
from unittest.mock import patch, MagicMock
from notifier import send_schema_alert, send_anomaly_alert


class TestSendSchemaAlert:
    """Tests for schema violation notifications."""

    def test_empty_webhook_does_not_crash(self):
        """Empty webhook URL should not raise an exception."""
        send_schema_alert("", "test.json", ["error1"])

    @patch("notifier.requests.post")
    def test_sends_request_with_correct_url(self, mock_post):
        """Should POST to the specified webhook URL."""
        mock_post.return_value = MagicMock(status_code=200)
        send_schema_alert("https://hooks.slack.com/test", "bad_file.json", ["missing field"])
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert args[0] == "https://hooks.slack.com/test"

    @patch("notifier.requests.post", side_effect=Exception("Network error"))
    def test_network_error_does_not_crash(self, mock_post):
        """Network errors should not crash the function."""
        send_schema_alert("https://hooks.slack.com/test", "file.json", ["error"])


class TestSendAnomalyAlert:
    """Tests for anomaly + AI diagnosis notifications."""

    def _make_anomaly_df(self) -> pl.DataFrame:
        return pl.DataFrame({
            "timestamp": ["2024-01-01T00:00:00"],
            "slurry_flow_rate_ml_min": [150.0],
            "motor_current_amps": [22.0],
            "error_code": ["ALM-3050"],
            "is_anomaly": [True],
        })

    def test_empty_webhook_does_not_crash(self):
        """Empty webhook URL should not raise an exception."""
        df = self._make_anomaly_df()
        send_anomaly_alert("", "anomaly.csv", df, "Some AI suggestion")

    @patch("notifier.requests.post")
    def test_sends_request_on_valid_webhook(self, mock_post):
        """Valid webhook should trigger a POST request."""
        mock_post.return_value = MagicMock(status_code=200)
        df = self._make_anomaly_df()
        send_anomaly_alert("https://hooks.slack.com/test", "anomaly.csv", df, "Fix the pump")
        assert mock_post.called

    @patch("notifier.requests.post")
    def test_handles_long_ai_suggestion(self, mock_post):
        """Very long AI suggestions should not crash (Slack has character limits)."""
        mock_post.return_value = MagicMock(status_code=200)
        df = self._make_anomaly_df()
        long_suggestion = "Fix the pump. " * 500
        send_anomaly_alert("https://hooks.slack.com/test", "anomaly.csv", df, long_suggestion)
        assert mock_post.called
