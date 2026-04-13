"""
Tests for anomaly_detector module.
Verifies metrics flattening and anomaly filtering logic.
"""
import polars as pl
from anomaly_detector import flatten_metrics, detect_anomalies


def _make_test_df(error_codes: list) -> pl.DataFrame:
    """
    Create a test DataFrame.
    error_codes: error_code for each row; None means normal.
    """
    n = len(error_codes)
    return pl.DataFrame({
        "timestamp": [f"2024-01-01T00:00:{i:02d}" for i in range(n)],
        "machine_id": ["CMP-TOOL-01"] * n,
        "recipe_id": ["RECIPE_A"] * n,
        "metrics": [{
            "slurry_flow_rate_ml_min": 200.0,
            "motor_current_amps": 15.0,
            "head_down_force_psi": 5.0,
            "platen_temp_c": 60.0
        }] * n,
        "status": ["RUNNING"] * n,
        "error_code": error_codes
    })


class TestFlattenMetrics:
    """Tests for metrics flattening."""

    def test_metrics_columns_appear(self):
        """Flattened output should contain the 4 sensor columns."""
        df = _make_test_df([None])
        flat = flatten_metrics(df)
        assert "slurry_flow_rate_ml_min" in flat.columns
        assert "motor_current_amps" in flat.columns
        assert "head_down_force_psi" in flat.columns
        assert "platen_temp_c" in flat.columns

    def test_metrics_column_removed(self):
        """Original metrics column should be removed after flattening."""
        df = _make_test_df([None])
        flat = flatten_metrics(df)
        assert "metrics" not in flat.columns

    def test_is_anomaly_column_added(self):
        """Flattened output should include an is_anomaly column."""
        df = _make_test_df([None])
        flat = flatten_metrics(df)
        assert "is_anomaly" in flat.columns

    def test_normal_record_not_anomaly(self):
        """Record with null error_code should have is_anomaly = False."""
        df = _make_test_df([None])
        flat = flatten_metrics(df)
        assert flat["is_anomaly"][0] == False

    def test_error_record_is_anomaly(self):
        """Record with error_code should have is_anomaly = True."""
        df = _make_test_df(["ALM-3050"])
        flat = flatten_metrics(df)
        assert flat["is_anomaly"][0] == True

    def test_row_count_preserved(self):
        """Flattening should not change the number of rows."""
        df = _make_test_df([None, "ALM-3050", None])
        flat = flatten_metrics(df)
        assert len(flat) == 3


class TestDetectAnomalies:
    """Tests for anomaly detection."""

    def test_no_anomalies_returns_empty(self):
        """All-normal data should return empty result."""
        df = _make_test_df([None, None, None])
        flat = flatten_metrics(df)
        anomalies = detect_anomalies(flat)
        assert anomalies.is_empty()

    def test_all_anomalies_returned(self):
        """All-anomaly data should be fully returned."""
        df = _make_test_df(["ALM-3050", "ERR-9999"])
        flat = flatten_metrics(df)
        anomalies = detect_anomalies(flat)
        assert len(anomalies) == 2

    def test_mixed_data_filters_correctly(self):
        """Mixed data should return only anomalous records."""
        df = _make_test_df([None, "ALM-3050", None, "ERR-9999", None])
        flat = flatten_metrics(df)
        anomalies = detect_anomalies(flat)
        assert len(anomalies) == 2

    def test_anomaly_preserves_error_code(self):
        """Filtered anomalies should preserve the original error_code."""
        df = _make_test_df(["ALM-3050"])
        flat = flatten_metrics(df)
        anomalies = detect_anomalies(flat)
        assert anomalies["error_code"][0] == "ALM-3050"
