"""
Tests for schema_validator module.
Verifies golden schema definition and schema comparison logic.
"""
import polars as pl
from schema_validator import get_golden_schema, verify_schema, compare_types_recursively


class TestGoldenSchema:
    """Tests for the golden schema definition."""

    def test_golden_schema_has_all_columns(self):
        """Golden schema should have 6 columns."""
        schema = get_golden_schema()
        expected_columns = {"timestamp", "machine_id", "recipe_id", "metrics", "status", "error_code"}
        assert set(schema.keys()) == expected_columns

    def test_golden_schema_timestamp_is_datetime(self):
        """Timestamp should be Datetime type."""
        schema = get_golden_schema()
        assert schema["timestamp"] == pl.Datetime("us")

    def test_golden_schema_metrics_is_struct(self):
        """Metrics should be a Struct type (nested fields)."""
        schema = get_golden_schema()
        assert isinstance(schema["metrics"], pl.Struct)

    def test_golden_schema_metrics_has_four_fields(self):
        """Metrics struct should contain 4 sensor fields."""
        schema = get_golden_schema()
        metrics_fields = {f.name for f in schema["metrics"].fields}
        expected = {"slurry_flow_rate_ml_min", "motor_current_amps", "head_down_force_psi", "platen_temp_c"}
        assert metrics_fields == expected


class TestVerifySchema:
    """Tests for schema validation logic."""

    def _make_valid_df(self) -> pl.DataFrame:
        """Create a test DataFrame that conforms to the golden schema."""
        return pl.DataFrame({
            "timestamp": ["2024-01-01T00:00:00"],
            "machine_id": ["CMP-TOOL-01"],
            "recipe_id": ["RECIPE_A"],
            "metrics": [{
                "slurry_flow_rate_ml_min": 200.0,
                "motor_current_amps": 15.0,
                "head_down_force_psi": 5.0,
                "platen_temp_c": 60.0
            }],
            "status": ["RUNNING"],
            "error_code": ["None"]
        }).with_columns(pl.col("timestamp").str.to_datetime())

    def test_valid_schema_passes(self):
        """Valid data should pass validation (return empty list)."""
        golden = get_golden_schema()
        valid_df = self._make_valid_df()
        errors = verify_schema(golden, valid_df.schema)
        assert errors == []

    def test_missing_column_detected(self):
        """Missing column should be detected."""
        golden = get_golden_schema()
        bad_df = self._make_valid_df().drop("machine_id")
        errors = verify_schema(golden, bad_df.schema)
        assert len(errors) > 0
        assert any("Missing" in e for e in errors)

    def test_extra_column_detected(self):
        """Extra column should be detected."""
        golden = get_golden_schema()
        bad_df = self._make_valid_df().with_columns(
            pl.lit("unexpected").alias("extra_field")
        )
        errors = verify_schema(golden, bad_df.schema)
        assert len(errors) > 0
        assert any("Extra" in e for e in errors)

    def test_type_mismatch_detected(self):
        """Type mismatch should be detected (e.g. machine_id as Int instead of String)."""
        golden = get_golden_schema()
        bad_df = pl.DataFrame({
            "timestamp": ["2024-01-01T00:00:00"],
            "machine_id": [123],  # Should be String, given Int
            "recipe_id": ["RECIPE_A"],
            "metrics": [{
                "slurry_flow_rate_ml_min": 200.0,
                "motor_current_amps": 15.0,
                "head_down_force_psi": 5.0,
                "platen_temp_c": 60.0
            }],
            "status": ["RUNNING"],
            "error_code": ["None"]
        }).with_columns(pl.col("timestamp").str.to_datetime())
        errors = verify_schema(golden, bad_df.schema)
        assert len(errors) > 0
        assert any("mismatch" in e.lower() for e in errors)

    def test_identical_schemas_pass(self):
        """Identical schemas should pass."""
        schema = get_golden_schema()
        errors = verify_schema(schema, schema)
        assert errors == []


class TestCompareTypesRecursively:
    """Tests for recursive type comparison."""

    def test_same_type_no_errors(self):
        """Same type should return empty list."""
        errors = compare_types_recursively("col", pl.Float64, pl.Float64)
        assert errors == []

    def test_different_type_has_error(self):
        """Different types should return an error."""
        errors = compare_types_recursively("col", pl.Float64, pl.Int64)
        assert len(errors) == 1
        assert "mismatch" in errors[0].lower()

    def test_nested_struct_missing_field(self):
        """Missing field in nested Struct should be detected."""
        expected = pl.Struct({"a": pl.Float64, "b": pl.Float64})
        actual = pl.Struct({"a": pl.Float64})  # Missing b
        errors = compare_types_recursively("metrics", expected, actual)
        assert len(errors) > 0
        assert any("Missing" in e for e in errors)
