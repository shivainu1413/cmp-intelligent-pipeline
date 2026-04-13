"""
Schema Validation Module
Defines the golden schema and validates incoming data against it.
"""
import polars as pl
from typing import List


def get_golden_schema() -> pl.Schema:
    """
    Define the standard schema for CMP sensor logs.
    All incoming data must conform to this format.
    """
    dummy_df = pl.DataFrame({
        "timestamp": ["2023-01-01T00:00:00"],
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
    return dummy_df.schema


def compare_types_recursively(
    col_name: str,
    expected: pl.DataType,
    actual: pl.DataType
) -> List[str]:
    """
    Recursively compare two Polars DataTypes.
    Handles Struct (nested fields) and List types.
    Returns a list of mismatch descriptions.
    """
    errors = []

    if expected == actual:
        return []

    if isinstance(expected, pl.Struct) and isinstance(actual, pl.Struct):
        exp_fields = {f.name: f.dtype for f in expected.fields}
        act_fields = {f.name: f.dtype for f in actual.fields}

        exp_keys = set(exp_fields.keys())
        act_keys = set(act_fields.keys())

        missing = exp_keys - act_keys
        if missing:
            errors.append(f"Column '{col_name}': Missing nested field(s) {missing}")

        extra = act_keys - exp_keys
        if extra:
            errors.append(f"Column '{col_name}': Found extra nested field(s) {extra}")

        for key in exp_keys.intersection(act_keys):
            nested_errors = compare_types_recursively(
                f"{col_name}.{key}", exp_fields[key], act_fields[key]
            )
            errors.extend(nested_errors)
        return errors

    if isinstance(expected, pl.List) and isinstance(actual, pl.List):
        return compare_types_recursively(
            f"{col_name}[]", expected.inner, actual.inner
        )

    errors.append(
        f"Column '{col_name}': Type mismatch. Expected: {expected}, Actual: {actual}"
    )
    return errors


def verify_schema(expected_schema: pl.Schema, actual_schema: pl.Schema) -> List[str]:
    """
    Compare golden schema against actual schema.
    Returns a list of errors (empty list = passed).
    """
    error_logs = []

    orig_keys = set(expected_schema.keys())
    new_keys = set(actual_schema.keys())

    if orig_keys - new_keys:
        error_logs.append(f"[Top-Level Missing] {orig_keys - new_keys}")
    if new_keys - orig_keys:
        error_logs.append(f"[Top-Level Extra] {new_keys - orig_keys}")

    for col in orig_keys.intersection(new_keys):
        diffs = compare_types_recursively(col, expected_schema[col], actual_schema[col])
        if diffs:
            error_logs.extend(diffs)

    return error_logs
