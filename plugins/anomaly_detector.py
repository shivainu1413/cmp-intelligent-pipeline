"""
Anomaly Detection Module
Identifies abnormal records from CMP sensor data.
"""
import polars as pl


def flatten_metrics(df: pl.DataFrame) -> pl.DataFrame:
    """
    Flatten the nested metrics struct column and add an is_anomaly flag.
    Input: DataFrame with metrics (Struct) column
    Output: Flattened DataFrame with is_anomaly column added
    """
    flat_df = df.unnest("metrics")
    flat_df = flat_df.with_columns(
        pl.col("error_code").is_not_null().alias("is_anomaly")
    )
    return flat_df


def detect_anomalies(flat_df: pl.DataFrame) -> pl.DataFrame:
    """
    Filter abnormal records from a flattened DataFrame.
    Current rule: error_code is not null → anomaly.
    Can be extended with threshold-based rules in the future.
    """
    return flat_df.filter(pl.col("is_anomaly") == True)
