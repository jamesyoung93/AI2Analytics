"""Configuration dataclass for the market mix modeling pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AdstockConfig:
    """Per-channel adstock and saturation parameters."""
    channel_name: str = ""
    decay_rate: float = 0.5
    saturation_type: str = "hill"  # "hill", "log", "none"
    saturation_half_max: float = 1.0
    saturation_steepness: float = 1.0


@dataclass
class MarketMixConfig:
    """Full configuration for the market mix modeling pipeline.

    All column names, parameters, and output paths are configurable so
    the same pipeline works across brands, regions, and data sources.
    """

    # -- Identity --------------------------------------------------------
    analysis_name: str = "MMM_Analysis"

    # -- Source table (Spark / Databricks) --------------------------------
    time_series_table: str = ""
    time_series_source_type: str = "spark_table"  # "spark_table", "csv", "delta"

    # -- Column name mappings --------------------------------------------
    col_date: str = "date"
    col_response: str = "revenue"
    media_columns: list[str] = field(default_factory=list)
    control_columns: list[str] = field(default_factory=list)

    # -- Time series settings --------------------------------------------
    frequency: str = "weekly"  # "weekly", "daily", "monthly"

    # -- Adstock & saturation settings -----------------------------------
    adstock_configs: list[AdstockConfig] = field(default_factory=list)
    default_decay_rate: float = 0.5
    default_saturation: str = "hill"  # "hill", "log", "none"

    # -- Structural features ---------------------------------------------
    include_trend: bool = True
    include_seasonality: bool = True
    seasonality_period: int = 52

    # -- Model settings --------------------------------------------------
    model_type: str = "ridge"  # "ridge", "ols", "lasso"
    alpha: float = 1.0
    fit_intercept: bool = True
    positive_coefficients: bool = True

    # -- Output ----------------------------------------------------------
    output_table: str = ""
    output_csv: str = ""

    def validate(self, dataframes: dict | None = None) -> list[str]:
        """Return a list of validation errors (empty if valid).

        Args:
            dataframes: If provided, skip table validation for keys that have
                        in-memory DataFrames (e.g. {"time_series": df}).
        """
        dfs = dataframes or {}
        errors = []

        def _empty(val):
            """Check if a config value is empty (works for str, None, or DataFrame)."""
            if val is None:
                return True
            if isinstance(val, str):
                return val == ""
            return False  # DataFrame or other non-empty object

        if _empty(self.time_series_table) and "time_series" not in dfs:
            errors.append(
                "time_series_table is required (or pass dataframes={'time_series': df})"
            )
        if _empty(self.col_date):
            errors.append("col_date is required")
        if _empty(self.col_response):
            errors.append("col_response is required")
        if not self.media_columns:
            errors.append("media_columns must not be empty")
        if self.model_type not in ("ridge", "ols", "lasso"):
            errors.append(f"model_type must be 'ridge', 'ols', or 'lasso', got '{self.model_type}'")
        if self.alpha < 0:
            errors.append("alpha must be >= 0")
        if self.seasonality_period < 2:
            errors.append("seasonality_period must be >= 2")
        if _empty(self.output_table) and _empty(self.output_csv):
            errors.append("At least one of output_table or output_csv is required")
        return errors

    @classmethod
    def from_dict(cls, d: dict) -> "MarketMixConfig":
        """Create config from a flat dictionary."""
        key_map = {
            "analysis_name": "analysis_name",
            "time_series_table": "time_series_table",
            "time_series_source_type": "time_series_source_type",
            "col_date": "col_date",
            "col_response": "col_response",
            "media_columns": "media_columns",
            "control_columns": "control_columns",
            "frequency": "frequency",
            "adstock_configs": "adstock_configs",
            "default_decay_rate": "default_decay_rate",
            "default_saturation": "default_saturation",
            "include_trend": "include_trend",
            "include_seasonality": "include_seasonality",
            "seasonality_period": "seasonality_period",
            "model_type": "model_type",
            "alpha": "alpha",
            "fit_intercept": "fit_intercept",
            "positive_coefficients": "positive_coefficients",
            "output_table": "output_table",
            "output_csv": "output_csv",
        }
        kwargs = {}
        for old_key, new_key in key_map.items():
            if old_key in d:
                val = d[old_key]
                # Convert list-of-dicts to AdstockConfig objects
                if new_key == "adstock_configs" and isinstance(val, list):
                    val = [
                        AdstockConfig(**item) if isinstance(item, dict) else item
                        for item in val
                    ]
                kwargs[new_key] = val
        return cls(**kwargs)
