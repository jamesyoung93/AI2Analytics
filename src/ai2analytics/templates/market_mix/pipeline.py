"""Pipeline orchestrator for market mix modeling — runs all stages end-to-end."""

from __future__ import annotations

from typing import Any

from ai2analytics.templates.base import BaseTemplate, TableRequirement, ColumnRequirement
from ai2analytics.templates.registry import register
from ai2analytics.templates.market_mix.config import MarketMixConfig
from ai2analytics.templates.market_mix.loader import MarketMixData, load_data
from ai2analytics.templates.market_mix.features import TransformedFeatures, transform_features
from ai2analytics.templates.market_mix.model import MarketMixResult, fit_model
from ai2analytics.templates.market_mix.output import (
    MarketMixOutput, build_output, write_output, plot_market_mix,
)


@register
class MarketMixPipeline(BaseTemplate):
    """Media contribution and ROI pipeline using market mix modeling.

    Stages:
        1. Data Loading
        2. Feature Transformation (adstock, saturation, trend, seasonality)
        3. Model Fitting (Ridge/OLS/Lasso regression)
        4. Output (contributions, ROI, response curves)
        5. Writing (CSV and/or Spark table)

    Usage in Databricks notebook::

        from ai2analytics.templates.market_mix import (
            MarketMixConfig, MarketMixPipeline,
        )

        cfg = MarketMixConfig(
            analysis_name="Brand_X_MMM",
            time_series_table="schema.weekly_media",
            col_date="week_date",
            col_response="revenue",
            media_columns=["tv_spend", "digital_spend", "print_spend"],
            control_columns=["competitor_price", "distribution"],
            output_csv="/dbfs/mnt/output/mmm_results.csv",
        )

        pipeline = MarketMixPipeline()
        results = pipeline.run(cfg, spark=spark)
        pipeline.show_dashboard(results)
    """

    name = "market_mix"
    description = (
        "Media contribution and ROI analysis using market mix modeling "
        "with adstock, saturation, and Ridge/OLS/Lasso regression"
    )
    config_class = MarketMixConfig

    required_tables = [
        TableRequirement(
            key="time_series",
            description="One row per time period with response, media spend, and control variables",
            source_type="spark_table",
            config_field="time_series_table",
            required_columns=[
                ColumnRequirement("date", "date", "Time period date",
                                  aliases=["week_date", "DATE", "WEEK_DATE"],
                                  config_field="col_date"),
                ColumnRequirement("response", "numeric", "Response variable (e.g. revenue, sales)",
                                  aliases=["revenue", "sales", "REVENUE", "SALES"],
                                  config_field="col_response"),
            ],
            optional_columns=[
                ColumnRequirement("media_spend", "numeric",
                                  "Media spend columns (configured via media_columns list)"),
                ColumnRequirement("control_var", "numeric",
                                  "Control variable columns (configured via control_columns list)"),
            ],
        ),
    ]

    def __init__(self):
        self._data: MarketMixData | None = None
        self._features: TransformedFeatures | None = None
        self._result: MarketMixResult | None = None
        self._output: MarketMixOutput | None = None

    def run(
        self,
        cfg: MarketMixConfig,
        spark: Any = None,
        dataframes: dict[str, Any] | None = None,
    ) -> MarketMixOutput:
        """Run the full pipeline end-to-end.

        Args:
            cfg: Pipeline configuration.
            spark: PySpark SparkSession (required for Spark table reads).
            dataframes: Optional dict of in-memory pandas DataFrames to use
                        instead of reading from files/tables. Keys:
                        "time_series".
                        If a key is present, the corresponding table config
                        is ignored and the DataFrame is used directly.
        """
        errors = cfg.validate(dataframes=dataframes)
        if errors:
            raise ValueError(f"Config validation failed:\n  " + "\n  ".join(errors))

        print("\n" + "=" * 70)
        print(f"  PIPELINE: {cfg.analysis_name} Market Mix Model")
        print("=" * 70 + "\n")

        # 1. Load data
        self._data = load_data(cfg, spark=spark, dataframes=dataframes)

        # 2. Feature transformation
        self._features = transform_features(cfg, self._data)

        # 3. Model fitting
        self._result = fit_model(cfg, self._features, self._data)

        # 4. Build output
        self._output = build_output(cfg, self._data, self._features, self._result)

        # 5. Write output
        write_output(cfg, self._output, spark=spark)

        self._print_summary(cfg)
        return self._output

    def show_dashboard(self, output: MarketMixOutput | None = None):
        """Show diagnostic plots."""
        out = output or self._output
        if out is None or self._data is None:
            print("Run the pipeline first.")
            return
        plot_market_mix(out, self._data)

    def _print_summary(self, cfg: MarketMixConfig):
        """Print final pipeline summary."""
        print("\n" + "=" * 70)
        print("PIPELINE COMPLETE")
        print("=" * 70)
        print(f"  Analysis:       {cfg.analysis_name}")
        print(f"  Model:          {cfg.model_type} (alpha={cfg.alpha})")
        if self._result is not None:
            print(f"  R-squared:      {self._result.r_squared:.4f}")
            print(f"  Adj R-squared:  {self._result.adjusted_r_squared:.4f}")
            print(f"  MAPE:           {self._result.mape:.2f}%")
            print(f"  Channels:       {len(cfg.media_columns)}")
            if self._result.channel_roi:
                best_ch = max(self._result.channel_roi, key=self._result.channel_roi.get)
                print(f"  Best ROI:       {best_ch} ({self._result.channel_roi[best_ch]:.4f})")
        if cfg.output_csv:
            print(f"  Output CSV:     {cfg.output_csv}")
        if cfg.output_table:
            print(f"  Output table:   {cfg.output_table}")
        print("=" * 70)
