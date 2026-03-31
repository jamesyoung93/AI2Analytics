"""Pipeline orchestrator for segmentation — runs all stages end-to-end."""

from __future__ import annotations

from typing import Any

from ai2analytics.templates.base import BaseTemplate, TableRequirement, ColumnRequirement
from ai2analytics.templates.registry import register
from ai2analytics.templates.segmentation.config import SegmentationConfig
from ai2analytics.templates.segmentation.loader import SegmentationData, load_data
from ai2analytics.templates.segmentation.features import PreparedFeatures, prepare_features
from ai2analytics.templates.segmentation.model import SegmentationResult, fit_segments
from ai2analytics.templates.segmentation.output import (
    SegmentationOutput, build_output, write_output, plot_segments,
)


@register
class SegmentationPipeline(BaseTemplate):
    """Entity segmentation pipeline using clustering algorithms.

    Stages:
        1. Data Loading
        2. Feature Preparation (normalize, impute, optional PCA)
        3. Segment Fitting (KMeans / Hierarchical / Auto)
        4. Output Building (assignments + profiles)
        5. Output Writing (CSV / Spark table)

    Usage in Databricks notebook::

        from ai2analytics.templates.segmentation import (
            SegmentationConfig, SegmentationPipeline,
        )

        cfg = SegmentationConfig(
            analysis_name="hcp_segments",
            entity_table="schema.hcp_features",
            col_entity_id="npi",
            n_segments=5,
            method="kmeans",
            output_csv="/dbfs/mnt/output/segments.csv",
        )

        pipeline = SegmentationPipeline()
        results = pipeline.run(cfg, spark=spark)
        pipeline.show_dashboard(results)

    Or with in-memory DataFrames::

        pipeline = SegmentationPipeline()
        results = pipeline.run(cfg, dataframes={"entity_data": df})
    """

    name = "segmentation"
    description = (
        "Entity segmentation using KMeans or hierarchical clustering "
        "with automatic feature selection and optional PCA"
    )
    config_class = SegmentationConfig

    required_tables = [
        TableRequirement(
            key="entity_data",
            description="One row per entity with numeric feature columns for clustering",
            source_type="spark_table",
            config_field="entity_table",
            required_columns=[
                ColumnRequirement(
                    "entity_id", "string",
                    "Unique entity identifier",
                    aliases=["id", "npi", "customer_id", "hcp_id"],
                    config_field="col_entity_id",
                ),
            ],
            optional_columns=[
                ColumnRequirement(
                    "feature", "numeric",
                    "Any numeric column used as a clustering feature",
                ),
            ],
        ),
    ]

    def __init__(self):
        self._data: SegmentationData | None = None
        self._features: PreparedFeatures | None = None
        self._result: SegmentationResult | None = None
        self._output: SegmentationOutput | None = None

    def run(
        self,
        cfg: SegmentationConfig,
        spark: Any = None,
        dataframes: dict[str, Any] | None = None,
    ) -> SegmentationOutput:
        """Run the full segmentation pipeline end-to-end.

        Args:
            cfg: Pipeline configuration.
            spark: PySpark SparkSession (required for Spark table reads/writes).
            dataframes: Optional dict of in-memory pandas DataFrames to use
                        instead of reading from files/tables. Keys:
                        "entity_data" — the entity feature table.
                        If the key is present, the corresponding table config
                        is ignored and the DataFrame is used directly.
        """
        errors = cfg.validate(dataframes=dataframes)
        if errors:
            raise ValueError(f"Config validation failed:\n  " + "\n  ".join(errors))

        print("\n" + "=" * 70)
        print(f"  PIPELINE: {cfg.analysis_name} Segmentation")
        print("=" * 70 + "\n")

        # 1. Load data
        self._data = load_data(cfg, spark=spark, dataframes=dataframes)

        # 2. Prepare features
        self._features = prepare_features(cfg, self._data)

        # 3. Fit segments
        self._result = fit_segments(cfg, self._features)

        # 4. Build output
        self._output = build_output(cfg, self._data, self._features, self._result)

        # 5. Write output
        write_output(cfg, self._output, spark=spark)

        self._print_summary(cfg)
        return self._output

    def show_dashboard(self, output: SegmentationOutput | None = None):
        """Show diagnostic plots."""
        out = output or self._output
        if out is None or self._result is None or self._features is None:
            print("Run the pipeline first.")
            return
        plot_segments(self._features, self._result, out)

    def _print_summary(self, cfg: SegmentationConfig):
        """Print final pipeline summary."""
        print("\n" + "=" * 70)
        print("PIPELINE COMPLETE")
        print("=" * 70)
        print(f"  Analysis:     {cfg.analysis_name}")
        if self._output is not None:
            stats = self._output.summary_stats
            print(f"  Entities:     {stats.get('n_entities', 0):,}")
            print(f"  Segments:     {stats.get('n_segments', 0)}")
            print(f"  Method:       {stats.get('method', '')}")
            print(f"  Silhouette:   {stats.get('silhouette_score', 0):.4f}")
            sizes = stats.get("segment_sizes", {})
            for seg in sorted(sizes.keys()):
                print(f"    Segment {seg}: {sizes[seg]:,} entities")
        if cfg.output_csv:
            print(f"  Output CSV:   {cfg.output_csv}")
        if cfg.output_table:
            print(f"  Output table: {cfg.output_table}")
        print("=" * 70)
