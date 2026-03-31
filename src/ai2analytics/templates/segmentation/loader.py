"""Data loading for the segmentation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ai2analytics.templates.segmentation.config import SegmentationConfig


@dataclass
class SegmentationData:
    """Container for all data loaded by the segmentation data loading stage."""
    entity_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    entity_id_col: str = ""
    original_columns: list[str] = field(default_factory=list)


def load_data(
    cfg: SegmentationConfig,
    spark: Any = None,
    dataframes: dict[str, pd.DataFrame] | None = None,
) -> SegmentationData:
    """Load entity data per config. Returns a SegmentationData container.

    Args:
        cfg: Pipeline configuration.
        spark: PySpark SparkSession (required for Spark table reads).
        dataframes: Optional dict of in-memory DataFrames to use instead of
                    reading from files/tables. Keys:
                    "entity_data" — the entity table.
                    If the key is present, the corresponding path/table config
                    is ignored and the DataFrame is used directly.
    """
    dfs = dataframes or {}
    data = SegmentationData()
    print("=" * 70)
    print("STAGE 1: Loading data")
    print("=" * 70)

    # Load entity table
    if "entity_data" in dfs:
        data.entity_df = dfs["entity_data"].copy()
        print(f"  Entity data:  {len(data.entity_df):,} rows (in-memory)")
    else:
        if cfg.entity_source_type == "csv":
            data.entity_df = pd.read_csv(cfg.entity_table)
            print(f"  Entity data:  {len(data.entity_df):,} rows (CSV)")
        else:
            if spark is None:
                raise RuntimeError("spark session is required for loading tables")
            data.entity_df = spark.table(cfg.entity_table).toPandas()
            print(f"  Entity data:  {len(data.entity_df):,} rows (Spark)")

    data.entity_id_col = cfg.col_entity_id
    data.original_columns = list(data.entity_df.columns)

    # Validate entity_id column exists
    if cfg.col_entity_id not in data.entity_df.columns:
        raise ValueError(
            f"Entity ID column '{cfg.col_entity_id}' not found in data. "
            f"Available columns: {list(data.entity_df.columns)}"
        )

    # Auto-detect numeric feature columns if feature_columns is empty
    if cfg.feature_columns:
        missing = [c for c in cfg.feature_columns if c not in data.entity_df.columns]
        if missing:
            raise ValueError(f"Feature columns not found in data: {missing}")
        print(f"  Feature cols: {len(cfg.feature_columns)} specified")
    else:
        numeric_cols = data.entity_df.select_dtypes(include=["number"]).columns.tolist()
        # Exclude the entity ID column and any explicit exclude columns
        auto_features = [
            c for c in numeric_cols
            if c != cfg.col_entity_id and c not in cfg.exclude_columns
        ]
        cfg.feature_columns = auto_features
        print(f"  Feature cols: {len(auto_features)} auto-detected (numeric)")

    # Drop rows with no entity ID
    before = len(data.entity_df)
    data.entity_df = data.entity_df.dropna(subset=[cfg.col_entity_id])
    dropped = before - len(data.entity_df)
    if dropped:
        print(f"  Dropped {dropped:,} rows with missing entity ID")

    print("  Done.\n")
    return data
