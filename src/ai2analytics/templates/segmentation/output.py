"""Output building, writing, and visualization for the segmentation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ai2analytics.templates.segmentation.config import SegmentationConfig
from ai2analytics.templates.segmentation.features import PreparedFeatures
from ai2analytics.templates.segmentation.loader import SegmentationData
from ai2analytics.templates.segmentation.model import SegmentationResult


@dataclass
class SegmentationOutput:
    """Container for segmentation pipeline output."""
    assignments: pd.DataFrame = field(default_factory=pd.DataFrame)
    profiles: pd.DataFrame = field(default_factory=pd.DataFrame)
    summary_stats: dict = field(default_factory=dict)


def build_output(
    cfg: SegmentationConfig,
    data: SegmentationData,
    features: PreparedFeatures,
    result: SegmentationResult,
) -> SegmentationOutput:
    """Build output DataFrames from clustering results.

    Args:
        cfg: Pipeline configuration.
        data: Loaded entity data.
        features: Prepared feature data.
        result: Clustering results.

    Returns:
        SegmentationOutput with assignments, profiles, and summary stats.
    """
    print("=" * 70)
    print("STAGE 4: Building output")
    print("=" * 70)

    output = SegmentationOutput()

    # Build assignments: entity_id + SEGMENT
    assignments = pd.DataFrame({
        cfg.col_entity_id: features.entity_ids,
        "SEGMENT": pd.to_numeric(
            pd.Series(result.labels), errors="coerce"
        ).fillna(0).astype(int),
    })
    output.assignments = assignments
    print(f"  Assignments:  {len(assignments):,} entities")

    # Build profiles: per-segment feature means (use original un-normalized data)
    entity_features = data.entity_df.set_index(cfg.col_entity_id)[cfg.feature_columns]
    # Align to the entity IDs that were actually clustered
    entity_features = entity_features.loc[
        entity_features.index.isin(features.entity_ids)
    ]
    entity_features["SEGMENT"] = result.labels[
        : len(entity_features)
    ]

    profiles = entity_features.groupby("SEGMENT")[cfg.feature_columns].mean()
    profiles.index = pd.to_numeric(profiles.index, errors="coerce").fillna(0).astype(int)
    profiles.index.name = "SEGMENT"
    output.profiles = profiles.reset_index()
    print(f"  Profiles:     {len(output.profiles)} segments x {len(cfg.feature_columns)} features")

    # Summary stats
    segment_counts = assignments["SEGMENT"].value_counts().sort_index()
    output.summary_stats = {
        "n_entities": len(assignments),
        "n_segments": result.n_segments,
        "method": result.method_used,
        "silhouette_score": result.silhouette_score,
        "segment_sizes": segment_counts.to_dict(),
    }
    print("  Done.\n")
    return output


def write_output(
    cfg: SegmentationConfig,
    output: SegmentationOutput,
    spark: Any = None,
) -> None:
    """Write segmentation output to CSV and/or Spark table.

    Args:
        cfg: Pipeline configuration.
        output: Segmentation output data.
        spark: PySpark SparkSession (required for Spark table writes).
    """
    print("=" * 70)
    print("STAGE 5: Writing output")
    print("=" * 70)

    if cfg.output_csv:
        output.assignments.to_csv(cfg.output_csv, index=False)
        print(f"  CSV:   {cfg.output_csv} ({len(output.assignments):,} rows)")

    if cfg.output_table:
        if spark is None:
            raise RuntimeError("spark session is required for writing tables")
        sdf = spark.createDataFrame(output.assignments)
        sdf.write.mode("overwrite").saveAsTable(cfg.output_table)
        print(f"  Table: {cfg.output_table} ({len(output.assignments):,} rows)")

    print("  Done.\n")


def plot_segments(
    features: PreparedFeatures,
    result: SegmentationResult,
    output: SegmentationOutput,
) -> None:
    """Plot segmentation diagnostics: PCA scatter, segment sizes, profile heatmap.

    Args:
        features: Prepared feature data.
        result: Clustering results.
        output: Segmentation output data.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is required for plotting. Install with: pip install matplotlib")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. 2D PCA scatter
    ax = axes[0]
    matrix = features.feature_matrix
    if matrix.shape[1] >= 2:
        # Use first 2 components (already PCA if use_pca was True, else raw)
        if matrix.shape[1] > 2:
            from sklearn.decomposition import PCA as PCA2D
            pca_2d = PCA2D(n_components=2)
            coords = pca_2d.fit_transform(matrix)
        else:
            coords = matrix
        scatter = ax.scatter(
            coords[:, 0], coords[:, 1],
            c=result.labels, cmap="tab10", alpha=0.6, s=15,
        )
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.set_title("Segments (2D PCA)")
        plt.colorbar(scatter, ax=ax, label="Segment")
    else:
        ax.text(0.5, 0.5, "Need >= 2 features\nfor scatter plot",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Segments (2D PCA)")

    # 2. Segment sizes bar chart
    ax = axes[1]
    sizes = output.summary_stats.get("segment_sizes", {})
    segments_sorted = sorted(sizes.keys())
    counts = [sizes[s] for s in segments_sorted]
    bars = ax.bar([str(s) for s in segments_sorted], counts, color="steelblue")
    ax.set_xlabel("Segment")
    ax.set_ylabel("Count")
    ax.set_title("Segment Sizes")
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height(),
            f"{count:,}", ha="center", va="bottom", fontsize=8,
        )

    # 3. Profile heatmap
    ax = axes[2]
    profile_data = output.profiles.set_index("SEGMENT")
    if not profile_data.empty:
        try:
            import seaborn as sns
            sns.heatmap(
                profile_data, ax=ax, cmap="YlOrRd", annot=True,
                fmt=".2f", linewidths=0.5,
            )
        except ImportError:
            im = ax.imshow(profile_data.values, aspect="auto", cmap="YlOrRd")
            ax.set_xticks(range(len(profile_data.columns)))
            ax.set_xticklabels(profile_data.columns, rotation=45, ha="right")
            ax.set_yticks(range(len(profile_data.index)))
            ax.set_yticklabels(profile_data.index)
            plt.colorbar(im, ax=ax)
        ax.set_title("Segment Profiles (Feature Means)")
    else:
        ax.text(0.5, 0.5, "No profile data", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("Segment Profiles")

    plt.tight_layout()
    plt.show()
