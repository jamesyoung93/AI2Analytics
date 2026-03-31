"""Feature preparation for the segmentation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from ai2analytics.templates.segmentation.config import SegmentationConfig
from ai2analytics.templates.segmentation.loader import SegmentationData


@dataclass
class PreparedFeatures:
    """Container for prepared feature data ready for clustering."""
    entity_ids: np.ndarray = field(default_factory=lambda: np.array([]))
    feature_matrix: np.ndarray = field(default_factory=lambda: np.array([]))
    feature_names: list[str] = field(default_factory=list)
    pca_components: np.ndarray | None = None
    explained_variance: np.ndarray | None = None


def prepare_features(
    cfg: SegmentationConfig,
    data: SegmentationData,
) -> PreparedFeatures:
    """Prepare features for clustering: impute, normalize, optionally PCA.

    Args:
        cfg: Pipeline configuration.
        data: Loaded entity data.

    Returns:
        PreparedFeatures with entity_ids, feature_matrix, and metadata.
    """
    print("=" * 70)
    print("STAGE 2: Preparing features")
    print("=" * 70)

    features = PreparedFeatures()
    df = data.entity_df.copy()

    # Extract entity IDs
    features.entity_ids = df[cfg.col_entity_id].values
    features.feature_names = list(cfg.feature_columns)

    # Extract feature matrix
    feature_df = df[cfg.feature_columns].copy()

    # Handle missing values
    if cfg.handle_missing == "drop":
        mask = feature_df.notna().all(axis=1)
        feature_df = feature_df.loc[mask]
        features.entity_ids = features.entity_ids[mask.values]
        print(f"  Missing handling: dropped {(~mask).sum():,} rows with NaN")
    elif cfg.handle_missing == "median":
        for col in feature_df.columns:
            median_val = feature_df[col].median()
            feature_df[col] = feature_df[col].fillna(median_val)
        print("  Missing handling: filled with column median")
    elif cfg.handle_missing == "mean":
        for col in feature_df.columns:
            mean_val = feature_df[col].mean()
            feature_df[col] = feature_df[col].fillna(mean_val)
        print("  Missing handling: filled with column mean")
    elif cfg.handle_missing == "zero":
        feature_df = feature_df.fillna(0)
        print("  Missing handling: filled with zero")

    # Convert to numeric and fill remaining NaN
    for col in feature_df.columns:
        feature_df[col] = pd.to_numeric(feature_df[col], errors="coerce").fillna(0)

    matrix = feature_df.values.astype(np.float64)
    print(f"  Feature matrix:   {matrix.shape[0]:,} entities x {matrix.shape[1]} features")

    # Normalize features
    if cfg.normalize:
        if cfg.normalization_method == "standard":
            scaler = StandardScaler()
        elif cfg.normalization_method == "minmax":
            scaler = MinMaxScaler()
        elif cfg.normalization_method == "robust":
            scaler = RobustScaler()
        else:
            scaler = StandardScaler()
        matrix = scaler.fit_transform(matrix)
        print(f"  Normalization:    {cfg.normalization_method}")

    # Optional PCA
    if cfg.use_pca:
        pca = PCA(n_components=cfg.pca_variance_threshold, svd_solver="full")
        matrix = pca.fit_transform(matrix)
        features.pca_components = pca.components_
        features.explained_variance = pca.explained_variance_ratio_
        print(
            f"  PCA:              {pca.n_components_} components "
            f"({pca.explained_variance_ratio_.sum():.1%} variance)"
        )

    features.feature_matrix = matrix
    print("  Done.\n")
    return features
