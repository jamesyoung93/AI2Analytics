"""Clustering model fitting for the segmentation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score

from ai2analytics.templates.segmentation.config import SegmentationConfig
from ai2analytics.templates.segmentation.features import PreparedFeatures


@dataclass
class SegmentationResult:
    """Container for clustering results."""
    labels: np.ndarray = field(default_factory=lambda: np.array([]))
    n_segments: int = 0
    silhouette_score: float = 0.0
    method_used: str = ""
    cluster_centers: np.ndarray | None = None
    k_scores: dict[int, float] = field(default_factory=dict)


def _fit_kmeans(matrix: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Fit KMeans and return (labels, centers)."""
    model = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = model.fit_predict(matrix)
    return labels, model.cluster_centers_


def _fit_hierarchical(matrix: np.ndarray, k: int) -> np.ndarray:
    """Fit AgglomerativeClustering and return labels."""
    model = AgglomerativeClustering(n_clusters=k)
    return model.fit_predict(matrix)


def _select_best_k(
    matrix: np.ndarray,
    k_range: tuple[int, int],
    method: str,
) -> tuple[int, dict[int, float]]:
    """Iterate over k values and pick the best by silhouette score.

    Returns:
        (best_k, {k: silhouette_score}).
    """
    k_lo, k_hi = k_range
    scores: dict[int, float] = {}

    for k in range(k_lo, k_hi + 1):
        if method == "kmeans":
            labels, _ = _fit_kmeans(matrix, k)
        else:
            labels = _fit_hierarchical(matrix, k)
        score = silhouette_score(matrix, labels)
        scores[k] = score
        print(f"    k={k}: silhouette={score:.4f}")

    best_k = max(scores, key=scores.get)
    print(f"    Best k={best_k} (silhouette={scores[best_k]:.4f})")
    return best_k, scores


def fit_segments(
    cfg: SegmentationConfig,
    features: PreparedFeatures,
) -> SegmentationResult:
    """Fit clustering model per config and return results.

    Args:
        cfg: Pipeline configuration.
        features: Prepared feature data.

    Returns:
        SegmentationResult with labels, scores, and metadata.
    """
    print("=" * 70)
    print("STAGE 3: Fitting segments")
    print("=" * 70)

    matrix = features.feature_matrix
    result = SegmentationResult()

    if cfg.method == "auto":
        # Try both methods and pick best silhouette
        print("  Method: auto (trying both kmeans and hierarchical)")

        # Determine k
        if cfg.auto_select_k:
            print("  Auto-selecting k (kmeans)...")
            k_km, scores_km = _select_best_k(matrix, cfg.k_range, "kmeans")
            print("  Auto-selecting k (hierarchical)...")
            k_hc, scores_hc = _select_best_k(matrix, cfg.k_range, "hierarchical")
        else:
            k_km = cfg.n_segments
            k_hc = cfg.n_segments
            scores_km = {}
            scores_hc = {}

        # Fit both at their best k
        labels_km, centers_km = _fit_kmeans(matrix, k_km)
        sil_km = silhouette_score(matrix, labels_km)

        labels_hc = _fit_hierarchical(matrix, k_hc)
        sil_hc = silhouette_score(matrix, labels_hc)

        print(f"  KMeans       (k={k_km}): silhouette={sil_km:.4f}")
        print(f"  Hierarchical (k={k_hc}): silhouette={sil_hc:.4f}")

        if sil_km >= sil_hc:
            result.labels = labels_km
            result.n_segments = k_km
            result.silhouette_score = sil_km
            result.method_used = "kmeans"
            result.cluster_centers = centers_km
            result.k_scores = scores_km
            print("  Selected: kmeans")
        else:
            result.labels = labels_hc
            result.n_segments = k_hc
            result.silhouette_score = sil_hc
            result.method_used = "hierarchical"
            result.cluster_centers = None
            result.k_scores = scores_hc
            print("  Selected: hierarchical")

    elif cfg.method == "kmeans":
        print("  Method: kmeans")

        if cfg.auto_select_k:
            print("  Auto-selecting k...")
            best_k, k_scores = _select_best_k(matrix, cfg.k_range, "kmeans")
            result.k_scores = k_scores
        else:
            best_k = cfg.n_segments

        labels, centers = _fit_kmeans(matrix, best_k)
        result.labels = labels
        result.n_segments = best_k
        result.silhouette_score = silhouette_score(matrix, labels)
        result.method_used = "kmeans"
        result.cluster_centers = centers

    elif cfg.method == "hierarchical":
        print("  Method: hierarchical")

        if cfg.auto_select_k:
            print("  Auto-selecting k...")
            best_k, k_scores = _select_best_k(matrix, cfg.k_range, "hierarchical")
            result.k_scores = k_scores
        else:
            best_k = cfg.n_segments

        labels = _fit_hierarchical(matrix, best_k)
        result.labels = labels
        result.n_segments = best_k
        result.silhouette_score = silhouette_score(matrix, labels)
        result.method_used = "hierarchical"
        result.cluster_centers = None

    print(f"  Segments:     {result.n_segments}")
    print(f"  Silhouette:   {result.silhouette_score:.4f}")
    print("  Done.\n")
    return result
