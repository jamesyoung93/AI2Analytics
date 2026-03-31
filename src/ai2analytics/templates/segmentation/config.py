"""Configuration dataclass for the segmentation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SegmentationConfig:
    """Full configuration for the entity segmentation pipeline.

    All column names, table paths, and clustering parameters are configurable
    so the same pipeline works across entities, brands, and data sources.
    """

    # -- Identity ────────────────────────────────────────────────────────
    analysis_name: str = "segmentation"
    entity_table: str = ""
    entity_source_type: str = "spark_table"  # "spark_table", "csv", "delta"

    # -- Column mappings ─────────────────────────────────────────────────
    col_entity_id: str = "entity_id"
    feature_columns: list[str] = field(default_factory=list)
    exclude_columns: list[str] = field(default_factory=list)

    # -- Clustering parameters ───────────────────────────────────────────
    n_segments: int = 4
    method: str = "kmeans"  # "kmeans", "hierarchical", "auto"
    normalize: bool = True
    normalization_method: str = "standard"  # "standard", "minmax", "robust"
    use_pca: bool = False
    pca_variance_threshold: float = 0.95

    # -- Missing data handling ───────────────────────────────────────────
    handle_missing: str = "median"  # "median", "mean", "zero", "drop"

    # -- Auto-select k ───────────────────────────────────────────────────
    auto_select_k: bool = False
    k_range: tuple[int, int] = (2, 10)

    # -- Output ──────────────────────────────────────────────────────────
    output_table: str = ""
    output_csv: str = ""

    def validate(self, dataframes: dict | None = None) -> list[str]:
        """Return a list of validation errors (empty if valid).

        Args:
            dataframes: If provided, skip path validation for keys that have
                        in-memory DataFrames (e.g. {"entity_data": df}).
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

        if _empty(self.entity_table) and "entity_data" not in dfs:
            errors.append(
                "entity_table is required (or pass dataframes={'entity_data': df})"
            )
        if self.method not in ("kmeans", "hierarchical", "auto"):
            errors.append(
                f"method must be 'kmeans', 'hierarchical', or 'auto', got '{self.method}'"
            )
        if self.normalization_method not in ("standard", "minmax", "robust"):
            errors.append(
                f"normalization_method must be 'standard', 'minmax', or 'robust', "
                f"got '{self.normalization_method}'"
            )
        if self.handle_missing not in ("median", "mean", "zero", "drop"):
            errors.append(
                f"handle_missing must be 'median', 'mean', 'zero', or 'drop', "
                f"got '{self.handle_missing}'"
            )
        if self.n_segments < 2:
            errors.append("n_segments must be >= 2")
        if self.auto_select_k:
            k_lo, k_hi = self.k_range
            if k_lo < 2 or k_hi < k_lo:
                errors.append(
                    f"k_range must satisfy 2 <= k_lo <= k_hi, got {self.k_range}"
                )
        if self.use_pca:
            if not (0.0 < self.pca_variance_threshold <= 1.0):
                errors.append(
                    "pca_variance_threshold must be in (0, 1]"
                )
        if _empty(self.output_table) and _empty(self.output_csv):
            errors.append("At least one of output_table or output_csv is required")
        return errors

    @classmethod
    def from_dict(cls, d: dict) -> "SegmentationConfig":
        """Create config from a flat dictionary."""
        key_map = {
            "analysis_name": "analysis_name",
            "entity_table": "entity_table",
            "entity_source_type": "entity_source_type",
            "col_entity_id": "col_entity_id",
            "feature_columns": "feature_columns",
            "exclude_columns": "exclude_columns",
            "n_segments": "n_segments",
            "method": "method",
            "normalize": "normalize",
            "normalization_method": "normalization_method",
            "use_pca": "use_pca",
            "pca_variance_threshold": "pca_variance_threshold",
            "handle_missing": "handle_missing",
            "auto_select_k": "auto_select_k",
            "k_range": "k_range",
            "output_table": "output_table",
            "output_csv": "output_csv",
        }
        kwargs = {}
        for old_key, new_key in key_map.items():
            if old_key in d:
                val = d[old_key]
                # Convert list to tuple for k_range
                if new_key == "k_range" and isinstance(val, list):
                    val = tuple(val)
                kwargs[new_key] = val
        return cls(**kwargs)
