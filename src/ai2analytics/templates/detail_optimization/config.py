"""Configuration dataclass for the detail optimization pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DetailOptimizationConfig:
    """Full configuration for the HCP call allocation pipeline.

    All column names, file paths, and parameters are configurable so
    the same pipeline works across brands, regions, and data sources.
    """

    # ── Identity ────────────────────────────────────────────────────────
    drug_name: str = "DRUG_A"
    drug_portfolio: str = "DRUG_B"

    # ── Source tables (Spark / Databricks) ──────────────────────────────
    hcp_weekly_table: str = ""
    calls_table: str = ""
    hcp_filter_col: str = "TARGET_FLAG"
    hcp_filter_val: str = "Y"

    # ── Alignment / reference files (CSV on DBFS) ──────────────────────
    team_a_align_path: str = ""
    team_b_align_path: str = ""
    portfolio_decile_path: str = ""
    priority_target_path: str = ""
    hcp_reference_path: str = ""

    # ── Column name mappings ───────────────────────────────────────────
    col_npi: str = "npi"
    col_week: str = "WEEK_ENDING"
    col_referrals: str = "PAT_COUNT_REFERRED"
    col_indication: str = "INDC"
    col_calls: str = "HCP_F2F_CALLS"
    col_writer_flag: str = "WRITER_FLAG"
    col_target_flag: str = "TARGET_FLAG"
    col_priority_flag: str = "PRIORITY_TARGET"
    col_portfolio_decile: str = "PORTFOLIO_UNITS_DECILE"

    # ── Team configuration ─────────────────────────────────────────────
    team_a_label: str = "TEAM_A"
    team_b_label: str = "TEAM_B"
    team_a_npi_col: str = "HCP_NPI"
    team_a_territory_col: str = "TERRITORY_ID"
    team_b_npi_col: str = "HCP_NPI"
    team_b_territory_col: str = "TERRITORY_ID"
    col_team_a_territory: str = "territory"
    col_team_b_territory: str = "GoldTerritory"

    # ── Output column names ────────────────────────────────────────────
    col_output_calls: str = "ALLOCATED_CALLS"
    col_output_territory: str = "Territory_ID"
    col_output_source: str = "ROW_SOURCE"
    col_output_npi_str: str = "NPI_STR"

    # ── Therapeutic-class Rx columns -> decile targets ─────────────────
    #    Each tuple: (source_rx_column, output_decile_column)
    #    Example: [("CLASS_A_TRX_L12M", "CLASS_A_DECILE")]
    il_rx_columns: list[tuple[str, str]] = field(default_factory=list)

    # ── Feature engineering ────────────────────────────────────────────
    target_horizon_weeks: int = 4
    scenario_range: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    lag_periods: list[int] = field(default_factory=lambda: [1, 2, 3, 4])
    rolling_windows: list[int] = field(default_factory=lambda: [4, 12])
    na_threshold: float = 0.10
    exclude_from_features: set[str] = field(default_factory=set)
    flag_columns_to_onehot: list[str] = field(default_factory=list)

    # ── Model training ─────────────────────────────────────────────────
    n_backtest_folds: int = 8
    backtest_gap_weeks: int = 8
    prob_threshold: float = 0.015
    retrain_on_full_data: bool = True

    prob_model_params: dict = field(default_factory=lambda: {
        "n_estimators": 100, "max_depth": 7, "random_state": 42,
    })
    depth_model_params: dict = field(default_factory=lambda: {
        "n_estimators": 200, "max_depth": 30, "min_samples_leaf": 4,
        "max_features": "sqrt", "n_jobs": -1, "random_state": 42,
    })
    look_model_params: dict = field(default_factory=lambda: {
        "n_estimators": 100, "max_depth": 5, "random_state": 42,
    })

    # ── Optimizer ──────────────────────────────────────────────────────
    team_a_budget_per_territory: int = 100
    team_b_target_per_territory: int = 100
    max_calls_nonpriority: int = 4
    priority_total_calls: list[int] = field(default_factory=lambda: [3, 4])
    require_mixed_at_max: bool = True
    beta_decile: float = 0.02
    big_m_penalty: int = 1000

    # ── Post-processing ────────────────────────────────────────────────
    decile_bins: int = 8
    lookalike_top_n: int = 10
    ev_scale_factor: float = 1.0

    # ── Output ─────────────────────────────────────────────────────────
    output_table: str = ""
    output_csv: str = ""

    def validate(self, dataframes: dict | None = None) -> list[str]:
        """Return a list of validation errors (empty if valid).

        Args:
            dataframes: If provided, skip path validation for keys that have
                        in-memory DataFrames (e.g. {"hcp_reference": df}).
        """
        dfs = dataframes or {}
        errors = []
        if not self.hcp_weekly_table and "hcp_weekly" not in dfs:
            errors.append("hcp_weekly_table is required (or pass dataframes={'hcp_weekly': df})")
        if not self.calls_table and "calls" not in dfs:
            errors.append("calls_table is required (or pass dataframes={'calls': df})")
        if not self.team_a_align_path and "team_a_align" not in dfs:
            errors.append("team_a_align_path is required (or pass dataframes={'team_a_align': df})")
        if not self.team_b_align_path and "team_b_align" not in dfs:
            errors.append("team_b_align_path is required (or pass dataframes={'team_b_align': df})")
        if not self.hcp_reference_path and "hcp_reference" not in dfs:
            errors.append("hcp_reference_path is required (or pass dataframes={'hcp_reference': df})")
        if self.target_horizon_weeks < 1:
            errors.append("target_horizon_weeks must be >= 1")
        if not self.scenario_range:
            errors.append("scenario_range must not be empty")
        if not self.output_table and not self.output_csv:
            errors.append("At least one of output_table or output_csv is required")
        return errors

    @classmethod
    def from_dict(cls, d: dict) -> "DetailOptimizationConfig":
        """Create config from a flat dictionary (e.g. the original CONFIG dict)."""
        # Map old CONFIG keys to new field names
        key_map = {
            "drug_name": "drug_name",
            "drug_portfolio": "drug_portfolio",
            "hcp_weekly_table": "hcp_weekly_table",
            "calls_table": "calls_table",
            "hcp_filter_col": "hcp_filter_col",
            "hcp_filter_val": "hcp_filter_val",
            "team_a_align_path": "team_a_align_path",
            "team_b_align_path": "team_b_align_path",
            "portfolio_decile_path": "portfolio_decile_path",
            "priority_target_path": "priority_target_path",
            "hcp_reference_path": "hcp_reference_path",
            "col_npi": "col_npi",
            "col_week": "col_week",
            "col_referrals": "col_referrals",
            "col_indication": "col_indication",
            "col_calls": "col_calls",
            "col_writer_flag": "col_writer_flag",
            "col_target_flag": "col_target_flag",
            "col_priority_flag": "col_priority_flag",
            "col_portfolio_decile": "col_portfolio_decile",
            "il_rx_columns": "il_rx_columns",
            "target_horizon_weeks": "target_horizon_weeks",
            "scenario_range": "scenario_range",
            "lag_periods": "lag_periods",
            "rolling_windows": "rolling_windows",
            "na_threshold": "na_threshold",
            "exclude_from_features": "exclude_from_features",
            "flag_columns_to_onehot": "flag_columns_to_onehot",
            "n_backtest_folds": "n_backtest_folds",
            "backtest_gap_weeks": "backtest_gap_weeks",
            "prob_threshold": "prob_threshold",
            "prob_model_params": "prob_model_params",
            "depth_model_params": "depth_model_params",
            "look_model_params": "look_model_params",
            "team_a_budget_per_territory": "team_a_budget_per_territory",
            "team_b_target_per_territory": "team_b_target_per_territory",
            "max_calls_nonpriority": "max_calls_nonpriority",
            "priority_total_calls": "priority_total_calls",
            "require_mixed_at_max": "require_mixed_at_max",
            "beta_decile": "beta_decile",
            "big_m_penalty": "big_m_penalty",
            "decile_bins": "decile_bins",
            "lookalike_top_n": "lookalike_top_n",
            "ev_scale_factor": "ev_scale_factor",
            "output_table": "output_table",
            "output_csv": "output_csv",
        }
        kwargs = {}
        for old_key, new_key in key_map.items():
            if old_key in d:
                kwargs[new_key] = d[old_key]
        return cls(**kwargs)
