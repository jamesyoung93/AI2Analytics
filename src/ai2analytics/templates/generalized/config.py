"""Configuration for the generalized detail optimization template."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, replace
from typing import Any

from ai2analytics.templates.detail_optimization.config import DetailOptimizationConfig
from ai2analytics.templates.generalized.cadence import normalize_cadence


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    return False


def _has_df(dataframes: dict[str, Any] | None, key: str) -> bool:
    return bool(dataframes) and key in dataframes and dataframes[key] is not None


@dataclass
class GeneralizedDetailOptimizationConfig(DetailOptimizationConfig):
    """Config for a more portable version of detail optimization.

    This keeps the existing two-field-force alignment structure, but makes
    cadence, priority input, and capacity assumptions explicit and verbose.
    """

    # Identity. The default is intentionally printable, so a run without a
    # supplied brand name makes the assumption obvious.
    drug_name: str = "PRIMARY_BRAND"
    drug_portfolio: str | None = None

    # Optional inputs. Defaults are None so the AI setup flow does not ask for
    # them as required fields.
    portfolio_decile_path: str | None = None
    priority_target_path: str | None = None

    # Planning cadence. Horizon and lag/rolling windows are row-period based:
    # weekly data means periods are weeks, monthly data means periods are
    # months, and quarterly data means periods are quarters.
    planning_cadence: str = "weekly"
    planning_horizon_periods: int | None = None
    backtest_gap_periods: int | None = None

    # Generalized runs default to no row filter unless the user opts in.
    apply_hcp_filter: bool = False

    # Optional per-territory or per-rep capacity files. If absent, the scalar
    # inherited team_a_budget_per_territory and team_b_target_per_territory
    # fields are used for every territory.
    team_a_capacity_path: str | None = None
    team_b_capacity_path: str | None = None
    team_a_capacity_territory_col: str = "TERRITORY_ID"
    team_a_capacity_value_col: str = "CAPACITY"
    team_b_capacity_territory_col: str = "TERRITORY_ID"
    team_b_capacity_value_col: str = "CAPACITY"
    capacity_aggregation: str = "sum"

    # User-facing trace behavior.
    verbose: bool = True
    warn_on_defaults: bool = True

    def normalize_for_run(self) -> "GeneralizedDetailOptimizationConfig":
        """Return a runtime config with cadence fields resolved."""
        cfg = replace(self)
        cfg.planning_cadence = normalize_cadence(cfg.planning_cadence)

        if cfg.planning_horizon_periods is not None:
            cfg.target_horizon_weeks = cfg.planning_horizon_periods
        if cfg.backtest_gap_periods is not None:
            cfg.backtest_gap_weeks = cfg.backtest_gap_periods

        return cfg

    def effective_horizon_periods(self) -> int:
        return self.planning_horizon_periods or self.target_horizon_weeks

    def effective_backtest_gap_periods(self) -> int:
        return self.backtest_gap_periods or self.backtest_gap_weeks

    def assumption_messages(self, dataframes: dict[str, Any] | None = None) -> list[str]:
        """Explain default behavior that may materially affect a run."""
        messages = []

        if self.drug_name == "PRIMARY_BRAND":
            messages.append(
                "drug_name was not supplied; outputs and logs will use PRIMARY_BRAND."
            )
        if _is_missing(self.drug_portfolio):
            messages.append(
                "drug_portfolio was not supplied; portfolio-specific labeling will be omitted."
            )

        if self.planning_horizon_periods is None:
            messages.append(
                "planning_horizon_periods was not supplied; using "
                f"target_horizon_weeks={self.target_horizon_weeks} as the number of "
                f"{self.planning_cadence} planning periods."
            )

        if self.planning_cadence != "weekly":
            messages.append(
                f"planning_cadence={self.planning_cadence!r}; horizon, lag, and rolling "
                "settings are interpreted as row periods, while model backtests and "
                "planning snapshots use calendar offsets for that cadence."
            )

        if not self.apply_hcp_filter:
            messages.append(
                "apply_hcp_filter=False; all rows from hcp_weekly_table will be loaded. "
                "Set apply_hcp_filter=True with hcp_filter_col/hcp_filter_val to filter."
            )
        elif self.hcp_filter_col == "TARGET_FLAG" and self.hcp_filter_val == "Y":
            messages.append(
                "apply_hcp_filter=True with the default TARGET_FLAG LIKE 'Y' filter."
            )

        if _is_missing(self.priority_target_path) and not _has_df(dataframes, "priority_targets"):
            messages.append(
                "No priority target input supplied; all HCPs will use non-priority call rules."
            )
        else:
            messages.append(
                "Priority input is treated as a single binary flag; P1/P2 tiers are not modeled."
            )

        if _is_missing(self.portfolio_decile_path) and not _has_df(dataframes, "portfolio_decile"):
            messages.append(
                "No portfolio decile input supplied; portfolio decile values default to zero "
                "and the optimizer will not apply a portfolio-decile boost."
            )

        has_team_a_capacity = (
            not _is_missing(self.team_a_capacity_path) or _has_df(dataframes, "team_a_capacity")
        )
        has_team_b_capacity = (
            not _is_missing(self.team_b_capacity_path) or _has_df(dataframes, "team_b_capacity")
        )
        if not has_team_a_capacity:
            messages.append(
                "No Team A capacity file supplied; using "
                f"team_a_budget_per_territory={self.team_a_budget_per_territory} for every "
                "Team A territory."
            )
        if not has_team_b_capacity:
            messages.append(
                "No Team B capacity file supplied; using "
                f"team_b_target_per_territory={self.team_b_target_per_territory} for every "
                "Team B territory."
            )

        return messages

    def validate(self, dataframes: dict | None = None) -> list[str]:
        errors = super().validate(dataframes=dataframes)

        try:
            normalize_cadence(self.planning_cadence)
        except ValueError as exc:
            errors.append(str(exc))

        if self.effective_horizon_periods() < 1:
            errors.append("planning horizon must be >= 1 period")
        if self.effective_backtest_gap_periods() < 0:
            errors.append("backtest gap must be >= 0 periods")
        if self.capacity_aggregation not in {"sum", "max", "first"}:
            errors.append("capacity_aggregation must be one of: sum, max, first")

        return errors

    @classmethod
    def from_dict(cls, d: dict) -> "GeneralizedDetailOptimizationConfig":
        """Create config from a dict, accepting inherited and generalized fields."""
        valid_fields = {f.name for f in dataclasses.fields(cls)}
        kwargs = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**kwargs)
