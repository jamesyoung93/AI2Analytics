from __future__ import annotations

import pandas as pd

from ai2analytics.templates.generalized.cadence import add_periods, resolve_existing_period
from ai2analytics.templates.generalized.config import GeneralizedDetailOptimizationConfig
from ai2analytics.templates.generalized.loader import _load_capacity, _load_priority_targets
from ai2analytics.templates.generalized.reporting import GeneralizedReporter


def _base_config(**kwargs) -> GeneralizedDetailOptimizationConfig:
    values = {
        "hcp_weekly_table": "catalog.schema.hcp_weekly",
        "calls_table": "catalog.schema.calls",
        "team_a_align_path": "team_a.csv",
        "team_b_align_path": "team_b.csv",
        "hcp_reference_path": "hcp_reference.csv",
        "output_csv": "out.csv",
    }
    values.update(kwargs)
    return GeneralizedDetailOptimizationConfig(**values)


def test_generalized_defaults_explain_assumptions():
    cfg = _base_config()

    messages = "\n".join(cfg.assumption_messages())

    assert "No priority target input supplied" in messages
    assert "No Team A capacity file supplied" in messages
    assert "No Team B capacity file supplied" in messages
    assert "apply_hcp_filter=False" in messages
    assert cfg.validate() == []


def test_cadence_normalization_uses_period_fields():
    cfg = _base_config(
        planning_cadence="monthly",
        planning_horizon_periods=2,
        backtest_gap_periods=3,
    )

    runtime_cfg = cfg.normalize_for_run()

    assert runtime_cfg.planning_cadence == "monthly"
    assert runtime_cfg.target_horizon_weeks == 2
    assert runtime_cfg.backtest_gap_weeks == 3
    assert runtime_cfg.effective_horizon_periods() == 2


def test_cadence_helpers_resolve_missing_periods():
    periods = pd.to_datetime(["2024-01-01", "2024-03-01"])

    before, used_before_fallback = resolve_existing_period(
        periods,
        pd.Timestamp("2024-02-01"),
        prefer="on_or_before",
    )
    after, used_after_fallback = resolve_existing_period(
        periods,
        pd.Timestamp("2024-02-01"),
        prefer="on_or_after",
    )

    assert add_periods(pd.Timestamp("2024-01-15"), 1, "monthly").month == 2
    assert before == pd.Timestamp("2024-01-01")
    assert after == pd.Timestamp("2024-03-01")
    assert used_before_fallback is True
    assert used_after_fallback is True


def test_priority_npi_list_without_flag_becomes_binary_priority():
    cfg = _base_config()
    reporter = GeneralizedReporter(verbose=False, warn_on_defaults=False)
    raw_priority = pd.DataFrame({"NPI": [101, 202]})

    priority = _load_priority_targets(
        cfg,
        {"priority_targets": raw_priority},
        reporter,
    )

    assert priority is not None
    assert priority[cfg.col_npi].tolist() == [101, 202]
    assert priority["PRIORITY_TARGET_FLAG"].tolist() == [1, 1]
    assert "treating every row" in reporter.warnings[0]


def test_capacity_aggregation_sums_duplicate_territories():
    reporter = GeneralizedReporter(verbose=False, warn_on_defaults=False)
    raw_capacity = pd.DataFrame({
        "TERRITORY_ID": ["A1", "A1", "B2"],
        "CAPACITY": [3, 4, 5],
    })

    capacity = _load_capacity(
        raw=raw_capacity,
        path=None,
        territory_col="TERRITORY_ID",
        value_col="CAPACITY",
        output_territory_col="territory",
        aggregation="sum",
        label="Team A",
        reporter=reporter,
    )

    observed = dict(zip(capacity["territory"], capacity["_capacity"]))
    assert observed == {"A1": 7, "B2": 5}
