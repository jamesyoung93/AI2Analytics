"""Cadence-aware scenario scoring for generalized detail optimization."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ai2analytics.templates.detail_optimization.features import FeatureSet
from ai2analytics.templates.generalized.cadence import add_periods, resolve_existing_period
from ai2analytics.templates.generalized.config import GeneralizedDetailOptimizationConfig
from ai2analytics.templates.generalized.models import TrainedModels
from ai2analytics.templates.generalized.reporting import GeneralizedReporter


def score_scenarios(
    cfg: GeneralizedDetailOptimizationConfig,
    features: FeatureSet,
    models: TrainedModels,
    reporter: GeneralizedReporter | None = None,
) -> pd.DataFrame:
    """Score all NPIs at each call level using cadence-aware snapshot selection."""
    reporter = reporter or GeneralizedReporter(cfg.verbose, cfg.warn_on_defaults)

    print("=" * 70)
    print("STAGE F: Scenario scoring (generalized)")
    print("=" * 70)

    col_week = cfg.col_week
    col_npi = cfg.col_npi
    horizon = cfg.effective_horizon_periods()
    max_period = pd.Timestamp(features.df[col_week].max())

    target_planning_period = add_periods(max_period, -horizon, cfg.planning_cadence)
    observed_periods = pd.to_datetime(features.df[col_week].unique())
    planning_period, fallback = resolve_existing_period(
        observed_periods,
        target_planning_period,
        prefer="on_or_before",
    )

    if planning_period is None:
        raise ValueError(
            f"No data at or before planning target {target_planning_period}. "
            f"Check that data covers at least {horizon} {cfg.planning_cadence} periods."
        )

    if fallback:
        reporter.warn(
            f"Exact planning period {target_planning_period.date()} was not present; "
            f"using latest observed period on or before it: {planning_period.date()}."
        )

    df_plan = features.df[features.df[col_week] == planning_period].copy()
    print(f"  Planning period: {planning_period}")
    print(f"  Planning NPIs:   {df_plan[col_npi].nunique():,}")

    if df_plan.empty:
        raise ValueError(f"No data found at resolved planning_period={planning_period}.")

    scenario_dfs = []
    for call_level in cfg.scenario_range:
        sc = df_plan.copy()
        sc["TS_CALLS_next"] = call_level
        sc["scenario"] = call_level
        scenario_dfs.append(sc)

    sc_df = pd.concat(scenario_dfs, ignore_index=True)

    is_look_mask = (
        sc_df["first_write_week"].isna()
        | (sc_df["first_write_week"] > planning_period)
    )

    sc_df["pred_prob"] = 0.0
    sc_df["pred_depth"] = 0.0

    general_mask = ~is_look_mask
    if models.prob_model is not None and general_mask.any():
        sc_df.loc[general_mask, "pred_prob"] = _positive_class_probability(
            models.prob_model,
            sc_df.loc[general_mask, features.feat_prob],
        )
    elif general_mask.any():
        reporter.warn(
            "No probability model is available for writer/general HCPs; their "
            "predicted probability defaults to 0."
        )

    if models.look_model is not None and is_look_mask.any():
        sc_df.loc[is_look_mask, "pred_prob"] = _positive_class_probability(
            models.look_model,
            sc_df.loc[is_look_mask, features.feat_look],
        )
    elif is_look_mask.any():
        reporter.warn(
            "No look-alike model is available for never-writers; their predicted "
            "probability defaults to 0."
        )

    if models.depth_model is not None:
        sc_df["pred_depth"] = models.depth_model.predict(sc_df[features.feat_depth])
    else:
        reporter.warn("No depth model is available; predicted depth defaults to 0.")

    sc_df["EV"] = sc_df["pred_prob"] * sc_df["pred_depth"]

    print(
        f"  Scenarios built: {len(sc_df):,} rows "
        f"({sc_df[col_npi].nunique():,} NPIs x {len(cfg.scenario_range)} levels)"
    )

    mask_na = sc_df["pred_depth"].isna()
    if mask_na.any():
        sc_df["_prob_pct"] = sc_df["pred_prob"].rank(method="average", pct=True)
        depth_vals = sc_df.loc[~mask_na, "pred_depth"].values
        if len(depth_vals) == 0:
            reporter.warn("All predicted depth values are missing; filling depth with zero.")
            sc_df.loc[mask_na, "pred_depth"] = 0
        else:
            sc_df.loc[mask_na, "pred_depth"] = np.quantile(
                depth_vals,
                q=sc_df.loc[mask_na, "_prob_pct"].values,
            )
        sc_df["EV"] = sc_df["pred_prob"] * sc_df["pred_depth"]
        sc_df.drop(columns="_prob_pct", inplace=True)
        print(f"  Imputed {mask_na.sum()} missing depth values")

    print("  Done.\n")
    return sc_df


def _positive_class_probability(model, x: pd.DataFrame) -> np.ndarray:
    proba = model.predict_proba(x)
    classes = list(getattr(model, "classes_", []))
    if 1 in classes:
        return proba[:, classes.index(1)]
    if classes == [0]:
        return np.zeros(len(x))
    if classes == [1]:
        return np.ones(len(x))
    return proba[:, -1]
