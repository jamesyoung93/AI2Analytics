"""Stage F: Scenario scoring — vectorized EV computation at each call level."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ai2analytics.templates.detail_optimization.config import DetailOptimizationConfig
from ai2analytics.templates.detail_optimization.features import FeatureSet
from ai2analytics.templates.detail_optimization.models import TrainedModels


def score_scenarios(
    cfg: DetailOptimizationConfig,
    features: FeatureSet,
    models: TrainedModels,
) -> pd.DataFrame:
    """Score all NPIs at each call level using vectorized batch prediction.

    Returns a DataFrame with columns: [col_npi, scenario, pred_prob, pred_depth, EV]
    plus all feature columns needed downstream.
    """
    print("=" * 70)
    print("STAGE F: Scenario scoring")
    print("=" * 70)

    col_week = cfg.col_week
    col_npi = cfg.col_npi
    horizon = cfg.target_horizon_weeks
    max_week = features.df[col_week].max()

    # F1. Select the planning snapshot
    planning_date = max_week - pd.Timedelta(weeks=horizon)
    df_plan = features.df[features.df[col_week] == planning_date].copy()
    print(f"  Planning date: {planning_date}")
    print(f"  Planning NPIs: {df_plan[col_npi].nunique():,}")

    if df_plan.empty:
        raise ValueError(
            f"No data at planning_date={planning_date}. "
            f"Check that data covers at least {horizon} weeks before max_week={max_week}."
        )

    # F2. Build scenario matrix — one copy of df_plan per call level (vectorized)
    scenario_dfs = []
    for call_level in cfg.scenario_range:
        sc = df_plan.copy()
        sc["TS_CALLS_next"] = call_level
        sc["scenario"] = call_level
        scenario_dfs.append(sc)

    sc_df = pd.concat(scenario_dfs, ignore_index=True)

    # F3. Vectorized predictions
    is_look_mask = (
        sc_df["first_write_week"].isna()
        | (sc_df["first_write_week"] > planning_date)
    )

    sc_df["pred_prob"] = 0.0
    sc_df["pred_depth"] = 0.0

    # Probability predictions — general model for writers, look-alike for non-writers
    general_mask = ~is_look_mask
    if models.prob_model is not None and general_mask.any():
        sc_df.loc[general_mask, "pred_prob"] = models.prob_model.predict_proba(
            sc_df.loc[general_mask, features.feat_prob]
        )[:, 1]

    if models.look_model is not None and is_look_mask.any():
        sc_df.loc[is_look_mask, "pred_prob"] = models.look_model.predict_proba(
            sc_df.loc[is_look_mask, features.feat_look]
        )[:, 1]

    # Depth predictions
    if models.depth_model is not None:
        sc_df["pred_depth"] = models.depth_model.predict(
            sc_df[features.feat_depth]
        )

    sc_df["EV"] = sc_df["pred_prob"] * sc_df["pred_depth"]

    print(
        f"  Scenarios built: {len(sc_df):,} rows "
        f"({sc_df[col_npi].nunique():,} NPIs x {len(cfg.scenario_range)} levels)"
    )

    # F4. Impute missing depths via probability-rank quantile matching
    mask_na = sc_df["pred_depth"].isna()
    if mask_na.any():
        sc_df["_prob_pct"] = sc_df["pred_prob"].rank(method="average", pct=True)
        depth_vals = sc_df.loc[~mask_na, "pred_depth"].values
        sc_df.loc[mask_na, "pred_depth"] = np.quantile(
            depth_vals, q=sc_df.loc[mask_na, "_prob_pct"].values
        )
        sc_df["EV"] = sc_df["pred_prob"] * sc_df["pred_depth"]
        sc_df.drop(columns="_prob_pct", inplace=True)
        print(f"  Imputed {mask_na.sum()} missing depth values")

    print("  Done.\n")
    return sc_df
