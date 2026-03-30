"""Stage D: Feature engineering — transforms raw data into model-ready features."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ai2analytics.templates.detail_optimization.config import DetailOptimizationConfig
from ai2analytics.templates.detail_optimization.loader import LoadedData
from ai2analytics.utils import forward_sum, safe_fill


@dataclass
class FeatureSet:
    """Container for feature-engineered data."""
    df: pd.DataFrame = field(default_factory=pd.DataFrame)
    feature_cols: list[str] = field(default_factory=list)
    feat_prob: list[str] = field(default_factory=list)
    feat_depth: list[str] = field(default_factory=list)
    feat_look: list[str] = field(default_factory=list)


def engineer_features(
    cfg: DetailOptimizationConfig,
    data: LoadedData,
) -> FeatureSet:
    """Run all feature engineering stages. Returns a FeatureSet."""
    print("=" * 70)
    print("STAGE D: Feature engineering")
    print("=" * 70)

    rheum = data.hcp_weekly.copy()
    details = data.calls.copy()

    COL_NPI = cfg.col_npi
    COL_WEEK = cfg.col_week
    COL_REF = cfg.col_referrals
    COL_INDC = cfg.col_indication
    COL_CALL = cfg.col_calls
    HORIZON = cfg.target_horizon_weeks

    rheum[COL_WEEK] = pd.to_datetime(rheum[COL_WEEK])
    details[COL_WEEK] = pd.to_datetime(details[COL_WEEK])

    # D1. Condense to one row per NPI x WEEK
    grp_max_cols = [
        c for c in rheum.columns
        if c not in {COL_REF, COL_INDC, COL_WEEK, COL_NPI}
        and rheum[c].dtype != "O"
    ]

    condensed = (
        rheum.groupby([COL_NPI, COL_WEEK])
        .agg({COL_REF: "sum", **{c: "max" for c in grp_max_cols}})
        .reset_index()
    )

    if COL_INDC in rheum.columns:
        indc_cnt = (
            rheum[~rheum[COL_INDC].isin(["Unknown", "Others"])]
            .groupby([COL_NPI, COL_WEEK])[COL_INDC]
            .nunique()
            .rename("INDC_unique_cumesum")
        )
        condensed = condensed.merge(
            indc_cnt, on=[COL_NPI, COL_WEEK], how="left"
        ).fillna({"INDC_unique_cumesum": 0})

    df = condensed.copy()
    print(f"  Condensed: {len(df):,} rows, {df[COL_NPI].nunique():,} NPIs")

    # D2. First-write week and binary referral label
    df["ref"] = (df[COL_REF] > 0).astype(int)
    first_write = (
        df[df["ref"] == 1]
        .groupby(COL_NPI)[COL_WEEK]
        .min()
        .rename("first_write_week")
    )
    df = df.merge(first_write, on=COL_NPI, how="left")

    # D3. Drop *CALL*DUP columns
    drop_dup = [c for c in df.columns if "CALL" in c.upper() and "DUP" in c.upper()]
    df.drop(columns=drop_dup, inplace=True, errors="ignore")

    # D4. One-hot encode flag columns
    for flag in cfg.flag_columns_to_onehot:
        if flag in df.columns:
            dummies = pd.get_dummies(df[flag].astype(str), prefix=flag)
            df = pd.concat([df, dummies], axis=1)
            df.drop(columns=flag, inplace=True)

    # D5. Merge weekly detail calls
    det_npi_col = resolve_npi_col(details, COL_NPI)
    det_agg = (
        details.groupby([det_npi_col, COL_WEEK])[COL_CALL]
        .sum()
        .reset_index()
        .rename(columns={det_npi_col: COL_NPI, COL_CALL: "TS_CALLS"})
    )
    df = df.merge(det_agg, on=[COL_NPI, COL_WEEK], how="left").fillna({"TS_CALLS": 0})

    # D6. Rolling / lag features
    rolling_cols = ["TS_CALLS", COL_REF]

    for col in rolling_cols:
        grp = df.groupby(COL_NPI)[col]

        df[f"{col}_cume"] = grp.cumsum().shift(1).fillna(0)

        for w in cfg.rolling_windows:
            df[f"{col}_trail{w}"] = grp.transform(
                lambda s: s.shift(1).rolling(w, min_periods=1).sum()
            )

        df[f"{col}_ma4"] = grp.transform(
            lambda s: s.shift(1).rolling(4, min_periods=1).mean()
        )

        for lag in cfg.lag_periods:
            df[f"{col}_lag{lag}"] = grp.shift(lag).fillna(0)

    print(f"  Rolling/lag features built for: {rolling_cols}")

    # D7. Forward-looking call count — this IS a feature (the treatment variable).
    #     The model learns the call-response relationship from historical data.
    #     Scenario scoring then varies this column to predict outcomes at each
    #     hypothetical call level. Do NOT exclude from the feature list.
    df["TS_CALLS_next"] = (
        df.groupby(COL_NPI)["TS_CALLS"]
        .apply(lambda s: forward_sum(s, HORIZON))
        .reset_index(level=0, drop=True)
    )

    # D8. Target construction
    df["target_prob"] = (
        df.groupby(COL_NPI)["ref"]
        .apply(lambda s: forward_sum(s, HORIZON))
        .reset_index(level=0, drop=True)
        .gt(0)
        .astype(int)
    )

    df["target_cnt"] = (
        df.groupby(COL_NPI)[COL_REF]
        .apply(lambda s: forward_sum(s, HORIZON))
        .reset_index(level=0, drop=True)
        .fillna(0)
    )

    df["target_look"] = (
        (df["target_prob"] == 1)
        & (df["first_write_week"].isna() | (df["first_write_week"] > df[COL_WEEK]))
    ).astype(int)

    print(f"  Targets built: target_prob, target_cnt, target_look")

    # D9. Final numeric feature list (< na_threshold % missing)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    always_exclude = {
        COL_NPI, "ref", "target_prob", "target_cnt", "target_look",
        "TS_CALLS",
        # NOTE: TS_CALLS_next is deliberately NOT excluded — it is the
        # treatment variable that the model learns call-response curves from.
    }
    always_exclude.update(cfg.exclude_from_features)

    na_pct = df[num_cols].replace([np.inf, -np.inf], np.nan).isna().mean()
    feat_base = [
        c for c in num_cols
        if c not in always_exclude and na_pct[c] < cfg.na_threshold
    ]

    df = safe_fill(df, feat_base)

    result = FeatureSet(
        df=df,
        feature_cols=feat_base,
        feat_prob=feat_base.copy(),
        feat_depth=feat_base.copy(),
        feat_look=feat_base.copy(),
    )

    print(f"  Feature list: {len(feat_base)} predictors")
    print(f"  Final dataset: {len(df):,} rows, {df[COL_NPI].nunique():,} NPIs")
    print(f"  Done.\n")
    return result


def resolve_npi_col(df: pd.DataFrame, preferred: str) -> str:
    """Find the NPI column in the dataframe, trying preferred name first."""
    for candidate in [preferred, "NPI", "npi", "npi_number"]:
        if candidate in df.columns:
            return candidate
    raise ValueError(f"Could not find NPI column in {list(df.columns)}")
