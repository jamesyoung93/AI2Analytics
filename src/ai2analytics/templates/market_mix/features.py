"""Stage 2: Feature transformation — adstock, saturation, trend, and seasonality."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ai2analytics.templates.market_mix.config import AdstockConfig, MarketMixConfig
from ai2analytics.templates.market_mix.loader import MarketMixData


@dataclass
class TransformedFeatures:
    """Container for transformed feature matrices."""
    X: pd.DataFrame = field(default_factory=pd.DataFrame)
    y: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    feature_names: list[str] = field(default_factory=list)
    media_feature_names: list[str] = field(default_factory=list)
    control_feature_names: list[str] = field(default_factory=list)
    structural_feature_names: list[str] = field(default_factory=list)
    adstock_applied: dict[str, float] = field(default_factory=dict)


def apply_adstock(series: pd.Series, decay_rate: float) -> pd.Series:
    """Apply geometric adstock transformation.

    x_t' = x_t + decay * x_{t-1}'
    """
    result = np.zeros(len(series))
    values = series.values.astype(float)
    result[0] = values[0]
    for t in range(1, len(values)):
        result[t] = values[t] + decay_rate * result[t - 1]
    return pd.Series(result, index=series.index, name=series.name)


def apply_saturation(
    series: pd.Series,
    saturation_type: str = "hill",
    half_max: float = 1.0,
    steepness: float = 1.0,
) -> pd.Series:
    """Apply saturation transformation.

    Supported types:
        hill: x^s / (x^s + h^s)
        log:  log(1 + x)
        none: identity (no transformation)
    """
    values = series.values.astype(float)

    if saturation_type == "hill":
        # Hill function: x^s / (x^s + h^s)
        numerator = np.power(np.abs(values), steepness)
        denominator = numerator + np.power(half_max, steepness)
        result = np.where(denominator > 0, numerator / denominator, 0.0)
    elif saturation_type == "log":
        result = np.log1p(np.abs(values))
    elif saturation_type == "none":
        result = values.copy()
    else:
        raise ValueError(
            f"Unknown saturation_type '{saturation_type}'. "
            f"Use 'hill', 'log', or 'none'."
        )

    return pd.Series(result, index=series.index, name=series.name)


def build_trend_seasonality(
    n_periods: int,
    include_trend: bool = True,
    include_seasonality: bool = True,
    period: int = 52,
) -> pd.DataFrame:
    """Build structural features: linear trend and Fourier seasonality.

    Returns a DataFrame with trend and/or sin/cos seasonality columns.
    """
    features = {}

    if include_trend:
        # Linear trend normalized to [0, 1]
        features["trend"] = np.linspace(0, 1, n_periods)

    if include_seasonality and period >= 2:
        # Fourier terms: sin and cos at the fundamental frequency
        t = np.arange(n_periods, dtype=float)
        features[f"sin_{period}"] = np.sin(2 * np.pi * t / period)
        features[f"cos_{period}"] = np.cos(2 * np.pi * t / period)

    return pd.DataFrame(features)


def _get_adstock_config(cfg: MarketMixConfig, channel_name: str) -> AdstockConfig:
    """Look up per-channel adstock config, or build one from defaults."""
    for ac in cfg.adstock_configs:
        if ac.channel_name == channel_name:
            return ac
    return AdstockConfig(
        channel_name=channel_name,
        decay_rate=cfg.default_decay_rate,
        saturation_type=cfg.default_saturation,
    )


def transform_features(
    cfg: MarketMixConfig,
    data: MarketMixData,
) -> TransformedFeatures:
    """Transform raw data into model-ready features.

    Steps:
        1. Apply adstock + saturation to each media column
        2. Pass through control columns
        3. Build trend and seasonality features
        4. Assemble final feature matrix X and response vector y
    """
    print("=" * 70)
    print("STAGE 2: Feature transformation")
    print("=" * 70)

    ts = data.time_series
    result = TransformedFeatures()
    feature_frames = []

    # -- Media features: adstock + saturation ----------------------------
    for col in data.media_cols:
        ac = _get_adstock_config(cfg, col)

        # Apply adstock
        adstocked = apply_adstock(ts[col], ac.decay_rate)
        result.adstock_applied[col] = ac.decay_rate

        # Apply saturation
        transformed = apply_saturation(
            adstocked,
            saturation_type=ac.saturation_type,
            half_max=ac.saturation_half_max,
            steepness=ac.saturation_steepness,
        )

        feature_name = f"{col}_transformed"
        transformed.name = feature_name
        feature_frames.append(transformed)
        result.media_feature_names.append(feature_name)
        print(f"  Media: {col} -> decay={ac.decay_rate}, sat={ac.saturation_type}")

    # -- Control features: pass through ----------------------------------
    for col in data.control_cols:
        control_series = ts[col].astype(float).reset_index(drop=True)
        control_series.name = col
        feature_frames.append(control_series)
        result.control_feature_names.append(col)
        print(f"  Control: {col}")

    # -- Structural features: trend + seasonality ------------------------
    structural = build_trend_seasonality(
        n_periods=data.n_periods,
        include_trend=cfg.include_trend,
        include_seasonality=cfg.include_seasonality,
        period=cfg.seasonality_period,
    )
    for col in structural.columns:
        result.structural_feature_names.append(col)
        print(f"  Structural: {col}")

    # -- Assemble feature matrix -----------------------------------------
    if feature_frames:
        media_control_df = pd.concat(
            [s.reset_index(drop=True) for s in feature_frames], axis=1
        )
    else:
        media_control_df = pd.DataFrame()

    structural = structural.reset_index(drop=True)

    if not media_control_df.empty and not structural.empty:
        result.X = pd.concat([media_control_df, structural], axis=1)
    elif not media_control_df.empty:
        result.X = media_control_df
    else:
        result.X = structural

    result.feature_names = list(result.X.columns)
    result.y = ts[data.response_col].astype(float).reset_index(drop=True)

    print(f"\n  Features: {len(result.feature_names)}")
    print(f"    Media:      {len(result.media_feature_names)}")
    print(f"    Control:    {len(result.control_feature_names)}")
    print(f"    Structural: {len(result.structural_feature_names)}")
    print(f"  Periods:  {len(result.y)}")
    print("  Done.\n")
    return result
