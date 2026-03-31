"""Stage 3: Model fitting — Ridge/OLS/Lasso regression and contribution decomposition."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ai2analytics.templates.market_mix.config import MarketMixConfig
from ai2analytics.templates.market_mix.features import TransformedFeatures
from ai2analytics.templates.market_mix.loader import MarketMixData


@dataclass
class MarketMixResult:
    """Container for model fitting results."""
    model: Any = None
    coefficients: dict[str, float] = field(default_factory=dict)
    intercept: float = 0.0
    r_squared: float = 0.0
    adjusted_r_squared: float = 0.0
    mape: float = 0.0
    contributions: pd.DataFrame = field(default_factory=pd.DataFrame)
    channel_roi: dict[str, float] = field(default_factory=dict)
    total_contribution_pct: dict[str, float] = field(default_factory=dict)


def fit_model(
    cfg: MarketMixConfig,
    features: TransformedFeatures,
    data: MarketMixData,
) -> MarketMixResult:
    """Fit the regression model and decompose contributions.

    Args:
        cfg: Pipeline configuration.
        features: Transformed feature matrix and response vector.
        data: Raw loaded data (for ROI spend calculation).

    Returns:
        MarketMixResult with fitted model, coefficients, metrics,
        per-period contributions, channel ROI, and contribution percentages.
    """
    from sklearn.linear_model import Lasso, LinearRegression, Ridge

    print("=" * 70)
    print("STAGE 3: Model fitting")
    print("=" * 70)

    X = features.X.values
    y = features.y.values
    n, p = X.shape

    result = MarketMixResult()

    # -- Fit model -------------------------------------------------------
    if cfg.model_type == "ridge":
        model = Ridge(
            alpha=cfg.alpha,
            fit_intercept=cfg.fit_intercept,
            positive=cfg.positive_coefficients,
        )
        print(f"  Model: Ridge (alpha={cfg.alpha}, positive={cfg.positive_coefficients})")
    elif cfg.model_type == "lasso":
        model = Lasso(
            alpha=cfg.alpha,
            fit_intercept=cfg.fit_intercept,
            positive=cfg.positive_coefficients,
            max_iter=10000,
        )
        print(f"  Model: Lasso (alpha={cfg.alpha}, positive={cfg.positive_coefficients})")
    else:
        model = LinearRegression(
            fit_intercept=cfg.fit_intercept,
            positive=cfg.positive_coefficients,
        )
        print(f"  Model: OLS (positive={cfg.positive_coefficients})")

    model.fit(X, y)
    result.model = model

    # -- Extract coefficients --------------------------------------------
    coefs = model.coef_
    intercept = model.intercept_ if cfg.fit_intercept else 0.0
    result.intercept = float(intercept)

    for i, fname in enumerate(features.feature_names):
        result.coefficients[fname] = float(coefs[i])

    print(f"  Intercept: {result.intercept:.4f}")
    for fname, cval in result.coefficients.items():
        print(f"  {fname}: {cval:.6f}")

    # -- Model diagnostics -----------------------------------------------
    y_pred = model.predict(X)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    result.r_squared = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    result.adjusted_r_squared = float(
        1.0 - (1.0 - result.r_squared) * (n - 1) / (n - p - 1)
    ) if n > p + 1 else result.r_squared

    # MAPE
    mask = y != 0
    if mask.any():
        result.mape = float(np.mean(np.abs((y[mask] - y_pred[mask]) / y[mask])) * 100)
    else:
        result.mape = 0.0

    print(f"\n  R-squared:          {result.r_squared:.4f}")
    print(f"  Adjusted R-squared: {result.adjusted_r_squared:.4f}")
    print(f"  MAPE:               {result.mape:.2f}%")

    # -- Contribution decomposition --------------------------------------
    # Contribution at time t for feature = coefficient * transformed_value_t
    contributions = pd.DataFrame(index=range(n))
    contributions["base"] = intercept

    for i, fname in enumerate(features.feature_names):
        contributions[fname] = coefs[i] * features.X[fname].values

    contributions["predicted"] = y_pred
    contributions["actual"] = y
    result.contributions = contributions

    # -- Total contribution percentage -----------------------------------
    total_response = float(np.sum(y))
    for fname in features.feature_names:
        total_contrib = float(contributions[fname].sum())
        result.total_contribution_pct[fname] = (
            (total_contrib / total_response * 100) if total_response != 0 else 0.0
        )

    base_pct = (float(intercept * n) / total_response * 100) if total_response != 0 else 0.0
    result.total_contribution_pct["base"] = base_pct

    print("\n  Contribution %:")
    print(f"    base: {base_pct:.2f}%")
    for fname in features.feature_names:
        print(f"    {fname}: {result.total_contribution_pct[fname]:.2f}%")

    # -- Channel ROI -----------------------------------------------------
    # ROI = sum(contribution) / sum(raw_spend)
    ts = data.time_series
    for media_col in data.media_cols:
        feature_name = f"{media_col}_transformed"
        if feature_name in contributions.columns and media_col in ts.columns:
            total_contrib = float(contributions[feature_name].sum())
            total_spend = float(ts[media_col].sum())
            result.channel_roi[media_col] = (
                total_contrib / total_spend if total_spend != 0 else 0.0
            )
            print(f"  ROI {media_col}: {result.channel_roi[media_col]:.4f}")

    print("  Done.\n")
    return result
