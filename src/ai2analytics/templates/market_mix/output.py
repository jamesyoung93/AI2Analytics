"""Stage 4: Output — summary tables, response curves, writing, and plotting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ai2analytics.templates.market_mix.config import MarketMixConfig
from ai2analytics.templates.market_mix.features import (
    TransformedFeatures,
    apply_adstock,
    apply_saturation,
    _get_adstock_config,
)
from ai2analytics.templates.market_mix.loader import MarketMixData
from ai2analytics.templates.market_mix.model import MarketMixResult


@dataclass
class MarketMixOutput:
    """Final pipeline output container."""
    contributions: pd.DataFrame = field(default_factory=pd.DataFrame)
    channel_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    model_diagnostics: pd.DataFrame = field(default_factory=pd.DataFrame)
    response_curves: pd.DataFrame = field(default_factory=pd.DataFrame)


def build_output(
    cfg: MarketMixConfig,
    data: MarketMixData,
    features: TransformedFeatures,
    result: MarketMixResult,
) -> MarketMixOutput:
    """Build summary tables and response curves from model results.

    Args:
        cfg: Pipeline configuration.
        data: Loaded time series data.
        features: Transformed feature matrix.
        result: Fitted model results.

    Returns:
        MarketMixOutput with contributions, channel summary, diagnostics,
        and response curves.
    """
    print("=" * 70)
    print("STAGE 4: Building output")
    print("=" * 70)

    output = MarketMixOutput()

    # -- Contributions table with date -----------------------------------
    contributions = result.contributions.copy()
    contributions[cfg.col_date] = data.time_series[cfg.col_date].values
    output.contributions = contributions
    print(f"  Contributions: {len(contributions):,} rows x {len(contributions.columns)} cols")

    # -- Channel summary -------------------------------------------------
    summary_rows = []
    for media_col in data.media_cols:
        feature_name = f"{media_col}_transformed"
        row = {
            "channel": media_col,
            "total_spend": float(data.time_series[media_col].sum()),
            "total_contribution": float(
                contributions[feature_name].sum()
            ) if feature_name in contributions.columns else 0.0,
            "contribution_pct": result.total_contribution_pct.get(feature_name, 0.0),
            "roi": result.channel_roi.get(media_col, 0.0),
            "coefficient": result.coefficients.get(feature_name, 0.0),
            "decay_rate": features.adstock_applied.get(media_col, cfg.default_decay_rate),
        }
        summary_rows.append(row)

    # Add base and structural rows
    summary_rows.append({
        "channel": "base",
        "total_spend": 0.0,
        "total_contribution": float(result.intercept * data.n_periods),
        "contribution_pct": result.total_contribution_pct.get("base", 0.0),
        "roi": 0.0,
        "coefficient": result.intercept,
        "decay_rate": 0.0,
    })
    for ctrl_col in data.control_cols:
        if ctrl_col in contributions.columns:
            summary_rows.append({
                "channel": ctrl_col,
                "total_spend": 0.0,
                "total_contribution": float(contributions[ctrl_col].sum()),
                "contribution_pct": result.total_contribution_pct.get(ctrl_col, 0.0),
                "roi": 0.0,
                "coefficient": result.coefficients.get(ctrl_col, 0.0),
                "decay_rate": 0.0,
            })

    output.channel_summary = pd.DataFrame(summary_rows)
    print(f"  Channel summary: {len(output.channel_summary)} channels")

    # -- Model diagnostics -----------------------------------------------
    output.model_diagnostics = pd.DataFrame([{
        "model_type": cfg.model_type,
        "alpha": cfg.alpha,
        "r_squared": result.r_squared,
        "adjusted_r_squared": result.adjusted_r_squared,
        "mape": result.mape,
        "n_periods": data.n_periods,
        "n_features": len(features.feature_names),
        "intercept": result.intercept,
    }])
    print(f"  R-squared: {result.r_squared:.4f}, MAPE: {result.mape:.2f}%")

    # -- Response curves -------------------------------------------------
    output.response_curves = generate_response_curves(
        cfg, features, result, n_points=50,
    )
    print(f"  Response curves: {len(output.response_curves)} points")

    print("  Done.\n")
    return output


def generate_response_curves(
    cfg: MarketMixConfig,
    features: TransformedFeatures,
    result: MarketMixResult,
    n_points: int = 50,
) -> pd.DataFrame:
    """Generate response curves by varying spend from 0 to 2x max.

    For each media channel, create a range of spend values, apply the same
    adstock and saturation transforms, then multiply by the coefficient
    to get the marginal response.

    Args:
        cfg: Pipeline configuration.
        features: Transformed features (for adstock config lookup).
        result: Fitted model results (for coefficients).
        n_points: Number of points along the spend axis.

    Returns:
        DataFrame with columns: channel, spend_level, response.
    """
    rows = []

    for media_col in cfg.media_columns:
        feature_name = f"{media_col}_transformed"
        coef = result.coefficients.get(feature_name, 0.0)
        ac = _get_adstock_config(cfg, media_col)

        # Get the max raw spend for this channel
        max_spend = float(features.X[feature_name].max()) if feature_name in features.X.columns else 1.0
        # Use the raw data max if available via adstock info
        raw_max = max_spend  # Approximation: use transformed max as proxy

        spend_range = np.linspace(0, 2.0 * raw_max, n_points)

        for spend_val in spend_range:
            # Create a single-period series and apply saturation
            # (adstock is cumulative so we approximate with a single period)
            sat_val = apply_saturation(
                pd.Series([spend_val]),
                saturation_type=ac.saturation_type,
                half_max=ac.saturation_half_max,
                steepness=ac.saturation_steepness,
            ).iloc[0]
            response = coef * sat_val
            rows.append({
                "channel": media_col,
                "spend_level": float(spend_val),
                "response": float(response),
            })

    return pd.DataFrame(rows)


def write_output(
    cfg: MarketMixConfig,
    output: MarketMixOutput,
    spark: Any = None,
) -> None:
    """Write output to CSV and/or Spark table."""
    print("=" * 70)
    print("STAGE 5: Writing output")
    print("=" * 70)

    summary = output.channel_summary
    print(f"\n  Channels: {len(summary)}")
    for _, row in summary.iterrows():
        if row["channel"] != "base":
            print(
                f"    {row['channel']}: "
                f"spend={row['total_spend']:,.0f}, "
                f"contrib={row['total_contribution']:,.2f}, "
                f"ROI={row['roi']:.4f}"
            )

    # Write CSV
    if cfg.output_csv:
        import os
        os.makedirs(os.path.dirname(cfg.output_csv) or ".", exist_ok=True)
        output.contributions.to_csv(cfg.output_csv, index=False)
        print(f"\n  Wrote CSV: {cfg.output_csv} ({len(output.contributions):,} rows)")

    # Write Spark table
    if cfg.output_table and spark is not None:
        try:
            spark_df = spark.createDataFrame(output.contributions)
            spark_df.write.mode("overwrite").saveAsTable(cfg.output_table)
            print(f"  Wrote Spark table: {cfg.output_table}")
        except Exception as e:
            print(f"  WARNING: Could not write Spark table: {e}")

    print("\n  Done.")


def plot_market_mix(
    output: MarketMixOutput,
    data: MarketMixData,
) -> None:
    """Plot diagnostic charts for the market mix model.

    Panels:
        (a) Waterfall: channel contributions
        (b) Actual vs predicted response
        (c) Response curves per channel
        (d) ROI bar chart
        (e) Stacked area: contributions over time
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    summary = output.channel_summary
    contributions = output.contributions
    response_curves = output.response_curves

    # -- (a) Waterfall: total contribution by channel --------------------
    ax = axes[0, 0]
    channels = summary[summary["channel"] != "base"]
    if not channels.empty:
        sorted_ch = channels.sort_values("total_contribution", ascending=True)
        ax.barh(
            sorted_ch["channel"],
            sorted_ch["total_contribution"],
            edgecolor="black",
            alpha=0.7,
        )
        ax.set_xlabel("Total Contribution")
        ax.set_title("Channel Contributions")

    # -- (b) Actual vs predicted -----------------------------------------
    ax = axes[0, 1]
    if "actual" in contributions.columns and "predicted" in contributions.columns:
        ax.plot(contributions["actual"].values, label="Actual", linewidth=1.5)
        ax.plot(contributions["predicted"].values, label="Predicted", linewidth=1.5, linestyle="--")
        ax.set_xlabel("Period")
        ax.set_ylabel("Response")
        ax.set_title("Actual vs Predicted")
        ax.legend()

    # -- (c) Response curves per channel ---------------------------------
    ax = axes[0, 2]
    if not response_curves.empty:
        for channel in response_curves["channel"].unique():
            ch_data = response_curves[response_curves["channel"] == channel]
            ax.plot(
                ch_data["spend_level"],
                ch_data["response"],
                label=channel,
                linewidth=1.5,
            )
        ax.set_xlabel("Spend Level")
        ax.set_ylabel("Response")
        ax.set_title("Response Curves")
        ax.legend(fontsize=8)

    # -- (d) ROI bar chart -----------------------------------------------
    ax = axes[1, 0]
    media_channels = summary[
        (summary["channel"] != "base")
        & (summary["roi"] > 0)
    ]
    if not media_channels.empty:
        sorted_roi = media_channels.sort_values("roi", ascending=True)
        ax.barh(
            sorted_roi["channel"],
            sorted_roi["roi"],
            edgecolor="black",
            alpha=0.7,
            color="steelblue",
        )
        ax.set_xlabel("ROI")
        ax.set_title("Channel ROI")

    # -- (e) Stacked area: contributions over time -----------------------
    ax = axes[1, 1]
    media_feature_cols = [
        f"{mc}_transformed" for mc in data.media_cols
        if f"{mc}_transformed" in contributions.columns
    ]
    if media_feature_cols:
        stack_data = contributions[media_feature_cols].clip(lower=0)
        ax.stackplot(
            range(len(stack_data)),
            *[stack_data[col].values for col in media_feature_cols],
            labels=[col.replace("_transformed", "") for col in media_feature_cols],
            alpha=0.7,
        )
        ax.set_xlabel("Period")
        ax.set_ylabel("Contribution")
        ax.set_title("Media Contributions Over Time")
        ax.legend(fontsize=8, loc="upper left")

    # -- (f) Contribution % pie ------------------------------------------
    ax = axes[1, 2]
    nonzero = summary[summary["total_contribution"] > 0]
    if not nonzero.empty:
        ax.pie(
            nonzero["total_contribution"],
            labels=nonzero["channel"],
            autopct="%1.1f%%",
            startangle=90,
        )
        ax.set_title("Contribution Share")

    plt.suptitle("Market Mix Model Results", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()
