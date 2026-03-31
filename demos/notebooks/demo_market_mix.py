"""Market Mix Model (MMM) Demo
===============================

Demonstrates the ai2analytics market mix modeling pipeline using
synthetic weekly media-spend and sales data.  No Spark required.

What this script covers:
    1. Building a synthetic time-series with known structure:
       - Linear trend + seasonality
       - Three media channels: TV, Digital, Print
       - Two control variables: Price Index, Distribution %
       - Log-transformed diminishing returns baked into the DGP
    2. Configuring and running the MarketMixPipeline.
    3. Understanding the output:
       - Adstock transformation (geometric carry-over of media spend)
       - Saturation curves (diminishing returns at high spend)
       - Contribution decomposition (how much each channel drove sales)
       - Channel ROI (return per unit of spend)
    4. Interpreting the channel summary and response curves.

Run from the repository root:
    python demos/notebooks/demo_market_mix.py
"""

from __future__ import annotations

import os
import sys

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# ============================================================================
# Step 1 -- Build synthetic time-series data
# ============================================================================
print("\n" + "#" * 70)
print("# STEP 1: Generate synthetic media + sales time-series")
print("#" * 70)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

rng = np.random.default_rng(42)
n_weeks = 104  # 2 years of weekly data

dates = pd.date_range("2024-01-05", periods=n_weeks, freq="W-FRI")

# Structural components
trend = np.linspace(100, 150, n_weeks)
seasonality = 20 * np.sin(2 * np.pi * np.arange(n_weeks) / 52)

# Media spend (exponentially distributed, clipped)
tv_spend = rng.exponential(50, n_weeks).clip(0, 200)
digital_spend = rng.exponential(30, n_weeks).clip(0, 150)
print_spend = rng.exponential(20, n_weeks).clip(0, 100)

# Sales = trend + seasonality + log-transformed media effects + noise
# The log1p transform creates diminishing returns -- the pipeline's
# saturation stage should recover this shape.
sales = (
    trend
    + seasonality
    + 0.8 * np.log1p(tv_spend) * 10       # TV: strongest effect
    + 0.5 * np.log1p(digital_spend) * 8   # Digital: moderate
    + 0.3 * np.log1p(print_spend) * 5     # Print: weakest
    + rng.normal(0, 10, n_weeks)           # noise
).clip(50)

ts = pd.DataFrame({
    "WEEK_ENDING": dates,
    "SALES": sales.round(0),
    "TV_SPEND": tv_spend.round(2),
    "DIGITAL_SPEND": digital_spend.round(2),
    "PRINT_SPEND": print_spend.round(2),
    "PRICE_INDEX": (100 + rng.normal(0, 2, n_weeks)).round(1),
    "DISTRIBUTION_PCT": (85 + rng.normal(0, 3, n_weeks)).clip(70, 100).round(1),
})

print(f"\nTime-series shape: {ts.shape}  ({n_weeks} weeks)")
print(f"Date range: {ts['WEEK_ENDING'].min().date()} to {ts['WEEK_ENDING'].max().date()}")
print(f"\nFirst 5 rows:")
print(ts.head().to_string(index=False))

print(f"\nDescriptive stats:")
print(ts.describe().round(2).to_string())

print("""
Data generation notes:
  - SALES is the response variable we want to decompose.
  - TV_SPEND, DIGITAL_SPEND, PRINT_SPEND are media channels.
  - PRICE_INDEX and DISTRIBUTION_PCT are control variables.
  - The true relationship uses log1p(spend), so the pipeline's
    saturation transform should capture diminishing returns.
""")

# ============================================================================
# Step 2 -- Configure the pipeline
# ============================================================================
print("#" * 70)
print("# STEP 2: Configure the MarketMix pipeline")
print("#" * 70)

from ai2analytics.templates.market_mix import (  # noqa: E402
    MarketMixConfig,
    MarketMixPipeline,
)

cfg = MarketMixConfig(
    analysis_name="demo_mmm",
    col_date="WEEK_ENDING",
    col_response="SALES",
    media_columns=["TV_SPEND", "DIGITAL_SPEND", "PRINT_SPEND"],
    control_columns=["PRICE_INDEX", "DISTRIBUTION_PCT"],
    output_csv="demos/data/mmm_output.csv",
)

print(f"\nConfiguration:")
print(f"  analysis_name:       {cfg.analysis_name}")
print(f"  col_date:            {cfg.col_date}")
print(f"  col_response:        {cfg.col_response}")
print(f"  media_columns:       {cfg.media_columns}")
print(f"  control_columns:     {cfg.control_columns}")
print(f"  model_type:          {cfg.model_type}  (default: ridge regression)")
print(f"  default_decay_rate:  {cfg.default_decay_rate}  (adstock carry-over)")
print(f"  default_saturation:  {cfg.default_saturation}  (hill function)")
print(f"  include_trend:       {cfg.include_trend}")
print(f"  include_seasonality: {cfg.include_seasonality}")
print(f"  output_csv:          {cfg.output_csv}")

# ============================================================================
# Step 3 -- Run the pipeline
# ============================================================================
print("\n" + "#" * 70)
print("# STEP 3: Run the MarketMix pipeline")
print("#" * 70)

print("""
The pipeline runs these stages:
  1. LOAD      -- Read time-series (here from a DataFrame, not Spark).
  2. TRANSFORM -- Apply adstock (geometric decay) and saturation (Hill)
                   to each media channel.  Add trend + seasonality features.
  3. FIT       -- Ridge regression: SALES ~ media_transformed + controls
                   + trend + seasonality.  Positive coefficients enforced.
  4. OUTPUT    -- Decompose contributions, compute ROI, build response curves.
  5. WRITE     -- Save results to CSV.
""")

pipeline = MarketMixPipeline()
output = pipeline.run(cfg, dataframes={"time_series": ts})

# ============================================================================
# Step 4 -- Interpret results
# ============================================================================
print("\n" + "#" * 70)
print("# STEP 4: Interpret results")
print("#" * 70)

# 4a. Channel summary
print("\n--- Channel Summary ---")
print(output.channel_summary.to_string(index=False))

print("""
How to read the channel summary:
  channel            -- media channel or structural component
  total_spend        -- sum of raw spend across all weeks
  total_contribution -- model-attributed sales contribution
  contribution_pct   -- share of total sales explained by this channel
  roi                -- total_contribution / total_spend
  coefficient        -- regression coefficient on the transformed feature
  decay_rate         -- adstock decay parameter used
""")

# 4b. ROI ranking
print("--- ROI Ranking ---")
media_rows = output.channel_summary[
    output.channel_summary["total_spend"] > 0
].sort_values("roi", ascending=False)
for _, row in media_rows.iterrows():
    print(f"  {row['channel']:<20s}  ROI = {row['roi']:.4f}")

print("""
ROI interpretation:
  A ROI of 0.50 means that for every $1 spent on this channel,
  the model attributes $0.50 of incremental sales.

  The true data-generating process gave TV the strongest effect,
  Digital moderate, Print weakest -- the model should roughly
  recover that ordering.
""")

# 4c. Model diagnostics
print("--- Model Diagnostics ---")
print(output.model_diagnostics.to_string(index=False))
print()

# 4d. Contributions over time (first/last 5 periods)
print("--- Contributions (first 5 weeks) ---")
print(output.contributions.head().to_string(index=False))

print("\n--- Contributions (last 5 weeks) ---")
print(output.contributions.tail().to_string(index=False))

print("""
Each row is one week.  The columns show how much of the predicted
SALES came from each channel, plus the base (intercept), trend,
seasonality, and control variables.  The 'actual' and 'predicted'
columns let you judge model fit.
""")

# 4e. Response curves (sample)
print("--- Response Curves (sample points) ---")
for channel in cfg.media_columns:
    ch_data = output.response_curves[
        output.response_curves["channel"] == channel
    ]
    # Show 5 evenly-spaced points
    sample = ch_data.iloc[:: max(1, len(ch_data) // 5)]
    print(f"\n  {channel}:")
    for _, r in sample.iterrows():
        print(f"    spend={r['spend_level']:>8.1f}  ->  response={r['response']:>8.4f}")

print("""
Response curves show the marginal sales contribution at different
spend levels after adstock and saturation transforms.  The curves
flatten at high spend -- this is the diminishing-returns (saturation)
effect.  Channels with steeper initial slopes deliver more bang per
dollar at low spend levels.
""")

# ============================================================================
# Done
# ============================================================================
print("#" * 70)
print("# DEMO COMPLETE")
print("#" * 70)
print(f"\nOutput written to: {os.path.abspath(cfg.output_csv)}")
print("\nThis demo showed how the MarketMixPipeline decomposes sales into")
print("base, media, and control contributions, computes per-channel ROI,")
print("and generates saturation response curves -- all without Spark.")
