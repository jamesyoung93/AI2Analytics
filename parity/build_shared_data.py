"""Generate the shared CSV used by the R<->Python parity check.

Deterministic given the seed. Both languages load this file (no independent
data regeneration) so any divergence in pipeline outputs is attributable to
the model implementations, not the inputs.

Run::

    python parity/build_shared_data.py

Writes ``parity/shared_mmm.csv``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def build(n_weeks: int = 156, seed: int = 12345) -> pd.DataFrame:
    """Synthetic weekly media + sales time-series (3 years).

    The data-generating process uses a log-saturation effect per channel so a
    market mix model fitted with default_saturation='log' should recover the
    relative ranking of channels and a non-trivial R^2.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-06", periods=n_weeks, freq="W-FRI")

    trend = np.linspace(100, 160, n_weeks)
    seasonality = 25 * np.sin(2 * np.pi * np.arange(n_weeks) / 52)

    tv  = rng.exponential(60, n_weeks).clip(0, 250)
    dig = rng.exponential(35, n_weeks).clip(0, 180)
    pr  = rng.exponential(25, n_weeks).clip(0, 120)

    sales = (
        trend
        + seasonality
        + 0.85 * np.log1p(tv)  * 12
        + 0.55 * np.log1p(dig) * 9
        + 0.25 * np.log1p(pr)  * 5
        + rng.normal(0, 8, n_weeks)
    )

    return pd.DataFrame({
        "WEEK_ENDING":      dates,
        "SALES":            sales.round(4),
        "TV_SPEND":         tv.round(4),
        "DIGITAL_SPEND":    dig.round(4),
        "PRINT_SPEND":      pr.round(4),
        "PRICE_INDEX":      (100 + rng.normal(0, 2, n_weeks)).round(4),
        "DISTRIBUTION_PCT": (85  + rng.normal(0, 3, n_weeks)).clip(70, 100).round(4),
    })


def main() -> None:
    out = Path(__file__).parent / "shared_mmm.csv"
    df = build()
    df.to_csv(out, index=False)
    print(f"Wrote {out}: {len(df)} rows, {len(df.columns)} cols.")


if __name__ == "__main__":
    main()
