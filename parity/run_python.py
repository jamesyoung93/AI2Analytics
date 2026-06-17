"""Run the canonical Market Mix demo in Python and save outputs as JSON.

Run after parity/build_shared_data.py::

    python parity/run_python.py

Writes ``parity/python_outputs.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ai2analytics.templates.market_mix import (
    MarketMixConfig, MarketMixPipeline,
)


def main() -> None:
    here = Path(__file__).parent
    ts = pd.read_csv(here / "shared_mmm.csv", parse_dates=["WEEK_ENDING"])

    cfg = MarketMixConfig(
        analysis_name="parity_mmm",
        col_date="WEEK_ENDING",
        col_response="SALES",
        media_columns=["TV_SPEND", "DIGITAL_SPEND", "PRINT_SPEND"],
        control_columns=["PRICE_INDEX", "DISTRIBUTION_PCT"],
        default_decay_rate=0.5,
        default_saturation="log",
        model_type="ridge",
        alpha=1.0,
        positive_coefficients=True,
        output_csv=str(here / "_py_contribs.csv"),
    )

    out = MarketMixPipeline().run(cfg, dataframes={"time_series": ts})

    summary = out.channel_summary.set_index("channel")
    diag = out.model_diagnostics.iloc[0]

    payload = {
        "language": "python",
        "n_periods": int(diag["n_periods"]),
        "r_squared": float(diag["r_squared"]),
        "adj_r_squared": float(diag["adjusted_r_squared"]),
        "mape": float(diag["mape"]),
        "intercept": float(diag["intercept"]),
        "coefficients": {
            ch: float(summary.loc[ch, "coefficient"])
            for ch in cfg.media_columns
        },
        "control_coefficients": {
            cc: float(summary.loc[cc, "coefficient"])
            for cc in cfg.control_columns
        },
        "channel_total_contribution": {
            ch: float(summary.loc[ch, "total_contribution"])
            for ch in cfg.media_columns
        },
        "channel_roi": {
            ch: float(summary.loc[ch, "roi"])
            for ch in cfg.media_columns
        },
        "predicted_sum": float(out.contributions["predicted"].sum()),
        "actual_sum": float(out.contributions["actual"].sum()),
        # First and last 5 predictions for spot-check
        "predicted_first5": out.contributions["predicted"].head(5).tolist(),
        "predicted_last5":  out.contributions["predicted"].tail(5).tolist(),
    }

    out_path = here / "python_outputs.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {out_path}")
    print(json.dumps(payload, indent=2)[:1000])


if __name__ == "__main__":
    main()
