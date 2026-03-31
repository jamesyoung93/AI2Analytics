"""Synthetic US pharma HCP-level data generator.

Generates realistic fake data matching the schema expected by the
ai2analytics detail-optimization pipeline. All NPIs are random 10-digit
integers -- they are NOT real National Provider Identifiers.

Usage:
    python demos/synthetic_us_hcp.py              # writes CSVs to demos/data/us_hcp/
    python demos/synthetic_us_hcp.py --output-dir /tmp/my_data

    from demos.synthetic_us_hcp import generate_all
    tables = generate_all()  # dict of DataFrames, no files written
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_INDICATIONS = ["IND_A", "IND_B", "IND_C", "IND_D"]
_IND_WEIGHTS = [0.40, 0.30, 0.20, 0.10]  # IND_A is most common


def _fridays(n_weeks: int, end_date: str = "2025-12-26") -> pd.DatetimeIndex:
    """Return *n_weeks* consecutive Fridays ending on *end_date*."""
    end = pd.Timestamp(end_date)
    # Shift to previous Friday if end_date is not a Friday
    if end.weekday() != 4:
        end = end - pd.Timedelta(days=(end.weekday() - 4) % 7)
    start = end - pd.Timedelta(weeks=n_weeks - 1)
    return pd.date_range(start, periods=n_weeks, freq="W-FRI")


def _seasonal_multiplier(dates: pd.DatetimeIndex) -> np.ndarray:
    """Return a multiplicative seasonal factor (peaks in winter)."""
    day_of_year = dates.dayofyear.values.astype(float)
    # cos peaks at day 0 (Jan 1), trough at day 182 (July)
    return 1.0 + 0.25 * np.cos(2 * np.pi * day_of_year / 365.25)


# ---------------------------------------------------------------------------
# 1. HCP Reference
# ---------------------------------------------------------------------------

def generate_hcp_reference(
    n_npis: int = 5000,
    writer_rate: float = 0.35,
    target_rate: float = 0.60,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate an HCP-level reference table.

    Returns
    -------
    DataFrame with columns:
        npi               Random 10-digit integer (NOT a real NPI).
        WRITER_FLAG        'Y' or 'N'.
        TARGET_FLAG        'Y' or 'N'.
        IL_17_TRX_L12M     Last-12-month IL-17 Rx count.
        IL_23_TRX_L12M     Last-12-month IL-23 Rx count.
    """
    rng = np.random.default_rng(seed)

    npis = rng.integers(1_000_000_000, 9_999_999_999, size=n_npis, endpoint=True)
    npis = np.unique(npis)
    # Pad back to exactly n_npis if uniqueness shrank it
    while len(npis) < n_npis:
        extra = rng.integers(1_000_000_000, 9_999_999_999,
                             size=n_npis - len(npis), endpoint=True)
        npis = np.unique(np.concatenate([npis, extra]))
    npis = npis[:n_npis]

    is_writer = rng.random(n_npis) < writer_rate
    is_target = rng.random(n_npis) < target_rate

    # Writers prescribe more
    il17 = np.where(
        is_writer,
        rng.negative_binomial(3, 0.15, n_npis),   # mean ~17
        rng.negative_binomial(1, 0.40, n_npis),    # mean ~1.5
    )
    il23 = np.where(
        is_writer,
        rng.negative_binomial(2, 0.20, n_npis),    # mean ~8
        rng.negative_binomial(1, 0.50, n_npis),     # mean ~1
    )

    return pd.DataFrame({
        "npi": npis,
        "WRITER_FLAG": np.where(is_writer, "Y", "N"),
        "TARGET_FLAG": np.where(is_target, "Y", "N"),
        "IL_17_TRX_L12M": il17,
        "IL_23_TRX_L12M": il23,
    })


# ---------------------------------------------------------------------------
# 2. HCP Weekly
# ---------------------------------------------------------------------------

def generate_hcp_weekly(
    hcp_ref: pd.DataFrame,
    n_weeks: int = 52,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate weekly HCP-level activity data.

    Returns
    -------
    DataFrame with columns:
        npi, WEEK_ENDING, INDC, PAT_COUNT_REFERRED, TARGET_FLAG,
        SYMPTOM_SEVERITY_SCORE, PRIOR_AUTH_COUNT, SWITCH_FLAG_NUM
    """
    rng = np.random.default_rng(seed)
    fridays = _fridays(n_weeks)
    seasonal = _seasonal_multiplier(fridays)

    all_npis = hcp_ref["npi"].values
    writer_set = set(hcp_ref.loc[hcp_ref["WRITER_FLAG"] == "Y", "npi"].values)
    target_map = dict(zip(hcp_ref["npi"], hcp_ref["TARGET_FLAG"]))
    n_npis = len(all_npis)

    records: list[dict] = []
    for week_idx, friday in enumerate(fridays):
        # Each NPI has a probability of being observed this week
        # Writers are observed more often
        obs_probs = np.where(
            np.isin(all_npis, list(writer_set)),
            0.70,   # writers appear ~70 % of weeks
            0.30,   # non-writers ~30 %
        )
        observed = rng.random(n_npis) < obs_probs
        week_npis = all_npis[observed]

        for npi in week_npis:
            is_w = npi in writer_set
            # Zero-inflated Poisson referrals
            lam = (2.5 if is_w else 0.4) * seasonal[week_idx]
            zero_inflate_prob = 0.20 if is_w else 0.55
            if rng.random() < zero_inflate_prob:
                referrals = 0
            else:
                referrals = int(rng.poisson(lam))

            records.append({
                "npi": npi,
                "WEEK_ENDING": friday,
                "INDC": rng.choice(_INDICATIONS, p=_IND_WEIGHTS),
                "PAT_COUNT_REFERRED": referrals,
                "TARGET_FLAG": target_map.get(npi, "N"),
                "SYMPTOM_SEVERITY_SCORE": round(
                    float(rng.lognormal(1.0, 0.6)), 2
                ),
                "PRIOR_AUTH_COUNT": int(rng.poisson(0.8 if is_w else 0.3)),
                "SWITCH_FLAG_NUM": int(rng.binomial(1, 0.12 if is_w else 0.04)),
            })

    df = pd.DataFrame(records)
    df["WEEK_ENDING"] = pd.to_datetime(df["WEEK_ENDING"])
    return df


# ---------------------------------------------------------------------------
# 3. Calls
# ---------------------------------------------------------------------------

def generate_calls(
    hcp_weekly: pd.DataFrame,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate sparse face-to-face call records.

    Targets receive more calls. Not every NPI appears every week.

    Returns
    -------
    DataFrame with columns: NPI, WEEK_ENDING, HCP_F2F_CALLS
    """
    rng = np.random.default_rng(seed)

    unique_npis = hcp_weekly["npi"].unique()
    weeks = sorted(hcp_weekly["WEEK_ENDING"].unique())
    target_set = set(
        hcp_weekly.loc[hcp_weekly["TARGET_FLAG"] == "Y", "npi"].unique()
    )

    records: list[dict] = []
    for npi in unique_npis:
        is_tgt = npi in target_set
        # probability of a call in any given week
        call_prob = 0.25 if is_tgt else 0.08
        for w in weeks:
            if rng.random() < call_prob:
                n_calls = int(rng.negative_binomial(2, 0.55))  # mean ~1.6
                if n_calls > 0:
                    records.append({
                        "NPI": npi,
                        "WEEK_ENDING": w,
                        "HCP_F2F_CALLS": n_calls,
                    })

    df = pd.DataFrame(records)
    df["WEEK_ENDING"] = pd.to_datetime(df["WEEK_ENDING"])
    return df


# ---------------------------------------------------------------------------
# 4. Team Alignment
# ---------------------------------------------------------------------------

def generate_team_alignment(
    npis: np.ndarray,
    n_territories: int,
    coverage_rate: float = 0.85,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a team alignment table (HCP -> territory mapping).

    Not every NPI is covered (realistic gap). Each covered NPI belongs to
    exactly one territory.

    Returns
    -------
    DataFrame with columns: HCP_NPI, TERRITORY_ID
    """
    rng = np.random.default_rng(seed)

    covered_mask = rng.random(len(npis)) < coverage_rate
    covered_npis = npis[covered_mask]
    territories = rng.integers(1, n_territories + 1, size=len(covered_npis))

    return pd.DataFrame({
        "HCP_NPI": covered_npis,
        "TERRITORY_ID": territories,
    })


# ---------------------------------------------------------------------------
# 5. Portfolio Decile
# ---------------------------------------------------------------------------

def generate_portfolio_decile(
    hcp_ref: pd.DataFrame,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate portfolio-drug decile scores (1-10).

    Writers are skewed toward higher deciles.

    Returns
    -------
    DataFrame with columns: npi, PORTFOLIO_UNITS_DECILE
    """
    rng = np.random.default_rng(seed)
    n = len(hcp_ref)

    is_writer = (hcp_ref["WRITER_FLAG"] == "Y").values
    # Beta distribution shifted by writer status
    raw = np.where(
        is_writer,
        rng.beta(5, 2, n),    # skew high
        rng.beta(2, 5, n),    # skew low
    )
    deciles = np.clip(np.ceil(raw * 10).astype(int), 1, 10)

    return pd.DataFrame({
        "npi": hcp_ref["npi"].values,
        "PORTFOLIO_UNITS_DECILE": deciles,
    })


# ---------------------------------------------------------------------------
# 6. Priority Targets
# ---------------------------------------------------------------------------

def generate_priority_targets(
    hcp_ref: pd.DataFrame,
    priority_rate: float = 0.10,
    seed: int = 42,
) -> pd.DataFrame:
    """Select a subset of high-value writers as priority targets.

    Returns
    -------
    DataFrame with columns: npi, PRIORITY_TARGET  (all rows have value 'Y')
    """
    rng = np.random.default_rng(seed)

    writers = hcp_ref[hcp_ref["WRITER_FLAG"] == "Y"].copy()
    # Priority is concentrated among the highest-prescribing writers
    il_total = (
        writers["IL_17_TRX_L12M"].values + writers["IL_23_TRX_L12M"].values
    ).astype(float)
    # Softmax-ish selection probability
    probs = il_total / (il_total.sum() + 1e-9)
    n_priority = max(1, int(len(hcp_ref) * priority_rate))
    n_priority = min(n_priority, len(writers))

    chosen_idx = rng.choice(len(writers), size=n_priority, replace=False, p=probs)

    return pd.DataFrame({
        "npi": writers.iloc[chosen_idx]["npi"].values,
        "PRIORITY_TARGET": "Y",
    })


# ---------------------------------------------------------------------------
# 7. Generate All
# ---------------------------------------------------------------------------

def generate_all(
    output_dir: str | Path | None = None,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Generate every US HCP synthetic table.

    Parameters
    ----------
    output_dir : str or Path, optional
        If provided, save each table as a CSV under this directory.
    seed : int
        Base seed for reproducibility.

    Returns
    -------
    dict mapping table name to DataFrame.  Keys:
        hcp_reference, hcp_weekly, calls,
        team_a_alignment, team_b_alignment,
        portfolio_decile, priority_targets
    """
    print("Generating synthetic US HCP data (seed=%d) ..." % seed)

    hcp_ref = generate_hcp_reference(seed=seed)
    print(f"  hcp_reference:      {len(hcp_ref):>7,} rows")

    hcp_weekly = generate_hcp_weekly(hcp_ref, seed=seed)
    print(f"  hcp_weekly:         {len(hcp_weekly):>7,} rows")

    calls = generate_calls(hcp_weekly, seed=seed)
    print(f"  calls:              {len(calls):>7,} rows")

    npis = hcp_ref["npi"].values
    team_a = generate_team_alignment(npis, n_territories=50, coverage_rate=0.85,
                                     seed=seed)
    print(f"  team_a_alignment:   {len(team_a):>7,} rows")

    team_b = generate_team_alignment(npis, n_territories=40, coverage_rate=0.80,
                                     seed=seed + 1)
    print(f"  team_b_alignment:   {len(team_b):>7,} rows")

    portfolio = generate_portfolio_decile(hcp_ref, seed=seed)
    print(f"  portfolio_decile:   {len(portfolio):>7,} rows")

    priority = generate_priority_targets(hcp_ref, seed=seed)
    print(f"  priority_targets:   {len(priority):>7,} rows")

    tables = {
        "hcp_reference": hcp_ref,
        "hcp_weekly": hcp_weekly,
        "calls": calls,
        "team_a_alignment": team_a,
        "team_b_alignment": team_b,
        "portfolio_decile": portfolio,
        "priority_targets": priority,
    }

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        for name, df in tables.items():
            path = out / f"{name}.csv"
            df.to_csv(path, index=False)
            print(f"  -> {path}")

    print("Done.\n")
    return tables


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic US HCP data for ai2analytics demos.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "data" / "us_hcp"),
        help="Directory to write CSV files (default: demos/data/us_hcp/)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate_all(output_dir=args.output_dir, seed=args.seed)
