"""Synthetic EU account-level pharma data generator.

Generates realistic fake data with European naming conventions: monthly
cadence, account IDs (not NPIs), KAM/medical team alignments, and
multi-country regions.

Usage:
    python demos/synthetic_eu_account.py
    python demos/synthetic_eu_account.py --output-dir /tmp/eu_data

    from demos.synthetic_eu_account import generate_all
    tables = generate_all()
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REGIONS = ["UK", "DE", "FR", "IT", "ES", "NL"]
_REGION_WEIGHTS = [0.22, 0.25, 0.20, 0.15, 0.12, 0.06]

_SEGMENTS = ["A", "B", "C"]
_SEGMENT_WEIGHTS = [0.20, 0.45, 0.35]

_ACCOUNT_NAME_PREFIXES = [
    "University Hospital", "City Clinic", "Regional Centre",
    "Medical Group", "Health Network", "Primary Care Practice",
    "Specialist Centre", "Polyclinic", "Community Health",
    "Integrated Care Hub",
]

_CITY_POOLS: dict[str, list[str]] = {
    "UK": ["London", "Manchester", "Birmingham", "Leeds", "Glasgow",
           "Bristol", "Liverpool", "Edinburgh", "Sheffield", "Newcastle"],
    "DE": ["Berlin", "Munich", "Hamburg", "Frankfurt", "Cologne",
           "Stuttgart", "Dusseldorf", "Leipzig", "Dortmund", "Dresden"],
    "FR": ["Paris", "Lyon", "Marseille", "Toulouse", "Bordeaux",
           "Lille", "Nantes", "Strasbourg", "Rennes", "Montpellier"],
    "IT": ["Rome", "Milan", "Naples", "Turin", "Florence",
           "Bologna", "Palermo", "Genoa", "Venice", "Verona"],
    "ES": ["Madrid", "Barcelona", "Valencia", "Seville", "Bilbao",
           "Malaga", "Zaragoza", "Murcia", "Palma", "Valladolid"],
    "NL": ["Amsterdam", "Rotterdam", "The Hague", "Utrecht", "Eindhoven",
           "Groningen", "Tilburg", "Almere", "Breda", "Nijmegen"],
}


def _month_ends(n_months: int, end_date: str = "2025-12-31") -> pd.DatetimeIndex:
    """Return *n_months* consecutive month-end dates ending near *end_date*."""
    end = pd.Timestamp(end_date) + pd.offsets.MonthEnd(0)
    start = end - pd.DateOffset(months=n_months - 1)
    return pd.date_range(start, periods=n_months, freq="ME")


def _seasonal_monthly(dates: pd.DatetimeIndex) -> np.ndarray:
    """Monthly seasonal multiplier (dip in summer, peak in winter)."""
    month = dates.month.values.astype(float)
    return 1.0 + 0.20 * np.cos(2 * np.pi * (month - 1) / 12)


# ---------------------------------------------------------------------------
# 1. Account Reference
# ---------------------------------------------------------------------------

def generate_account_reference(
    n_accounts: int = 3000,
    seed: int = 43,
) -> pd.DataFrame:
    """Generate an EU-style account reference table.

    Returns
    -------
    DataFrame with columns:
        PRESCRIBER_ID       ACC-XXXXX format identifier.
        ACCOUNT_NAME        Synthetic institution name.
        REGION              Two-letter country code.
        SEGMENT             A / B / C.
        TIER                1-4 (1 = highest value).
        UNITS_SOLD_L12M     Last-12-month unit volume.
    """
    rng = np.random.default_rng(seed)

    ids = [f"ACC-{i:05d}" for i in range(1, n_accounts + 1)]
    regions = rng.choice(_REGIONS, size=n_accounts, p=_REGION_WEIGHTS)
    segments = rng.choice(_SEGMENTS, size=n_accounts, p=_SEGMENT_WEIGHTS)

    # Tier correlates with segment: A -> more likely Tier 1-2
    tier_probs = {
        "A": [0.40, 0.35, 0.15, 0.10],
        "B": [0.10, 0.30, 0.35, 0.25],
        "C": [0.05, 0.15, 0.30, 0.50],
    }
    tiers = np.array([
        rng.choice([1, 2, 3, 4], p=tier_probs[s]) for s in segments
    ])

    # Account names: prefix + city
    names = []
    for region in regions:
        city = rng.choice(_CITY_POOLS[region])
        prefix = rng.choice(_ACCOUNT_NAME_PREFIXES)
        names.append(f"{prefix} {city}")

    # Units sold: higher for better segments / lower tiers
    base_units = np.where(
        segments == "A",
        rng.lognormal(6.5, 0.8, n_accounts),     # median ~665
        np.where(
            segments == "B",
            rng.lognormal(5.5, 0.9, n_accounts),  # median ~245
            rng.lognormal(4.5, 1.0, n_accounts),   # median ~90
        ),
    )
    # Tier multiplier: lower tier number = higher multiplier
    tier_mult = np.array([{1: 1.6, 2: 1.2, 3: 0.9, 4: 0.6}[t] for t in tiers])
    units = (base_units * tier_mult).astype(int)

    return pd.DataFrame({
        "PRESCRIBER_ID": ids,
        "ACCOUNT_NAME": names,
        "REGION": regions,
        "SEGMENT": segments,
        "TIER": tiers,
        "UNITS_SOLD_L12M": units,
    })


# ---------------------------------------------------------------------------
# 2. Account Monthly
# ---------------------------------------------------------------------------

def generate_account_monthly(
    account_ref: pd.DataFrame,
    n_months: int = 12,
    seed: int = 43,
) -> pd.DataFrame:
    """Generate monthly account-level performance data.

    Not every account appears every month (realistic sparsity: ~75% obs rate).

    Returns
    -------
    DataFrame with columns:
        PRESCRIBER_ID, MONTH_END, UNITS_SOLD, NEW_PATIENTS, MARKET_SHARE
    """
    rng = np.random.default_rng(seed)
    months = _month_ends(n_months)
    seasonal = _seasonal_monthly(months)

    ids = account_ref["PRESCRIBER_ID"].values
    annual_units = account_ref["UNITS_SOLD_L12M"].values.astype(float)
    segments = account_ref["SEGMENT"].values
    n_accts = len(ids)

    records: list[dict] = []
    for m_idx, month in enumerate(months):
        # Observation probability: Segment A observed more reliably
        obs_prob = np.where(segments == "A", 0.90,
                            np.where(segments == "B", 0.78, 0.60))
        observed = rng.random(n_accts) < obs_prob
        obs_ids = ids[observed]
        obs_annual = annual_units[observed]

        for j, acc_id in enumerate(obs_ids):
            monthly_mean = obs_annual[j] / 12.0 * seasonal[m_idx]
            units = max(0, int(rng.poisson(max(monthly_mean, 0.5))))
            new_patients = max(0, int(rng.poisson(units * 0.08 + 0.3)))
            # Market share: noisy but centred around segment-driven baseline
            share_base = min(0.45, max(0.02, obs_annual[j] / 5000.0))
            market_share = round(
                float(np.clip(rng.normal(share_base, 0.03), 0.0, 1.0)), 3
            )

            records.append({
                "PRESCRIBER_ID": acc_id,
                "MONTH_END": month,
                "UNITS_SOLD": units,
                "NEW_PATIENTS": new_patients,
                "MARKET_SHARE": market_share,
            })

    df = pd.DataFrame(records)
    df["MONTH_END"] = pd.to_datetime(df["MONTH_END"])
    return df


# ---------------------------------------------------------------------------
# 3. Visits
# ---------------------------------------------------------------------------

def generate_visits(
    account_monthly: pd.DataFrame,
    seed: int = 43,
) -> pd.DataFrame:
    """Generate multi-channel engagement records aligned to account monthly.

    Returns
    -------
    DataFrame with columns:
        PRESCRIBER_ID, MONTH_END, KAM_VISITS, MEDICAL_VISITS,
        DIGITAL_CONTACTS
    """
    rng = np.random.default_rng(seed)

    unique_pairs = (
        account_monthly[["PRESCRIBER_ID", "MONTH_END"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    n = len(unique_pairs)

    # Sparse: only a fraction of account-months get any visit
    has_kam = rng.random(n) < 0.35
    has_med = rng.random(n) < 0.20
    has_digi = rng.random(n) < 0.50

    keep = has_kam | has_med | has_digi
    sub = unique_pairs[keep].copy()
    m = len(sub)

    sub["KAM_VISITS"] = np.where(
        has_kam[keep], rng.negative_binomial(2, 0.60, m), 0
    )
    sub["MEDICAL_VISITS"] = np.where(
        has_med[keep], rng.negative_binomial(1, 0.55, m), 0
    )
    sub["DIGITAL_CONTACTS"] = np.where(
        has_digi[keep], rng.poisson(2.0, m), 0
    )

    return sub.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 4 & 5. Alignment tables
# ---------------------------------------------------------------------------

def generate_kam_alignment(
    account_ids: np.ndarray,
    n_territories: int = 30,
    seed: int = 43,
) -> pd.DataFrame:
    """Assign accounts to Key Account Manager territories.

    Returns
    -------
    DataFrame with columns: PRESCRIBER_ID, KAM_TERRITORY_ID
    """
    rng = np.random.default_rng(seed)
    coverage = rng.random(len(account_ids)) < 0.88
    covered = account_ids[coverage]
    territories = rng.integers(1, n_territories + 1, size=len(covered))

    return pd.DataFrame({
        "PRESCRIBER_ID": covered,
        "KAM_TERRITORY_ID": territories,
    })


def generate_medical_alignment(
    account_ids: np.ndarray,
    n_territories: int = 25,
    seed: int = 43,
) -> pd.DataFrame:
    """Assign accounts to Medical Science Liaison territories.

    Returns
    -------
    DataFrame with columns: PRESCRIBER_ID, MEDICAL_TERRITORY_ID
    """
    rng = np.random.default_rng(seed + 100)
    coverage = rng.random(len(account_ids)) < 0.72
    covered = account_ids[coverage]
    territories = rng.integers(1, n_territories + 1, size=len(covered))

    return pd.DataFrame({
        "PRESCRIBER_ID": covered,
        "MEDICAL_TERRITORY_ID": territories,
    })


# ---------------------------------------------------------------------------
# 6. Generate All
# ---------------------------------------------------------------------------

def generate_all(
    output_dir: str | Path | None = None,
    seed: int = 43,
) -> dict[str, pd.DataFrame]:
    """Generate every EU account synthetic table.

    Returns
    -------
    dict with keys:
        account_reference, account_monthly, visits,
        kam_alignment, medical_alignment
    """
    print("Generating synthetic EU account data (seed=%d) ..." % seed)

    acct_ref = generate_account_reference(seed=seed)
    print(f"  account_reference:  {len(acct_ref):>7,} rows")

    acct_monthly = generate_account_monthly(acct_ref, seed=seed)
    print(f"  account_monthly:    {len(acct_monthly):>7,} rows")

    visits = generate_visits(acct_monthly, seed=seed)
    print(f"  visits:             {len(visits):>7,} rows")

    acc_ids = acct_ref["PRESCRIBER_ID"].values
    kam = generate_kam_alignment(acc_ids, seed=seed)
    print(f"  kam_alignment:      {len(kam):>7,} rows")

    med = generate_medical_alignment(acc_ids, seed=seed)
    print(f"  medical_alignment:  {len(med):>7,} rows")

    tables = {
        "account_reference": acct_ref,
        "account_monthly": acct_monthly,
        "visits": visits,
        "kam_alignment": kam,
        "medical_alignment": med,
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
        description="Generate synthetic EU account data for ai2analytics demos.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "data" / "eu_account"),
        help="Directory to write CSV files (default: demos/data/eu_account/)",
    )
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args()

    generate_all(output_dir=args.output_dir, seed=args.seed)
