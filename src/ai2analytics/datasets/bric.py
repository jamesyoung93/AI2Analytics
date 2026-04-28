"""Synthetic BRIC quarterly account data generator.

Generates realistic fake data for Brazil, Russia, India, and China markets.
Quarterly cadence, local-currency revenue, institution-level granularity.

Usage:
    python demos/synthetic_bric.py
    python demos/synthetic_bric.py --output-dir /tmp/bric_data

    from demos.synthetic_bric import generate_all
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

_COUNTRIES = ["BR", "RU", "IN", "CN"]
_COUNTRY_WEIGHTS = [0.25, 0.15, 0.35, 0.25]

_INSTITUTION_TYPES = ["hospital", "clinic", "pharmacy"]
_INSTITUTION_WEIGHTS = [0.35, 0.40, 0.25]

_CHANNELS = ["retail", "hospital"]
_SEGMENTS = ["A", "B", "C", "D"]
_SEGMENT_WEIGHTS = [0.12, 0.28, 0.35, 0.25]

_CITY_POOLS: dict[str, list[str]] = {
    "BR": ["Sao Paulo", "Rio de Janeiro", "Brasilia", "Salvador", "Belo Horizonte",
           "Fortaleza", "Curitiba", "Recife", "Manaus", "Porto Alegre",
           "Campinas", "Goiania", "Guarulhos", "Florianopolis"],
    "RU": ["Moscow", "Saint Petersburg", "Novosibirsk", "Yekaterinburg",
           "Kazan", "Nizhny Novgorod", "Samara", "Rostov-on-Don",
           "Chelyabinsk", "Omsk", "Krasnoyarsk", "Voronezh"],
    "IN": ["Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai",
           "Kolkata", "Ahmedabad", "Pune", "Jaipur", "Lucknow",
           "Chandigarh", "Kochi", "Bhopal", "Indore"],
    "CN": ["Shanghai", "Beijing", "Guangzhou", "Shenzhen", "Chengdu",
           "Hangzhou", "Wuhan", "Nanjing", "Chongqing", "Tianjin",
           "Suzhou", "Xi'an", "Zhengzhou", "Changsha"],
}

# Average revenue per unit by country (rough local-currency proxy)
_REVENUE_PER_UNIT: dict[str, float] = {
    "BR": 350.0,   # BRL
    "RU": 4500.0,  # RUB
    "IN": 1200.0,  # INR
    "CN": 280.0,   # CNY
}


def _quarter_ends(n_quarters: int, end_date: str = "2025-12-31") -> pd.DatetimeIndex:
    """Return *n_quarters* consecutive quarter-end dates ending near *end_date*."""
    end = pd.Timestamp(end_date) + pd.offsets.QuarterEnd(0)
    start = end - pd.DateOffset(months=3 * (n_quarters - 1))
    return pd.date_range(start, periods=n_quarters, freq="QE")


def _seasonal_quarterly(dates: pd.DatetimeIndex) -> np.ndarray:
    """Quarterly seasonal multiplier."""
    q = dates.quarter.values.astype(float)
    # Q1 dip, Q4 peak (budget flush)
    mult = {1: 0.88, 2: 1.00, 3: 1.02, 4: 1.12}
    return np.array([mult[int(qq)] for qq in q])


# ---------------------------------------------------------------------------
# 1. Account Master
# ---------------------------------------------------------------------------

def generate_account_master(
    n_accounts: int = 2000,
    seed: int = 44,
) -> pd.DataFrame:
    """Generate a BRIC-style account master table.

    Returns
    -------
    DataFrame with columns:
        ACCOUNT_ID           Numeric identifier.
        COUNTRY_CODE         BR / RU / IN / CN.
        CITY                 Synthetic city name.
        INSTITUTION_TYPE     hospital / clinic / pharmacy.
        CHANNEL              retail / hospital.
        CUSTOMER_SEGMENT     A / B / C / D.
    """
    rng = np.random.default_rng(seed)

    ids = np.arange(100_001, 100_001 + n_accounts)
    countries = rng.choice(_COUNTRIES, size=n_accounts, p=_COUNTRY_WEIGHTS)

    cities = np.array([rng.choice(_CITY_POOLS[c]) for c in countries])

    inst_types = rng.choice(
        _INSTITUTION_TYPES, size=n_accounts, p=_INSTITUTION_WEIGHTS
    )

    # Channel correlates with institution type
    channel_probs_hospital = 0.85  # hospitals are mostly hospital channel
    channels = np.where(
        inst_types == "hospital",
        np.where(rng.random(n_accounts) < channel_probs_hospital, "hospital", "retail"),
        np.where(
            inst_types == "pharmacy",
            "retail",
            np.where(rng.random(n_accounts) < 0.40, "hospital", "retail"),
        ),
    )

    segments = rng.choice(_SEGMENTS, size=n_accounts, p=_SEGMENT_WEIGHTS)

    return pd.DataFrame({
        "ACCOUNT_ID": ids,
        "COUNTRY_CODE": countries,
        "CITY": cities,
        "INSTITUTION_TYPE": inst_types,
        "CHANNEL": channels,
        "CUSTOMER_SEGMENT": segments,
    })


# ---------------------------------------------------------------------------
# 2. Quarterly Performance
# ---------------------------------------------------------------------------

def generate_quarterly_performance(
    account_master: pd.DataFrame,
    n_quarters: int = 8,
    seed: int = 44,
) -> pd.DataFrame:
    """Generate quarterly performance data for each account.

    Realistic features:
        - Zero-inflated: ~15-40% of account-quarters have zero activity
          depending on segment.
        - Seasonal variation with Q4 peak.
        - Revenue in local currency.

    Returns
    -------
    DataFrame with columns:
        ACCOUNT_ID, QUARTER_END, REVENUE_LOCAL, UNITS_DISPENSED,
        PATIENT_STARTS, COMPLIANCE_RATE
    """
    rng = np.random.default_rng(seed)
    quarters = _quarter_ends(n_quarters)
    seasonal = _seasonal_quarterly(quarters)

    ids = account_master["ACCOUNT_ID"].values
    countries = account_master["COUNTRY_CODE"].values
    segments = account_master["CUSTOMER_SEGMENT"].values
    n_accts = len(ids)

    # Base quarterly units by segment
    seg_base_units = {"A": 120.0, "B": 50.0, "C": 20.0, "D": 6.0}

    records: list[dict] = []
    for q_idx, qend in enumerate(quarters):
        # Observation rate by segment (sparsity)
        obs_prob = np.where(
            segments == "A", 0.92,
            np.where(segments == "B", 0.82,
                     np.where(segments == "C", 0.68, 0.55)),
        )
        observed = rng.random(n_accts) < obs_prob
        obs_idx = np.where(observed)[0]

        for j in obs_idx:
            seg = segments[j]
            cc = countries[j]
            base = seg_base_units[seg] * seasonal[q_idx]

            # Zero-inflate
            zero_prob = {"A": 0.05, "B": 0.15, "C": 0.30, "D": 0.45}[seg]
            if rng.random() < zero_prob:
                units = 0
            else:
                units = max(0, int(rng.poisson(max(base, 0.5))))

            rev_per_unit = _REVENUE_PER_UNIT[cc]
            revenue = round(units * rev_per_unit * rng.uniform(0.85, 1.15), 2)

            patient_starts = max(0, int(rng.poisson(units * 0.06 + 0.2)))

            # Compliance rate: beta-distributed, higher for better segments
            comp_alpha = {"A": 8, "B": 5, "C": 3, "D": 2}[seg]
            comp_beta = {"A": 2, "B": 3, "C": 4, "D": 5}[seg]
            compliance = round(float(rng.beta(comp_alpha, comp_beta)), 3)

            records.append({
                "ACCOUNT_ID": ids[j],
                "QUARTER_END": qend,
                "REVENUE_LOCAL": revenue,
                "UNITS_DISPENSED": units,
                "PATIENT_STARTS": patient_starts,
                "COMPLIANCE_RATE": compliance,
            })

    df = pd.DataFrame(records)
    df["QUARTER_END"] = pd.to_datetime(df["QUARTER_END"])
    return df


# ---------------------------------------------------------------------------
# 3. Engagement
# ---------------------------------------------------------------------------

def generate_engagement(
    quarterly: pd.DataFrame,
    seed: int = 44,
) -> pd.DataFrame:
    """Generate multi-channel engagement data aligned to quarterly periods.

    Returns
    -------
    DataFrame with columns:
        ACCOUNT_ID, QUARTER_END, REP_VISITS, CONGRESS_ATTENDANCE,
        SAMPLE_UNITS, DIGITAL_ENGAGEMENT_SCORE
    """
    rng = np.random.default_rng(seed)

    pairs = (
        quarterly[["ACCOUNT_ID", "QUARTER_END"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    n = len(pairs)

    # Rep visits: negative binomial, sparse
    has_rep = rng.random(n) < 0.45
    rep_visits = np.where(
        has_rep, rng.negative_binomial(2, 0.45, n), 0
    )

    # Congress attendance: binary, rare
    congress = rng.binomial(1, 0.08, n)

    # Sample units: zero-inflated Poisson
    has_sample = rng.random(n) < 0.30
    samples = np.where(has_sample, rng.poisson(15, n), 0)

    # Digital engagement score: 0-100, right-skewed
    digital = np.clip(rng.lognormal(3.0, 0.8, n), 0, 100).astype(int)
    # Zero out ~25 % (no digital footprint)
    digital[rng.random(n) < 0.25] = 0

    return pd.DataFrame({
        "ACCOUNT_ID": pairs["ACCOUNT_ID"].values,
        "QUARTER_END": pairs["QUARTER_END"].values,
        "REP_VISITS": rep_visits,
        "CONGRESS_ATTENDANCE": congress,
        "SAMPLE_UNITS": samples,
        "DIGITAL_ENGAGEMENT_SCORE": digital,
    })


# ---------------------------------------------------------------------------
# 4. Sales Alignment
# ---------------------------------------------------------------------------

def generate_sales_alignment(
    account_ids: np.ndarray,
    n_territories: int = 20,
    seed: int = 44,
    country_codes: np.ndarray | None = None,
) -> pd.DataFrame:
    """Assign accounts to sales territories.

    Returns
    -------
    DataFrame with columns: ACCOUNT_ID, SALES_TERRITORY_ID, COUNTRY_CODE
    """
    rng = np.random.default_rng(seed)

    coverage = rng.random(len(account_ids)) < 0.82
    covered = account_ids[coverage]
    territories = rng.integers(1, n_territories + 1, size=len(covered))

    df = pd.DataFrame({
        "ACCOUNT_ID": covered,
        "SALES_TERRITORY_ID": territories,
    })

    if country_codes is not None:
        cc_map = dict(zip(account_ids, country_codes))
        df["COUNTRY_CODE"] = df["ACCOUNT_ID"].map(cc_map)
    else:
        df["COUNTRY_CODE"] = ""

    return df


# ---------------------------------------------------------------------------
# 5. Generate All
# ---------------------------------------------------------------------------

def generate_all(
    output_dir: str | Path | None = None,
    seed: int = 44,
) -> dict[str, pd.DataFrame]:
    """Generate every BRIC synthetic table.

    Returns
    -------
    dict with keys:
        account_master, quarterly_performance, engagement,
        sales_alignment
    """
    print("Generating synthetic BRIC data (seed=%d) ..." % seed)

    master = generate_account_master(seed=seed)
    print(f"  account_master:           {len(master):>7,} rows")

    quarterly = generate_quarterly_performance(master, seed=seed)
    print(f"  quarterly_performance:    {len(quarterly):>7,} rows")

    engagement = generate_engagement(quarterly, seed=seed)
    print(f"  engagement:               {len(engagement):>7,} rows")

    alignment = generate_sales_alignment(
        master["ACCOUNT_ID"].values,
        seed=seed,
        country_codes=master["COUNTRY_CODE"].values,
    )
    print(f"  sales_alignment:          {len(alignment):>7,} rows")

    tables = {
        "account_master": master,
        "quarterly_performance": quarterly,
        "engagement": engagement,
        "sales_alignment": alignment,
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
        description="Generate synthetic BRIC account data for ai2analytics demos.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "data" / "bric"),
        help="Directory to write CSV files (default: demos/data/bric/)",
    )
    parser.add_argument("--seed", type=int, default=44)
    args = parser.parse_args()

    generate_all(output_dir=args.output_dir, seed=args.seed)
