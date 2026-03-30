"""Stage G-H: Prep pipeline and PuLP LP optimizer for call allocation."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import pulp

from ai2analytics.templates.detail_optimization.config import DetailOptimizationConfig
from ai2analytics.utils import allowed_call_pairs


@dataclass
class OptimizationResult:
    """Container for optimizer outputs."""
    plan_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    prep_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    status: str = ""
    team_a_slack: pd.DataFrame | None = None
    team_b_slack: pd.DataFrame | None = None


def prep_and_optimize(
    cfg: DetailOptimizationConfig,
    sc_df: pd.DataFrame,
    team_a_align: pd.DataFrame,
    team_b_align: pd.DataFrame,
    portfolio_decile: pd.DataFrame,
    priority_df: pd.DataFrame | None,
) -> OptimizationResult:
    """Run stages G (prep pipeline) and H (LP optimizer)."""

    col_npi = cfg.col_npi
    result = OptimizationResult()

    # ── G: Prep pipeline ────────────────────────────────────────────────
    print("=" * 70)
    print("STAGE G: Prep pipeline")
    print("=" * 70)

    # G1. Merge alignments
    territory_align = team_a_align.merge(team_b_align, on=col_npi, how="outer")
    print(
        f"  Combined alignment: {len(territory_align):,} rows, "
        f"{territory_align[col_npi].nunique():,} unique NPIs"
    )

    # G2. Build prepared dataframe
    prep_cols = [col_npi, "scenario", "EV", "pred_prob", "pred_depth"]
    prep_df = sc_df[prep_cols].copy()
    prep_df = prep_df.merge(territory_align, on=col_npi, how="left")

    # G3. Priority target flag
    if priority_df is not None:
        prep_df = prep_df.merge(
            priority_df[[col_npi, "PRIORITY_TARGET_FLAG"]], on=col_npi, how="left"
        )
        prep_df["PRIORITY_TARGET_FLAG"] = prep_df["PRIORITY_TARGET_FLAG"].fillna(0).astype(int)
    else:
        prep_df["PRIORITY_TARGET_FLAG"] = 0

    # G4. Portfolio-drug decile
    if not portfolio_decile.empty:
        prep_df = prep_df.merge(
            portfolio_decile[[col_npi, cfg.col_portfolio_decile]],
            on=col_npi, how="left",
        )
        prep_df[cfg.col_portfolio_decile] = (
            prep_df[cfg.col_portfolio_decile].fillna(0).astype(int)
        )
    else:
        prep_df[cfg.col_portfolio_decile] = 0

    # G5. Fill missing territories
    col_ta = cfg.col_team_a_territory
    col_tb = cfg.col_team_b_territory
    prep_df[col_ta] = prep_df[col_ta].fillna(0).astype(int)
    prep_df[col_tb] = prep_df[col_tb].fillna(0).astype(int)

    # G6. Deduplicate
    prep_df = prep_df.drop_duplicates(subset=[col_npi, "scenario"])
    result.prep_df = prep_df

    print(
        f"  Prepared: {len(prep_df):,} rows, {prep_df[col_npi].nunique():,} NPIs, "
        f"{prep_df['scenario'].nunique()} call levels"
    )

    # ── H: Optimizer ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STAGE H: Optimizer")
    print("=" * 70)

    # H1. Build lookup structures
    ev_lookup = {
        (row[col_npi], row["scenario"]): row["EV"]
        for _, row in prep_df.iterrows()
    }

    dec_lookup = (
        prep_df.drop_duplicates(col_npi)
        .set_index(col_npi)[cfg.col_portfolio_decile]
        .to_dict()
    )

    prf_lookup = (
        prep_df.drop_duplicates(col_npi)
        .set_index(col_npi)["PRIORITY_TARGET_FLAG"]
        .to_dict()
    )

    # One row per NPI per territory combo (at scenario=0)
    hp = (
        prep_df[prep_df["scenario"] == 0]
        .drop_duplicates([col_ta, col_tb, col_npi])
        [[col_ta, col_tb, col_npi]]
        .reset_index(drop=True)
    )

    team_a_terrs = sorted(hp[col_ta].unique())
    team_b_terrs = sorted(hp[col_tb].unique())
    all_npis = sorted(hp[col_npi].unique())

    print(f"  Team A territories: {len(team_a_terrs)}")
    print(f"  Team B territories: {len(team_b_terrs)}")
    print(f"  NPIs to allocate:   {len(all_npis):,}")

    # H2. Pre-compute allowed call pairs
    max_calls_np = cfg.max_calls_nonpriority
    max_scenario = max(cfg.scenario_range)

    pair_cache_np = allowed_call_pairs(
        max_calls_np, is_priority=False,
        priority_totals=cfg.priority_total_calls,
        require_mixed_at_max=cfg.require_mixed_at_max,
        max_scenario=max_scenario,
    )
    pair_cache_pt = allowed_call_pairs(
        max_scenario, is_priority=True,
        priority_totals=cfg.priority_total_calls,
        require_mixed_at_max=cfg.require_mixed_at_max,
        max_scenario=max_scenario,
    )

    def get_pairs(npi):
        return pair_cache_pt if prf_lookup.get(npi, 0) == 1 else pair_cache_np

    # H3. Build LP
    BIG_M = cfg.big_m_penalty
    BETA = cfg.beta_decile
    BUD_A = cfg.team_a_budget_per_territory
    TGT_B = cfg.team_b_target_per_territory

    prob_lp = pulp.LpProblem("Call_Allocation", pulp.LpMaximize)

    z = {}
    for npi in all_npis:
        for a, b in get_pairs(npi):
            z[(npi, a, b)] = pulp.LpVariable(f"z_{npi}_{a}_{b}", 0, 1, cat="Binary")

    # Each NPI picks exactly one (a, b) pair
    for npi in all_npis:
        prob_lp += (
            pulp.lpSum(z[(npi, a, b)] for a, b in get_pairs(npi)) == 1,
            f"one_pair_{npi}",
        )

    # H4. Team A budget constraints
    team_a_under = {}
    for ta in team_a_terrs:
        if ta == 0:
            continue
        npis_in_ta = hp[hp[col_ta] == ta][col_npi].unique()
        team_a_under[ta] = pulp.LpVariable(f"a_under_{ta}", 0, cat="Integer")
        prob_lp += (
            pulp.lpSum(
                a * z[(npi, a, b)]
                for npi in npis_in_ta
                for a, b in get_pairs(npi)
            )
            + team_a_under[ta]
            == BUD_A,
            f"team_a_budget_{ta}",
        )

    # H5. Team B target constraints
    team_b_over = {}
    team_b_under = {}
    for tb in team_b_terrs:
        if tb == 0:
            continue
        npis_in_tb = hp[hp[col_tb] == tb][col_npi].unique()
        team_b_over[tb] = pulp.LpVariable(f"b_over_{tb}", 0, cat="Integer")
        team_b_under[tb] = pulp.LpVariable(f"b_under_{tb}", 0, cat="Integer")
        prob_lp += (
            pulp.lpSum(
                b * z[(npi, a, b)]
                for npi in npis_in_tb
                for a, b in get_pairs(npi)
            )
            - TGT_B
            == team_b_over[tb] - team_b_under[tb],
            f"team_b_target_{tb}",
        )

    # H6. Objective
    obj_ev = pulp.lpSum(
        ev_lookup.get((npi, a + b), 0) * z[(npi, a, b)]
        for npi in all_npis
        for a, b in get_pairs(npi)
    )
    obj_decile = pulp.lpSum(
        BETA * dec_lookup.get(npi, 0) * b * z[(npi, a, b)]
        for npi in all_npis
        for a, b in get_pairs(npi)
    )
    obj_penalty = (
        BIG_M * pulp.lpSum(team_a_under.values())
        + BIG_M * pulp.lpSum(team_b_under.values())
        + BIG_M * pulp.lpSum(team_b_over.values())
    )
    prob_lp += obj_ev + obj_decile - obj_penalty

    # H7. Solve
    print("  Solving LP...")
    prob_lp.solve(pulp.PULP_CBC_CMD(msg=False))
    result.status = pulp.LpStatus[prob_lp.status]
    print(f"  LP status: {result.status}")

    if result.status != "Optimal":
        print(f"  WARNING: LP did not find optimal solution (status={result.status}).")
        print("  Proceeding with best available solution — review results carefully.")

    # H8. Extract solution
    npi_allocation = {}
    for npi in all_npis:
        for a, b in get_pairs(npi):
            val = z[(npi, a, b)].value()
            if val is not None and val > 0.5:
                npi_allocation[npi] = (a, b)
                break

    plan_rows = []
    for _, row in hp.iterrows():
        npi = row[col_npi]
        a_calls, b_calls = npi_allocation.get(npi, (0, 0))
        plan_rows.append({
            col_ta: row[col_ta],
            col_tb: row[col_tb],
            col_npi: npi,
            "team_a_calls": a_calls,
            "team_b_calls": b_calls,
            "total_calls": a_calls + b_calls,
            cfg.col_portfolio_decile: dec_lookup.get(npi, 0),
            "PRIORITY_TARGET_FLAG": prf_lookup.get(npi, 0),
        })

    result.plan_df = pd.DataFrame(plan_rows)

    # H9. Diagnostics
    print(
        f"\n  Optimal plan: {len(result.plan_df):,} rows, "
        f"{result.plan_df[col_npi].nunique():,} NPIs"
    )
    deduped = result.plan_df.drop_duplicates(col_npi)
    print(f"  Total Team A calls: {deduped['team_a_calls'].sum():,}")
    print(f"  Total Team B calls: {deduped['team_b_calls'].sum():,}")

    # Consistency check
    npi_check = result.plan_df.groupby(col_npi)[["team_a_calls", "team_b_calls"]].nunique()
    inconsistent = npi_check[(npi_check["team_a_calls"] > 1) | (npi_check["team_b_calls"] > 1)]
    if len(inconsistent) == 0:
        print("  All NPIs have consistent allocations across territories")
    else:
        print(f"  WARNING: {len(inconsistent)} NPIs have inconsistent allocations!")

    # Slack diagnostics
    if team_a_under:
        result.team_a_slack = pd.DataFrame({
            col_ta: list(team_a_under.keys()),
            "missing_calls": [int(v.value()) if v.value() else 0 for v in team_a_under.values()],
        })
        n_slack = (result.team_a_slack["missing_calls"] > 0).sum()
        print(f"  Team A territories with missing calls: {n_slack}")

    if team_b_under:
        result.team_b_slack = pd.DataFrame({
            col_tb: list(team_b_under.keys()),
            "over": [int(team_b_over[k].value()) if team_b_over[k].value() else 0 for k in team_b_under],
            "under": [int(team_b_under[k].value()) if team_b_under[k].value() else 0 for k in team_b_under],
        })
        print(
            f"  Team B slack — over: {result.team_b_slack['over'].sum()}, "
            f"under: {result.team_b_slack['under'].sum()}"
        )

    print("  Done.\n")
    return result
