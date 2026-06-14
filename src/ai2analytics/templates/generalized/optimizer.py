"""Generalized two-field-force optimizer with optional capacity tables."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import pulp

from ai2analytics.templates.generalized.config import GeneralizedDetailOptimizationConfig
from ai2analytics.templates.generalized.loader import _CAPACITY_COL
from ai2analytics.templates.generalized.reporting import GeneralizedReporter
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
    cfg: GeneralizedDetailOptimizationConfig,
    sc_df: pd.DataFrame,
    team_a_align: pd.DataFrame,
    team_b_align: pd.DataFrame,
    portfolio_decile: pd.DataFrame,
    priority_df: pd.DataFrame | None,
    team_a_capacity: pd.DataFrame | None = None,
    team_b_capacity: pd.DataFrame | None = None,
    reporter: GeneralizedReporter | None = None,
) -> OptimizationResult:
    """Run prep and LP optimization."""
    reporter = reporter or GeneralizedReporter(cfg.verbose, cfg.warn_on_defaults)

    col_npi = cfg.col_npi
    result = OptimizationResult()

    print("=" * 70)
    print("STAGE G: Prep pipeline (generalized)")
    print("=" * 70)

    territory_align = team_a_align.merge(team_b_align, on=col_npi, how="outer")
    print(
        f"  Combined alignment: {len(territory_align):,} rows, "
        f"{territory_align[col_npi].nunique():,} unique NPIs"
    )

    prep_cols = [col_npi, "scenario", "EV", "pred_prob", "pred_depth"]
    prep_df = sc_df[prep_cols].copy()
    prep_df = prep_df.merge(territory_align, on=col_npi, how="left")

    if priority_df is not None:
        prep_df = prep_df.merge(
            priority_df[[col_npi, "PRIORITY_TARGET_FLAG"]], on=col_npi, how="left"
        )
        prep_df["PRIORITY_TARGET_FLAG"] = prep_df["PRIORITY_TARGET_FLAG"].fillna(0).astype(int)
    else:
        prep_df["PRIORITY_TARGET_FLAG"] = 0

    if not portfolio_decile.empty:
        prep_df = prep_df.merge(
            portfolio_decile[[col_npi, cfg.col_portfolio_decile]],
            on=col_npi,
            how="left",
        )
        prep_df[cfg.col_portfolio_decile] = (
            prep_df[cfg.col_portfolio_decile].fillna(0).astype(int)
        )
    else:
        prep_df[cfg.col_portfolio_decile] = 0

    col_ta = cfg.col_team_a_territory
    col_tb = cfg.col_team_b_territory
    prep_df[col_ta] = prep_df[col_ta].fillna("0").astype(str)
    prep_df[col_tb] = prep_df[col_tb].fillna("0").astype(str)

    prep_df = prep_df.drop_duplicates(subset=[col_npi, "scenario"])
    result.prep_df = prep_df

    print(
        f"  Prepared: {len(prep_df):,} rows, {prep_df[col_npi].nunique():,} NPIs, "
        f"{prep_df['scenario'].nunique()} call levels"
    )

    print("\n" + "=" * 70)
    print("STAGE H: Optimizer (generalized)")
    print("=" * 70)

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

    baseline_scenario = 0 if 0 in set(prep_df["scenario"]) else min(prep_df["scenario"])
    if baseline_scenario != 0:
        reporter.warn(
            "scenario_range does not include 0; using the smallest scenario as the "
            "alignment baseline. Downstream EV base output may be less interpretable."
        )

    hp = (
        prep_df[prep_df["scenario"] == baseline_scenario]
        .drop_duplicates([col_ta, col_tb, col_npi])
        [[col_ta, col_tb, col_npi]]
        .reset_index(drop=True)
    )

    team_a_terrs = sorted(hp[col_ta].unique(), key=str)
    team_b_terrs = sorted(hp[col_tb].unique(), key=str)
    all_npis = sorted(hp[col_npi].unique())

    print(f"  {cfg.team_a_label} territories: {len(team_a_terrs)}")
    print(f"  {cfg.team_b_label} territories: {len(team_b_terrs)}")
    print(f"  NPIs to allocate:   {len(all_npis):,}")

    team_a_budget = _capacity_map(
        team_a_capacity,
        territory_col=col_ta,
        territories=team_a_terrs,
        fallback=cfg.team_a_budget_per_territory,
        label=cfg.team_a_label,
        reporter=reporter,
    )
    team_b_target = _capacity_map(
        team_b_capacity,
        territory_col=col_tb,
        territories=team_b_terrs,
        fallback=cfg.team_b_target_per_territory,
        label=cfg.team_b_label,
        reporter=reporter,
    )

    max_calls_np = cfg.max_calls_nonpriority
    max_scenario = max(cfg.scenario_range)

    pair_cache_np = allowed_call_pairs(
        max_calls_np,
        is_priority=False,
        priority_totals=cfg.priority_total_calls,
        require_mixed_at_max=cfg.require_mixed_at_max,
        max_scenario=max_scenario,
    )
    pair_cache_pt = allowed_call_pairs(
        max_scenario,
        is_priority=True,
        priority_totals=cfg.priority_total_calls,
        require_mixed_at_max=cfg.require_mixed_at_max,
        max_scenario=max_scenario,
    )

    def get_pairs(npi):
        return pair_cache_pt if prf_lookup.get(npi, 0) == 1 else pair_cache_np

    big_m = cfg.big_m_penalty
    beta = cfg.beta_decile

    prob_lp = pulp.LpProblem("Generalized_Call_Allocation", pulp.LpMaximize)

    z = {}
    for npi in all_npis:
        for a, b in get_pairs(npi):
            z[(npi, a, b)] = pulp.LpVariable(f"z_{npi}_{a}_{b}", 0, 1, cat="Binary")

    for npi in all_npis:
        prob_lp += (
            pulp.lpSum(z[(npi, a, b)] for a, b in get_pairs(npi)) == 1,
            f"one_pair_{npi}",
        )

    team_a_under = {}
    for ta in team_a_terrs:
        if _is_missing_territory(ta):
            continue
        npis_in_ta = hp[hp[col_ta] == ta][col_npi].unique()
        budget = team_a_budget.get(ta, cfg.team_a_budget_per_territory)
        team_a_under[ta] = pulp.LpVariable(f"a_under_{_safe_name(ta)}", 0, cat="Integer")
        prob_lp += (
            pulp.lpSum(
                a * z[(npi, a, b)]
                for npi in npis_in_ta
                for a, b in get_pairs(npi)
            )
            + team_a_under[ta]
            == budget,
            f"team_a_budget_{_safe_name(ta)}",
        )

    team_b_over = {}
    team_b_under = {}
    for tb in team_b_terrs:
        if _is_missing_territory(tb):
            continue
        npis_in_tb = hp[hp[col_tb] == tb][col_npi].unique()
        target = team_b_target.get(tb, cfg.team_b_target_per_territory)
        team_b_over[tb] = pulp.LpVariable(f"b_over_{_safe_name(tb)}", 0, cat="Integer")
        team_b_under[tb] = pulp.LpVariable(f"b_under_{_safe_name(tb)}", 0, cat="Integer")
        prob_lp += (
            pulp.lpSum(
                b * z[(npi, a, b)]
                for npi in npis_in_tb
                for a, b in get_pairs(npi)
            )
            - target
            == team_b_over[tb] - team_b_under[tb],
            f"team_b_target_{_safe_name(tb)}",
        )

    obj_ev = pulp.lpSum(
        ev_lookup.get((npi, a + b), 0) * z[(npi, a, b)]
        for npi in all_npis
        for a, b in get_pairs(npi)
    )
    obj_decile = pulp.lpSum(
        beta * dec_lookup.get(npi, 0) * b * z[(npi, a, b)]
        for npi in all_npis
        for a, b in get_pairs(npi)
    )
    obj_penalty = (
        big_m * pulp.lpSum(team_a_under.values())
        + big_m * pulp.lpSum(team_b_under.values())
        + big_m * pulp.lpSum(team_b_over.values())
    )
    prob_lp += obj_ev + obj_decile - obj_penalty

    print("  Solving LP...")
    prob_lp.solve(pulp.PULP_CBC_CMD(msg=False))
    result.status = pulp.LpStatus[prob_lp.status]
    print(f"  LP status: {result.status}")

    if result.status != "Optimal":
        reporter.warn(
            f"LP did not find an optimal solution (status={result.status}); proceeding "
            "with the best available solution."
        )

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

    print(
        f"\n  Plan: {len(result.plan_df):,} rows, "
        f"{result.plan_df[col_npi].nunique():,} NPIs"
    )
    deduped = result.plan_df.drop_duplicates(col_npi)
    print(f"  Total {cfg.team_a_label} calls: {deduped['team_a_calls'].sum():,}")
    print(f"  Total {cfg.team_b_label} calls: {deduped['team_b_calls'].sum():,}")

    npi_check = result.plan_df.groupby(col_npi)[["team_a_calls", "team_b_calls"]].nunique()
    inconsistent = npi_check[(npi_check["team_a_calls"] > 1) | (npi_check["team_b_calls"] > 1)]
    if len(inconsistent) == 0:
        print("  All NPIs have consistent allocations across territories")
    else:
        reporter.warn(f"{len(inconsistent)} NPIs have inconsistent allocations.")

    if team_a_under:
        result.team_a_slack = pd.DataFrame({
            col_ta: list(team_a_under.keys()),
            "missing_calls": [
                int(v.value()) if v.value() else 0 for v in team_a_under.values()
            ],
            "target_calls": [
                team_a_budget.get(k, cfg.team_a_budget_per_territory)
                for k in team_a_under
            ],
        })
        n_slack = (result.team_a_slack["missing_calls"] > 0).sum()
        print(f"  {cfg.team_a_label} territories with missing calls: {n_slack}")

    if team_b_under:
        result.team_b_slack = pd.DataFrame({
            col_tb: list(team_b_under.keys()),
            "over": [
                int(team_b_over[k].value()) if team_b_over[k].value() else 0
                for k in team_b_under
            ],
            "under": [
                int(team_b_under[k].value()) if team_b_under[k].value() else 0
                for k in team_b_under
            ],
            "target_calls": [
                team_b_target.get(k, cfg.team_b_target_per_territory)
                for k in team_b_under
            ],
        })
        print(
            f"  {cfg.team_b_label} slack - over: {result.team_b_slack['over'].sum()}, "
            f"under: {result.team_b_slack['under'].sum()}"
        )

    print("  Done.\n")
    return result


def _capacity_map(
    capacity: pd.DataFrame | None,
    territory_col: str,
    territories: list,
    fallback: int,
    label: str,
    reporter: GeneralizedReporter,
) -> dict:
    active_territories = [t for t in territories if not _is_missing_territory(t)]
    if capacity is None or capacity.empty:
        return {territory: fallback for territory in active_territories}

    cap = dict(zip(capacity[territory_col], capacity[_CAPACITY_COL]))
    missing = [territory for territory in active_territories if territory not in cap]
    if missing:
        reporter.warn(
            f"{label} capacity input is missing {len(missing):,} aligned territories; "
            f"using fallback {fallback} for those territories."
        )

    out = {}
    for territory in active_territories:
        value = cap.get(territory, fallback)
        rounded = int(round(float(value)))
        if float(value) != rounded:
            reporter.warn(
                f"{label} capacity for territory {territory!r} is non-integer ({value}); "
                f"rounding to {rounded} for the integer LP."
            )
        out[territory] = rounded
    return out


def _is_missing_territory(value) -> bool:
    return str(value).strip().lower() in {"", "0", "nan", "none"}


def _safe_name(value) -> str:
    text = str(value)
    return "".join(ch if ch.isalnum() else "_" for ch in text)
