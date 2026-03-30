"""Stage I-J: Post-processing and final output."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ai2analytics.templates.detail_optimization.config import DetailOptimizationConfig
from ai2analytics.templates.detail_optimization.optimizer import OptimizationResult
from ai2analytics.utils import make_decile, yn_flag


@dataclass
class PipelineOutput:
    """Final pipeline output."""
    portfolio: pd.DataFrame = field(default_factory=pd.DataFrame)
    plan_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    ev_by_npi: pd.DataFrame = field(default_factory=pd.DataFrame)
    metadata: pd.DataFrame = field(default_factory=pd.DataFrame)


def post_process(
    cfg: DetailOptimizationConfig,
    opt_result: OptimizationResult,
    hcp_reference: pd.DataFrame,
    priority_df: pd.DataFrame | None,
) -> PipelineOutput:
    """Run post-processing stages I and J."""
    print("=" * 70)
    print("STAGE I: Post-processing")
    print("=" * 70)

    col_npi = cfg.col_npi
    col_ta = cfg.col_team_a_territory
    col_tb = cfg.col_team_b_territory
    col_out_calls = cfg.col_output_calls
    col_out_terr = cfg.col_output_territory
    col_out_src = cfg.col_output_source
    col_out_npi = cfg.col_output_npi_str

    plan_df = opt_result.plan_df
    prep_df = opt_result.prep_df
    output = PipelineOutput(plan_df=plan_df)

    # ── I1. Build metadata from HCP reference ────────────────────────
    meta_cols = [col_npi]
    if cfg.col_writer_flag in hcp_reference.columns:
        meta_cols.append(cfg.col_writer_flag)
    if cfg.col_target_flag in hcp_reference.columns:
        meta_cols.append(cfg.col_target_flag)
    for src, _ in cfg.il_rx_columns:
        if src in hcp_reference.columns:
            meta_cols.append(src)

    meta = hcp_reference[meta_cols].drop_duplicates(col_npi).copy()

    if cfg.col_writer_flag in meta.columns:
        meta[cfg.col_writer_flag] = yn_flag(meta[cfg.col_writer_flag])
    if cfg.col_target_flag in meta.columns:
        meta[cfg.col_target_flag] = yn_flag(meta[cfg.col_target_flag])

    for src, tgt in cfg.il_rx_columns:
        if src in meta.columns:
            meta = make_decile(meta, src, tgt, n_bins=cfg.decile_bins)

    meta[col_out_npi] = meta[col_npi].astype(str)
    output.metadata = meta
    print(f"  Metadata: {len(meta):,} NPIs")

    # ── I2. Build EV lookup ──────────────────────────────────────────
    SCALE = cfg.ev_scale_factor

    ev_base = (
        prep_df[prep_df["scenario"] == 0][[col_npi, "EV"]]
        .rename(columns={"EV": "EV_base"})
        .drop_duplicates(col_npi)
    )
    ev_base["EV_base"] = ev_base["EV_base"] * SCALE

    ev_all = (
        prep_df[[col_npi, "scenario", "EV"]]
        .rename(columns={"EV": "EV_planned"})
    )
    ev_all["EV_planned"] = ev_all["EV_planned"] * SCALE

    npi_calls = plan_df[[col_npi, "total_calls"]].drop_duplicates(col_npi)

    ev_by_npi = (
        npi_calls
        .merge(ev_base, on=col_npi, how="left")
        .merge(
            ev_all,
            left_on=[col_npi, "total_calls"],
            right_on=[col_npi, "scenario"],
            how="left",
        )
        .drop(columns=["scenario"], errors="ignore")
    )

    ev_by_npi["EV_pct_boost"] = (
        ((ev_by_npi["EV_planned"] - ev_by_npi["EV_base"]) / ev_by_npi["EV_base"])
        .clip(lower=0)
        .round(4) * 100
    )
    ev_by_npi["EV_rank"] = ev_by_npi["EV_planned"].fillna(ev_by_npi["EV_base"])
    ev_by_npi[col_out_npi] = ev_by_npi[col_npi].astype(str)
    output.ev_by_npi = ev_by_npi

    print(f"  EV lookup: {len(ev_by_npi):,} NPIs, avg boost: {ev_by_npi['EV_pct_boost'].mean():.2f}%")

    # ── I3. Build long-format portfolio (one row per team x NPI) ──────
    # THIS WAS MISSING in the original code
    team_a_rows = (
        plan_df[[col_ta, col_npi, "team_a_calls"]]
        .drop_duplicates()
        .rename(columns={col_ta: col_out_terr, "team_a_calls": col_out_calls})
    )
    team_a_rows[col_out_src] = cfg.team_a_label

    team_b_rows = (
        plan_df[[col_tb, col_npi, "team_b_calls"]]
        .drop_duplicates()
        .rename(columns={col_tb: col_out_terr, "team_b_calls": col_out_calls})
    )
    team_b_rows[col_out_src] = cfg.team_b_label

    portfolio = pd.concat([team_a_rows, team_b_rows], ignore_index=True)
    portfolio[col_out_npi] = portfolio[col_npi].astype(str)

    print(
        f"  Long portfolio: {len(portfolio):,} rows "
        f"({portfolio[col_out_npi].nunique():,} NPIs x 2 teams)"
    )

    # ── I3b. Attach metadata ─────────────────────────────────────────
    meta_attach = [col_out_npi]
    if cfg.col_writer_flag in meta.columns:
        meta_attach.append(cfg.col_writer_flag)
    if cfg.col_target_flag in meta.columns:
        meta_attach.append(cfg.col_target_flag)
    for _, tgt in cfg.il_rx_columns:
        if tgt in meta.columns:
            meta_attach.append(tgt)

    portfolio = portfolio.merge(
        meta[meta_attach].drop_duplicates(col_out_npi),
        on=col_out_npi, how="left",
    )

    # Attach EV columns
    portfolio = portfolio.merge(
        ev_by_npi[[col_out_npi, "EV_base", "EV_planned", "EV_pct_boost", "EV_rank"]],
        on=col_out_npi, how="left",
    ).drop_duplicates()

    # Attach priority flag
    pt_merge = (
        plan_df[[col_npi, "PRIORITY_TARGET_FLAG"]]
        .drop_duplicates(col_npi)
        .copy()
    )
    pt_merge[col_out_npi] = pt_merge[col_npi].astype(str)
    pt_merge["PRIORITY_TARGET_FLAG"] = (
        pt_merge["PRIORITY_TARGET_FLAG"].apply(lambda x: "Y" if x == 1 else "N")
    )
    portfolio = portfolio.merge(
        pt_merge[[col_out_npi, "PRIORITY_TARGET_FLAG"]],
        on=col_out_npi, how="left",
    )
    portfolio["PRIORITY_TARGET_FLAG"] = portfolio["PRIORITY_TARGET_FLAG"].fillna("N")

    # ── I4. Lookalike flag ───────────────────────────────────────────
    LOOK_N = cfg.lookalike_top_n

    team_a_look_keys = _compute_lookalike_keys(
        plan_df, col_ta, "team_a_calls", col_npi,
        pt_merge, meta, ev_by_npi, cfg, LOOK_N,
    )
    team_b_look_keys = _compute_lookalike_keys(
        plan_df, col_tb, "team_b_calls", col_npi,
        pt_merge, meta, ev_by_npi, cfg, LOOK_N,
    )

    portfolio["LOOKALIKE_FLAG"] = "N"
    is_a_look = (
        portfolio[col_out_src].eq(cfg.team_a_label)
        & portfolio.apply(
            lambda r: (r[col_out_terr], r[col_out_npi]) in team_a_look_keys, axis=1
        )
    )
    is_b_look = (
        portfolio[col_out_src].eq(cfg.team_b_label)
        & portfolio.apply(
            lambda r: (r[col_out_terr], r[col_out_npi]) in team_b_look_keys, axis=1
        )
    )
    portfolio.loc[is_a_look | is_b_look, "LOOKALIKE_FLAG"] = "Y"
    n_look = portfolio[portfolio["LOOKALIKE_FLAG"] == "Y"][col_out_npi].nunique()
    print(f"  Lookalikes flagged: {n_look} unique NPIs")

    # ── I5. Call increase flag ───────────────────────────────────────
    priority_or_writer = portfolio["PRIORITY_TARGET_FLAG"].eq("Y")
    if cfg.col_writer_flag in portfolio.columns:
        priority_or_writer = priority_or_writer | portfolio[cfg.col_writer_flag].eq("Y")

    call_increase_mask = (
        (portfolio[col_out_calls] >= 1)
        & (~priority_or_writer)
        & (portfolio["LOOKALIKE_FLAG"].ne("Y"))
    )
    portfolio["CALL_INCREASE_FLAG"] = "N"
    portfolio.loc[call_increase_mask, "CALL_INCREASE_FLAG"] = "Y"
    n_ci = portfolio[portfolio["CALL_INCREASE_FLAG"] == "Y"][col_out_npi].nunique()
    print(f"  Call increase flagged: {n_ci} unique NPIs")

    # ── I6. Guard: label flags require calls >= 1 ────────────────────
    label_flags = ["PRIORITY_TARGET_FLAG", "LOOKALIKE_FLAG", "CALL_INCREASE_FLAG"]
    if cfg.col_writer_flag in portfolio.columns:
        label_flags.append(cfg.col_writer_flag)
    if cfg.col_target_flag in portfolio.columns:
        label_flags.append(cfg.col_target_flag)

    no_calls = portfolio[col_out_calls].fillna(0).astype(int) < 1
    for lf in label_flags:
        if lf in portfolio.columns:
            portfolio.loc[no_calls, lf] = "N"

    # ── I7. Composite insight label ──────────────────────────────────
    def _build_insight(row):
        labels = []
        if row.get("PRIORITY_TARGET_FLAG") == "Y":
            labels.append("Priority Target")
        if row.get(cfg.col_writer_flag, "N") == "Y":
            labels.append("Writer")
        if row.get("LOOKALIKE_FLAG") == "Y":
            labels.append("Lookalike")
        if row.get("CALL_INCREASE_FLAG") == "Y":
            labels.append("Call Increase")
        return ";".join(labels) if labels else ""

    portfolio["INSIGHTS"] = portfolio.apply(_build_insight, axis=1)

    # ── I8. Final column selection ───────────────────────────────────
    final_cols = [col_out_terr, col_out_npi, col_out_calls, col_out_src]
    for fc in [cfg.col_writer_flag, "PRIORITY_TARGET_FLAG", cfg.col_target_flag,
               "LOOKALIKE_FLAG", "CALL_INCREASE_FLAG"]:
        if fc in portfolio.columns and fc not in final_cols:
            final_cols.append(fc)
    for _, tgt in cfg.il_rx_columns:
        if tgt in portfolio.columns and tgt not in final_cols:
            final_cols.append(tgt)
    for ec in ["EV_base", "EV_planned", "EV_pct_boost"]:
        if ec in portfolio.columns:
            final_cols.append(ec)
    final_cols.append("INSIGHTS")
    final_cols = [c for c in final_cols if c in portfolio.columns]

    portfolio_final = (
        portfolio[final_cols]
        .sort_values(
            [col_out_src, col_out_terr, col_out_calls],
            ascending=[True, True, False],
        )
        .drop_duplicates()
        .reset_index(drop=True)
    )

    output.portfolio = portfolio_final
    print(f"\n  Final portfolio: {len(portfolio_final):,} rows")
    print(f"  Columns: {list(portfolio_final.columns)}")
    print("  Done.\n")
    return output


def _compute_lookalike_keys(
    plan_df, terr_col, call_col, col_npi,
    pt_merge, meta, ev_by_npi, cfg, top_n,
):
    """Find top-N NPIs by EV per territory, excluding priority/writers."""
    base = (
        plan_df[[terr_col, col_npi, call_col]]
        .merge(pt_merge[[col_npi, "PRIORITY_TARGET_FLAG"]], on=col_npi, how="left")
    )
    if cfg.col_writer_flag in meta.columns:
        base = base.merge(meta[[col_npi, cfg.col_writer_flag]], on=col_npi, how="left")

    base = base.merge(ev_by_npi[[col_npi, "EV_rank"]], on=col_npi, how="left")
    base = base[base[call_col].fillna(0) >= 1].copy()

    excl = base["PRIORITY_TARGET_FLAG"].eq("Y")
    if cfg.col_writer_flag in base.columns:
        excl = excl | base[cfg.col_writer_flag].eq("Y")
    base = base[~excl]

    top_idx = (
        base.sort_values("EV_rank", ascending=False)
        .groupby(terr_col, dropna=False)
        .head(top_n)
    )
    top_idx[cfg.col_output_npi_str] = top_idx[col_npi].astype(str)
    return set(zip(
        top_idx[terr_col].astype(object),
        top_idx[cfg.col_output_npi_str],
    ))


def write_output(
    cfg: DetailOptimizationConfig,
    output: PipelineOutput,
    spark: Any = None,
):
    """Write final portfolio to CSV and/or Spark table."""
    print("=" * 70)
    print("STAGE J: Final output")
    print("=" * 70)

    portfolio = output.portfolio
    col_out_npi = cfg.col_output_npi_str
    col_out_calls = cfg.col_output_calls
    col_out_src = cfg.col_output_source

    deduped = portfolio.drop_duplicates(col_out_npi)
    print(f"\n  Unique NPIs:     {deduped[col_out_npi].nunique():,}")
    print(f"  Total calls:     {deduped[col_out_calls].sum():,}")

    for team in portfolio[col_out_src].unique():
        team_slice = portfolio[portfolio[col_out_src] == team]
        team_dedup = team_slice.drop_duplicates(col_out_npi)
        print(f"\n  {team}:")
        print(f"    NPIs:        {team_dedup[col_out_npi].nunique():,}")
        print(f"    Total calls: {team_dedup[col_out_calls].sum():,}")
        if "EV_planned" in team_slice.columns:
            print(f"    Total EV:    {team_dedup['EV_planned'].sum():,.2f}")

    # Write CSV
    if cfg.output_csv:
        import os
        os.makedirs(os.path.dirname(cfg.output_csv), exist_ok=True)
        portfolio.to_csv(cfg.output_csv, index=False)
        print(f"\n  Wrote CSV: {cfg.output_csv} ({len(portfolio):,} rows)")

    # Write Spark table
    if cfg.output_table and spark is not None:
        try:
            spark_df = spark.createDataFrame(portfolio)
            spark_df.write.mode("overwrite").saveAsTable(cfg.output_table)
            print(f"  Wrote Spark table: {cfg.output_table}")
        except Exception as e:
            print(f"  WARNING: Could not write Spark table: {e}")

    print("\n  Done.")


def plot_diagnostics(
    cfg: DetailOptimizationConfig,
    output: PipelineOutput,
    opt_result: OptimizationResult,
):
    """Plot diagnostic charts for the optimization results."""
    col_out_calls = cfg.col_output_calls
    col_out_terr = cfg.col_output_territory
    col_out_src = cfg.col_output_source
    col_out_npi = cfg.col_output_npi_str

    portfolio = output.portfolio
    plan_df = opt_result.plan_df

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) Calls per Team A territory
    team_a_plan = portfolio[portfolio[col_out_src] == cfg.team_a_label]
    team_a_by_terr = team_a_plan.groupby(col_out_terr)[col_out_calls].sum()
    axes[0, 0].hist(team_a_by_terr.values, bins=20, edgecolor="black", alpha=0.7)
    axes[0, 0].set_xlabel("Calls per territory")
    axes[0, 0].set_ylabel("# Territories")
    axes[0, 0].set_title(f"{cfg.team_a_label}: Calls per Territory")
    axes[0, 0].axvline(
        cfg.team_a_budget_per_territory, color="red", ls="--",
        label=f"Budget={cfg.team_a_budget_per_territory}"
    )
    axes[0, 0].legend()

    # (b) Calls per Team B territory
    team_b_plan = portfolio[portfolio[col_out_src] == cfg.team_b_label]
    team_b_by_terr = team_b_plan.groupby(col_out_terr)[col_out_calls].sum()
    axes[0, 1].hist(team_b_by_terr.values, bins=20, edgecolor="black", alpha=0.7)
    axes[0, 1].set_xlabel("Calls per territory")
    axes[0, 1].set_ylabel("# Territories")
    axes[0, 1].set_title(f"{cfg.team_b_label}: Calls per Territory")
    axes[0, 1].axvline(
        cfg.team_b_target_per_territory, color="red", ls="--",
        label=f"Target={cfg.team_b_target_per_territory}"
    )
    axes[0, 1].legend()

    # (c) Call pair distribution
    col_npi = cfg.col_npi
    pair_counts = (
        plan_df.drop_duplicates(col_npi)
        .groupby(["team_a_calls", "team_b_calls"])
        .size()
        .reset_index(name="count")
    )
    pair_labels = [f"({r['team_a_calls']},{r['team_b_calls']})" for _, r in pair_counts.iterrows()]
    axes[1, 0].bar(pair_labels, pair_counts["count"], edgecolor="black", alpha=0.7)
    axes[1, 0].set_xlabel("(Team A, Team B) calls")
    axes[1, 0].set_ylabel("# NPIs")
    axes[1, 0].set_title("Call Pair Distribution")
    axes[1, 0].tick_params(axis="x", rotation=45)

    # (d) EV distribution
    if "EV_planned" in portfolio.columns:
        ev_vals = portfolio.drop_duplicates(col_out_npi)["EV_planned"].dropna()
        axes[1, 1].hist(ev_vals.values, bins=40, edgecolor="black", alpha=0.7)
        axes[1, 1].set_xlabel("EV (planned)")
        axes[1, 1].set_ylabel("# NPIs")
        axes[1, 1].set_title("Expected Value Distribution")

    plt.tight_layout()
    plt.show()
