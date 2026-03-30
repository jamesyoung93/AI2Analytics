"""Pipeline orchestrator for detail optimization — runs all stages end-to-end."""

from __future__ import annotations

from typing import Any

from ai2analytics.templates.base import BaseTemplate, TableRequirement, ColumnRequirement
from ai2analytics.templates.registry import register
from ai2analytics.templates.detail_optimization.config import DetailOptimizationConfig
from ai2analytics.templates.detail_optimization.loader import LoadedData, load_data
from ai2analytics.templates.detail_optimization.features import FeatureSet, engineer_features
from ai2analytics.templates.detail_optimization.models import (
    TrainedModels, train_models, print_feature_importance,
)
from ai2analytics.templates.detail_optimization.scoring import score_scenarios
from ai2analytics.templates.detail_optimization.optimizer import OptimizationResult, prep_and_optimize
from ai2analytics.templates.detail_optimization.output import (
    PipelineOutput, post_process, write_output, plot_diagnostics,
)


@register
class DetailOptimizationPipeline(BaseTemplate):
    """HCP call allocation optimization pipeline.

    Stages:
        C. Data Loading
        D. Feature Engineering
        E. Model Training & Backtesting
        F. Scenario Scoring
        G-H. Prep & Optimization (PuLP LP)
        I-J. Post-Processing & Output

    Usage in Databricks notebook::

        from ai2analytics.templates.detail_optimization import (
            DetailOptimizationConfig, DetailOptimizationPipeline,
        )

        cfg = DetailOptimizationConfig(
            drug_name="DRUG_X",
            hcp_weekly_table="schema.hcp_weekly",
            calls_table="schema.calls",
            team_a_align_path="/dbfs/mnt/data/team_a.csv",
            team_b_align_path="/dbfs/mnt/data/team_b.csv",
            hcp_reference_path="/dbfs/mnt/data/hcp_ref.csv",
            output_csv="/dbfs/mnt/output/plan.csv",
        )

        pipeline = DetailOptimizationPipeline()
        results = pipeline.run(cfg, spark=spark)
        pipeline.show_dashboard(results)
    """

    name = "detail_optimization"
    description = (
        "HCP call allocation optimization using probability depth and "
        "look-alike models with PuLP LP territory-level budgets"
    )
    config_class = DetailOptimizationConfig

    required_tables = [
        TableRequirement(
            key="hcp_weekly",
            description="One row per HCP x week with referral counts and features",
            source_type="spark_table",
            config_field="hcp_weekly_table",
            required_columns=[
                ColumnRequirement("npi", "int", "HCP National Provider Identifier",
                                  aliases=["NPI", "npi_number"],
                                  config_field="col_npi"),
                ColumnRequirement("week_ending", "date", "Week ending date",
                                  aliases=["WEEK_ENDING", "week_end"],
                                  config_field="col_week"),
                ColumnRequirement("referrals", "numeric", "Patient referral count",
                                  aliases=["PAT_COUNT_REFERRED", "referral_count"],
                                  config_field="col_referrals"),
            ],
            optional_columns=[
                ColumnRequirement("indication", "string", "Indication/diagnosis",
                                  aliases=["INDC", "indication_code"],
                                  config_field="col_indication"),
            ],
        ),
        TableRequirement(
            key="calls",
            description="HCP-level detail/call activity per week",
            source_type="spark_table",
            config_field="calls_table",
            required_columns=[
                ColumnRequirement("npi", "int", "HCP NPI",
                                  config_field="col_npi"),
                ColumnRequirement("week_ending", "date", "Week ending date",
                                  config_field="col_week"),
                ColumnRequirement("calls", "numeric", "Face-to-face call count",
                                  aliases=["HCP_F2F_CALLS", "CALLS"],
                                  config_field="col_calls"),
            ],
        ),
        TableRequirement(
            key="team_a_alignment",
            description="NPI to Team A territory mapping",
            source_type="csv",
            config_field="team_a_align_path",
            required_columns=[
                ColumnRequirement("npi", "int", "HCP NPI",
                                  aliases=["HCP_NPI"],
                                  config_field="team_a_npi_col"),
                ColumnRequirement("territory_id", "int", "Territory identifier",
                                  aliases=["TERRITORY_ID"],
                                  config_field="team_a_territory_col"),
            ],
        ),
        TableRequirement(
            key="team_b_alignment",
            description="NPI to Team B territory mapping",
            source_type="csv",
            config_field="team_b_align_path",
            required_columns=[
                ColumnRequirement("npi", "int", "HCP NPI",
                                  aliases=["HCP_NPI"],
                                  config_field="team_b_npi_col"),
                ColumnRequirement("territory_id", "int", "Territory identifier",
                                  aliases=["TERRITORY_ID"],
                                  config_field="team_b_territory_col"),
            ],
        ),
        TableRequirement(
            key="hcp_reference",
            description="HCP reference table with writer flags and Rx counts",
            source_type="csv",
            config_field="hcp_reference_path",
            required_columns=[
                ColumnRequirement("npi", "int", "HCP NPI",
                                  config_field="col_npi"),
            ],
            optional_columns=[
                ColumnRequirement("writer_flag", "string", "Whether HCP has written the drug",
                                  aliases=["WRITER_FLAG"],
                                  config_field="col_writer_flag"),
                ColumnRequirement("target_flag", "string", "Whether HCP is a target",
                                  aliases=["TARGET_FLAG"],
                                  config_field="col_target_flag"),
            ],
        ),
        TableRequirement(
            key="portfolio_decile",
            description="Portfolio drug decile per NPI",
            source_type="csv",
            config_field="portfolio_decile_path",
            required_columns=[
                ColumnRequirement("npi", "int", "HCP NPI",
                                  config_field="col_npi"),
                ColumnRequirement("decile", "int", "Portfolio units decile",
                                  aliases=["PORTFOLIO_UNITS_DECILE"],
                                  config_field="col_portfolio_decile"),
            ],
        ),
        TableRequirement(
            key="priority_targets",
            description="Priority target NPI list",
            source_type="csv",
            config_field="priority_target_path",
            required_columns=[
                ColumnRequirement("npi", "int", "HCP NPI",
                                  aliases=["npi_number", "NPI"],
                                  config_field="col_npi"),
            ],
            optional_columns=[
                ColumnRequirement("priority_flag", "string", "Priority target flag",
                                  aliases=["PRIORITY_TARGET", "PRIORITY_TARGET_FLAG"],
                                  config_field="col_priority_flag"),
            ],
        ),
    ]

    def __init__(self):
        self._data: LoadedData | None = None
        self._features: FeatureSet | None = None
        self._models: TrainedModels | None = None
        self._scenarios: Any = None
        self._opt_result: OptimizationResult | None = None
        self._output: PipelineOutput | None = None

    def run(
        self,
        cfg: DetailOptimizationConfig,
        spark: Any = None,
    ) -> PipelineOutput:
        """Run the full pipeline end-to-end."""
        errors = cfg.validate()
        if errors:
            raise ValueError(f"Config validation failed:\n  " + "\n  ".join(errors))

        print("\n" + "=" * 70)
        print(f"  PIPELINE: {cfg.drug_name} Detail Optimization")
        print("=" * 70 + "\n")

        # C. Load data
        self._data = load_data(cfg, spark=spark)

        # D. Feature engineering
        self._features = engineer_features(cfg, self._data)

        # E. Model training
        self._models = train_models(cfg, self._features)

        # F. Scenario scoring
        self._scenarios = score_scenarios(cfg, self._features, self._models)

        # G-H. Prep and optimize
        self._opt_result = prep_and_optimize(
            cfg, self._scenarios,
            self._data.team_a_align,
            self._data.team_b_align,
            self._data.portfolio_decile,
            self._data.priority_targets,
        )

        # I. Post-processing
        self._output = post_process(
            cfg, self._opt_result,
            self._data.hcp_reference,
            self._data.priority_targets,
        )

        # J. Write output
        write_output(cfg, self._output, spark=spark)

        # Feature importance
        print_feature_importance(self._models, self._features)

        self._print_summary(cfg)
        return self._output

    def show_dashboard(self, output: PipelineOutput | None = None):
        """Show diagnostic plots."""
        out = output or self._output
        if out is None or self._opt_result is None:
            print("Run the pipeline first.")
            return
        # We need cfg stored — caller should pass it or we store it
        # For now, just call plot_diagnostics if we have the data
        print("Use plot_diagnostics(cfg, output, opt_result) for charts.")

    def _print_summary(self, cfg: DetailOptimizationConfig):
        """Print final pipeline summary."""
        print("\n" + "=" * 70)
        print("PIPELINE COMPLETE")
        print("=" * 70)
        print(f"  Drug:           {cfg.drug_name}")
        print(f"  Portfolio drug: {cfg.drug_portfolio}")
        if self._output is not None:
            col_npi = cfg.col_output_npi_str
            pf = self._output.portfolio
            print(f"  NPIs in plan:   {pf[col_npi].nunique():,}")
            print(f"  Total rows:     {len(pf):,}")
        if cfg.output_csv:
            print(f"  Output CSV:     {cfg.output_csv}")
        if cfg.output_table:
            print(f"  Output table:   {cfg.output_table}")
        print("=" * 70)

    # Model performance summary
    def print_model_summary(self):
        if self._models is None:
            print("No models trained yet.")
            return
        print("\n" + "=" * 70)
        print("MODEL PERFORMANCE SUMMARY")
        print("=" * 70)
        m = self._models
        if not m.metrics_prob.empty:
            print("\n  Probability Model:")
            print(f"    Avg PR-AUC:    {m.metrics_prob['pr_auc'].mean():.4f}")
            print(f"    Avg ROC-AUC:   {m.metrics_prob['roc_auc'].mean():.4f}")
            print(f"    Avg Recall:    {m.metrics_prob['recall'].mean():.4f}")
            print(f"    Avg Precision: {m.metrics_prob['precision'].mean():.4f}")
        if not m.metrics_depth.empty:
            print("\n  Depth Model:")
            print(f"    Avg MAE: {m.metrics_depth['MAE'].mean():.4f}")
            print(f"    Avg R2:  {m.metrics_depth['R2'].mean():.4f}")
        if not m.metrics_look.empty:
            print("\n  Look-alike Model:")
            print(f"    Avg PR-AUC:    {m.metrics_look['pr_auc'].mean():.4f}")
            print(f"    Avg ROC-AUC:   {m.metrics_look['roc_auc'].mean():.4f}")
