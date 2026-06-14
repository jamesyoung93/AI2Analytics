"""Generalized detail optimization pipeline."""

from __future__ import annotations

from typing import Any

from ai2analytics.templates.base import BaseTemplate, ColumnRequirement, TableRequirement
from ai2analytics.templates.detail_optimization.features import FeatureSet, engineer_features
from ai2analytics.templates.detail_optimization.output import (
    PipelineOutput,
    plot_diagnostics,
    post_process,
    write_output,
)
from ai2analytics.templates.generalized.config import GeneralizedDetailOptimizationConfig
from ai2analytics.templates.generalized.loader import GeneralizedLoadedData, load_data
from ai2analytics.templates.generalized.models import (
    TrainedModels,
    print_feature_importance,
    train_models,
)
from ai2analytics.templates.generalized.optimizer import OptimizationResult, prep_and_optimize
from ai2analytics.templates.generalized.reporting import GeneralizedReporter
from ai2analytics.templates.generalized.scoring import score_scenarios
from ai2analytics.templates.registry import register


@register
class GeneralizedDetailOptimizationPipeline(BaseTemplate):
    """Generalized HCP call allocation pipeline.

    This is a second version of the detail optimization workflow. It preserves
    the existing two-field-force alignment design while adding explicit defaults
    for missing priority input, cadence-aware planning, optional capacity files,
    and verbose assumption reporting.
    """

    name = "generalized"
    description = (
        "Generalized HCP call allocation optimization with two field-force "
        "alignments, optional binary priority targets, configurable cadence, "
        "and optional capacity inputs"
    )
    config_class = GeneralizedDetailOptimizationConfig

    required_tables = [
        TableRequirement(
            key="hcp_weekly",
            description="One row per HCP x planning period with referral counts and features",
            source_type="spark_table",
            config_field="hcp_weekly_table",
            required_columns=[
                ColumnRequirement(
                    "npi",
                    "int",
                    "HCP National Provider Identifier",
                    aliases=["NPI", "npi_number"],
                    config_field="col_npi",
                ),
                ColumnRequirement(
                    "period",
                    "date",
                    "Planning period date",
                    aliases=["WEEK_ENDING", "week_end", "month_end", "period_end"],
                    config_field="col_week",
                ),
                ColumnRequirement(
                    "referrals",
                    "numeric",
                    "Patient referral count or outcome count",
                    aliases=["PAT_COUNT_REFERRED", "referral_count"],
                    config_field="col_referrals",
                ),
            ],
            optional_columns=[
                ColumnRequirement(
                    "indication",
                    "string",
                    "Indication or diagnosis",
                    aliases=["INDC", "indication_code"],
                    config_field="col_indication",
                ),
            ],
        ),
        TableRequirement(
            key="calls",
            description="HCP-level detail or call activity by planning period",
            source_type="spark_table",
            config_field="calls_table",
            required_columns=[
                ColumnRequirement("npi", "int", "HCP NPI", config_field="col_npi"),
                ColumnRequirement(
                    "period",
                    "date",
                    "Planning period date",
                    config_field="col_week",
                ),
                ColumnRequirement(
                    "calls",
                    "numeric",
                    "Face-to-face call count",
                    aliases=["HCP_F2F_CALLS", "CALLS"],
                    config_field="col_calls",
                ),
            ],
        ),
        TableRequirement(
            key="team_a_alignment",
            description="NPI to first field-force territory mapping",
            source_type="csv",
            config_field="team_a_align_path",
            required_columns=[
                ColumnRequirement(
                    "npi",
                    "int",
                    "HCP NPI",
                    aliases=["HCP_NPI"],
                    config_field="team_a_npi_col",
                ),
                ColumnRequirement(
                    "territory_id",
                    "string",
                    "Territory identifier",
                    aliases=["TERRITORY_ID"],
                    config_field="team_a_territory_col",
                ),
            ],
        ),
        TableRequirement(
            key="team_b_alignment",
            description="NPI to second field-force territory mapping",
            source_type="csv",
            config_field="team_b_align_path",
            required_columns=[
                ColumnRequirement(
                    "npi",
                    "int",
                    "HCP NPI",
                    aliases=["HCP_NPI"],
                    config_field="team_b_npi_col",
                ),
                ColumnRequirement(
                    "territory_id",
                    "string",
                    "Territory identifier",
                    aliases=["TERRITORY_ID"],
                    config_field="team_b_territory_col",
                ),
            ],
        ),
        TableRequirement(
            key="hcp_reference",
            description="HCP reference table with writer flags and optional Rx fields",
            source_type="csv",
            config_field="hcp_reference_path",
            required_columns=[
                ColumnRequirement("npi", "int", "HCP NPI", config_field="col_npi"),
            ],
            optional_columns=[
                ColumnRequirement(
                    "writer_flag",
                    "string",
                    "Whether HCP has written the drug",
                    aliases=["WRITER_FLAG"],
                    config_field="col_writer_flag",
                ),
                ColumnRequirement(
                    "target_flag",
                    "string",
                    "Whether HCP is a target",
                    aliases=["TARGET_FLAG"],
                    config_field="col_target_flag",
                ),
            ],
        ),
    ]

    def __init__(self):
        self._data: GeneralizedLoadedData | None = None
        self._features: FeatureSet | None = None
        self._models: TrainedModels | None = None
        self._scenarios: Any = None
        self._opt_result: OptimizationResult | None = None
        self._output: PipelineOutput | None = None
        self._reporter: GeneralizedReporter | None = None

    def run(
        self,
        cfg: GeneralizedDetailOptimizationConfig,
        spark: Any = None,
        dataframes: dict[str, Any] | None = None,
    ) -> PipelineOutput:
        """Run the generalized pipeline end-to-end."""
        cfg = cfg.normalize_for_run()
        self._reporter = GeneralizedReporter(
            verbose=cfg.verbose,
            warn_on_defaults=cfg.warn_on_defaults,
        )

        print("\n" + "=" * 70)
        print(f"  PIPELINE: {cfg.drug_name} Generalized Detail Optimization")
        print("=" * 70 + "\n")

        self._reporter.progress(
            f"Using planning_cadence={cfg.planning_cadence}, "
            f"horizon={cfg.effective_horizon_periods()} period(s)."
        )
        for message in cfg.assumption_messages(dataframes=dataframes):
            self._reporter.warn(message)

        errors = cfg.validate(dataframes=dataframes)
        if errors:
            raise ValueError("Config validation failed:\n  " + "\n  ".join(errors))

        self._data = load_data(cfg, spark=spark, dataframes=dataframes, reporter=self._reporter)
        self._features = engineer_features(cfg, self._data)
        self._models = train_models(cfg, self._features, reporter=self._reporter)
        self._scenarios = score_scenarios(
            cfg,
            self._features,
            self._models,
            reporter=self._reporter,
        )
        self._opt_result = prep_and_optimize(
            cfg,
            self._scenarios,
            self._data.team_a_align,
            self._data.team_b_align,
            self._data.portfolio_decile,
            self._data.priority_targets,
            team_a_capacity=self._data.team_a_capacity,
            team_b_capacity=self._data.team_b_capacity,
            reporter=self._reporter,
        )
        self._output = post_process(
            cfg,
            self._opt_result,
            self._data.hcp_reference,
            self._data.priority_targets,
        )
        write_output(cfg, self._output, spark=spark)
        print_feature_importance(self._models, self._features)

        self._print_summary(cfg)
        self._reporter.print_summary()
        return self._output

    def show_dashboard(self, output: PipelineOutput | None = None):
        out = output or self._output
        if out is None or self._opt_result is None:
            print("Run the pipeline first.")
            return
        print("Use plot_diagnostics(cfg, output, opt_result) for charts.")

    def _print_summary(self, cfg: GeneralizedDetailOptimizationConfig) -> None:
        print("\n" + "=" * 70)
        print("GENERALIZED PIPELINE COMPLETE")
        print("=" * 70)
        print(f"  Drug:             {cfg.drug_name}")
        if cfg.drug_portfolio:
            print(f"  Portfolio drug:   {cfg.drug_portfolio}")
        print(f"  Cadence:          {cfg.planning_cadence}")
        print(f"  Horizon periods:  {cfg.effective_horizon_periods()}")
        if self._output is not None:
            col_npi = cfg.col_output_npi_str
            pf = self._output.portfolio
            print(f"  NPIs in plan:     {pf[col_npi].nunique():,}")
            print(f"  Total rows:       {len(pf):,}")
        if cfg.output_csv:
            print(f"  Output CSV:       {cfg.output_csv}")
        if cfg.output_table:
            print(f"  Output table:     {cfg.output_table}")
        print("=" * 70)


__all__ = [
    "GeneralizedDetailOptimizationConfig",
    "GeneralizedDetailOptimizationPipeline",
    "plot_diagnostics",
]
