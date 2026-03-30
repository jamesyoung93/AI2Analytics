"""Main session orchestrator — the primary user-facing API."""

from __future__ import annotations

from typing import Any

from ai2analytics.llm import LLMClient
from ai2analytics.discovery.surveyor import DataSurvey, survey_tables, profile_for_llm
from ai2analytics.discovery.profiler import deep_profile, format_deep_profile
from ai2analytics.conversation.manager import ConversationManager, ConversationState
from ai2analytics.templates.base import BaseTemplate
from ai2analytics.templates.registry import list_templates, get_template, find_template


class AnalyticsSession:
    """AI-powered analytics session for configuring and running pipeline templates.

    This is the main entry point for users. It orchestrates:
    1. Data discovery — survey what tables exist
    2. Template matching — which pipeline template fits
    3. Conversation — structured Q&A to fill config gaps
    4. Code generation — adapter code for data mismatches
    5. Pipeline execution — run the configured pipeline

    Usage in Databricks notebook::

        from ai2analytics import AnalyticsSession

        # Cell 1: Initialize
        session = AnalyticsSession(
            spark=spark,
            llm_endpoint="databricks-qwen3-next-80b-a3b-instruct",
        )

        # Cell 2: Discover data
        session.discover(
            schemas=["pharma_data"],
            prompt="Optimize HCP call allocation for DRUG_X",
        )

        # Cell 3: Review questions and answer
        session.show_questions()
        session.answer({
            "drug_name": "DRUG_X",
            "output_csv": "/dbfs/mnt/output/drug_x_plan.csv",
        })

        # Cell 4: Run
        results = session.run()
    """

    def __init__(
        self,
        spark: Any = None,
        llm_endpoint: str = "databricks-qwen3-next-80b-a3b-instruct",
        llm_client: Any = None,
    ):
        self.spark = spark
        self.llm = LLMClient(
            endpoint=llm_endpoint,
            openai_client=llm_client,
        )
        self.conversation = ConversationManager(self.llm)

        self._survey: DataSurvey | None = None
        self._template: type[BaseTemplate] | None = None
        self._state: ConversationState | None = None
        self._config: Any = None
        self._pipeline: BaseTemplate | None = None
        self._results: Any = None

    def discover(
        self,
        schemas: list[str],
        prompt: str,
        catalog: str = "hive_metastore",
        template_name: str | None = None,
    ) -> DataSurvey:
        """Survey available data and match to a template.

        Args:
            schemas: Schema names to scan for tables.
            prompt: User's description of what they want to do.
            catalog: Spark catalog name.
            template_name: Explicit template name, or auto-detect from prompt.
        """
        print("=" * 70)
        print("DATA DISCOVERY")
        print("=" * 70)

        # Import templates to trigger registration
        import ai2analytics.templates.detail_optimization  # noqa: F401

        # Survey tables
        print("\nScanning tables...")
        self._survey = survey_tables(self.spark, schemas, catalog)
        print(f"\n{self._survey.summary}\n")

        # Match template
        if template_name:
            self._template = get_template(template_name)
        else:
            matches = find_template(prompt)
            if matches:
                self._template = matches[0][1]
                print(f"Best template match: {matches[0][0]}")
            else:
                available = list_templates()
                print("Available templates:")
                for name, desc in available.items():
                    print(f"  {name}: {desc}")
                print("\nCall session.select_template('name') to choose.")
                return self._survey

        print(f"\nUsing template: {self._template.name}")
        print(self._template.get_schema_summary())

        # Analyze fit
        print("\nAnalyzing data fit...")
        self._state = self.conversation.analyze_fit(
            self._survey, self._template, prompt
        )
        print(self.conversation.present_questions(self._state))

        return self._survey

    def select_template(self, name: str):
        """Manually select a template by name."""
        import ai2analytics.templates.detail_optimization  # noqa: F401
        self._template = get_template(name)
        print(f"Selected template: {self._template.name}")
        print(self._template.get_schema_summary())

    def show_questions(self):
        """Display current configuration questions."""
        if self._state is None:
            print("Run discover() first.")
            return
        print(self.conversation.present_questions(self._state))

    def answer(self, answers: dict[str, Any]):
        """Provide answers to configuration questions."""
        if self._state is None:
            print("Run discover() first.")
            return
        self._state = self.conversation.apply_answers(self._state, answers)
        if self._state.is_complete:
            print("All required questions answered. Ready to run.")
            print("Call session.run() to execute the pipeline.")
        else:
            self.show_questions()

    def generate_adapter(self) -> str:
        """Generate adapter code if data needs transformation."""
        if self._state is None or self._survey is None or self._template is None:
            print("Run discover() first.")
            return ""
        code = self.conversation.generate_adapter(
            self._state, self._survey, self._template
        )
        print("Generated adapter code:")
        print(code)
        return code

    def profile_table(self, table_name: str, **kwargs) -> str:
        """Run a deep profile on a specific table."""
        profile = deep_profile(self.spark, table_name, **kwargs)
        result = format_deep_profile(profile)
        print(result)
        return result

    def build_config(self) -> Any:
        """Build the configuration object from collected answers."""
        if self._template is None or self._state is None:
            print("Run discover() and answer questions first.")
            return None

        config_class = self._template.config_class
        if config_class is None:
            print("Template does not define a config class.")
            return None

        if hasattr(config_class, "from_dict"):
            self._config = config_class.from_dict(self._state.config_dict)
        else:
            self._config = config_class(**self._state.config_dict)

        errors = self._config.validate() if hasattr(self._config, "validate") else []
        if errors:
            print("Config validation issues:")
            for e in errors:
                print(f"  - {e}")
            print("\nUpdate with session.answer({...}) and rebuild.")
        else:
            print("Config is valid.")

        return self._config

    def run(self, config: Any = None) -> Any:
        """Run the pipeline with the current or provided config.

        Args:
            config: Optional config object. If not provided, builds from answers.
        """
        if config is None:
            config = self.build_config()
        if config is None:
            return None

        if self._template is None:
            print("No template selected.")
            return None

        self._pipeline = self._template()
        self._results = self._pipeline.run(config, spark=self.spark)
        return self._results

    def run_direct(self, config: Any) -> Any:
        """Run a pipeline directly with a pre-built config, skipping discovery.

        Usage::

            from ai2analytics.templates.detail_optimization import (
                DetailOptimizationConfig, DetailOptimizationPipeline,
            )

            cfg = DetailOptimizationConfig(drug_name="X", ...)
            pipeline = DetailOptimizationPipeline()
            results = pipeline.run(cfg, spark=spark)
        """
        if self._template is None:
            # Try to infer template from config type
            import ai2analytics.templates.detail_optimization as do
            if isinstance(config, do.DetailOptimizationConfig):
                self._template = do.DetailOptimizationPipeline

        if self._template is None:
            print("Could not determine template. Use session.select_template() first.")
            return None

        return self.run(config)
