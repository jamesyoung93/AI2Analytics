"""Main session orchestrator — the primary user-facing API."""

from __future__ import annotations

import traceback
from typing import Any

from ai2analytics.llm import LLMClient, strip_markdown_fences
from ai2analytics.discovery.surveyor import DataSurvey, survey_tables, profile_for_llm
from ai2analytics.discovery.profiler import deep_profile, format_deep_profile
from ai2analytics.conversation.manager import ConversationManager, ConversationState
from ai2analytics.templates.base import BaseTemplate
from ai2analytics.templates.registry import list_templates, get_template, find_template


class AnalyticsSession:
    """AI-powered analytics session for configuring and running pipeline templates.

    This is the main entry point. It orchestrates:
    1. Data discovery — survey what tables exist
    2. Template matching — which pipeline template fits
    3. Conversation — structured Q&A to fill config gaps
    4. Adapter execution — transform data to fit template requirements
    5. Pipeline execution — run the configured pipeline

    Usage in Databricks notebook::

        from ai2analytics import AnalyticsSession

        session = AnalyticsSession(spark=spark, llm_endpoint="your-endpoint")

        # Discover data and get questions
        session.discover(schemas=["my_schema"], prompt="Optimize call allocation")

        # Answer questions
        session.answer({"drug_name": "X", "hcp_weekly_table": "schema.table", ...})

        # If data needs transformation
        session.generate_adapter()
        session.run_adapter()

        # Run the pipeline
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

    # ── Discovery ───────────────────────────────────────────────────────

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
            prompt: What you want to do (e.g. "optimize call allocation for Brand X").
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

        # Analyze fit — LLM maps data to config fields
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

    def profile_table(self, table_name: str, **kwargs) -> str:
        """Run a deep profile on a specific table."""
        profile = deep_profile(self.spark, table_name, **kwargs)
        result = format_deep_profile(profile)
        print(result)
        return result

    # ── Configuration ───────────────────────────────────────────────────

    def show_questions(self):
        """Display current configuration questions."""
        if self._state is None:
            print("Run discover() first.")
            return
        print(self.conversation.present_questions(self._state))

    def answer(self, answers: dict[str, Any]):
        """Provide answers to configuration questions.

        Args:
            answers: Dict mapping config field names to values.
                     e.g. {"drug_name": "BRAND_X", "hcp_weekly_table": "schema.tbl"}
        """
        if self._state is None:
            print("Run discover() first.")
            return
        self._state = self.conversation.apply_answers(self._state, answers)
        if self._state.is_complete:
            print("Ready to run. Call session.run() or session.generate_adapter() first.")

    def show_config(self):
        """Show the current config state."""
        if self._state is None:
            print("Run discover() first.")
            return
        print("Current config values:")
        for k, v in sorted(self._state.config_dict.items()):
            print(f"  {k}: {repr(v)}")

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
            print("\nCall session.answer({...}) to fix, then session.build_config() again.")
        else:
            print("Config is valid.")

        return self._config

    # ── Adapter code ────────────────────────────────────────────────────

    def generate_adapter(self) -> str:
        """Generate adapter/preprocessing code if data needs transformation.

        Returns the code string. Call run_adapter() to execute it.
        """
        if self._state is None or self._survey is None or self._template is None:
            print("Run discover() first.")
            return ""
        code = self.conversation.generate_adapter(
            self._state, self._survey, self._template
        )
        print("=" * 70)
        print("GENERATED ADAPTER CODE")
        print("=" * 70)
        print(code)
        print("=" * 70)
        print("\nReview the code above. Call session.run_adapter() to execute it.")
        print("Or edit with: session.set_adapter_code('your code')")
        return code

    def set_adapter_code(self, code: str):
        """Manually set or edit the adapter code."""
        if self._state is None:
            print("Run discover() first.")
            return
        self._state.adapter_code = code
        print(f"Adapter code set ({len(code)} chars).")

    def run_adapter(self, code: str | None = None, max_retries: int = 2) -> bool:
        """Execute adapter code with spark in scope.

        If execution fails, the LLM will attempt to fix the code and retry.

        Args:
            code: Code string to execute. If None, uses previously generated code.
            max_retries: Number of LLM-assisted fix attempts on failure.
        """
        if code is None:
            if self._state and self._state.adapter_code:
                code = self._state.adapter_code
            else:
                print("No adapter code. Call generate_adapter() first.")
                return False

        # Validate syntax first
        from ai2analytics.codegen.adapter import validate_adapter_code
        warnings = validate_adapter_code(code)
        if warnings:
            print("Code warnings:")
            for w in warnings:
                print(f"  - {w}")

        # Build execution namespace
        namespace = {"spark": self.spark, "__builtins__": __builtins__}

        for attempt in range(1 + max_retries):
            try:
                compiled = compile(code, "<adapter>", "exec")
                exec(compiled, namespace)
                print("Adapter executed successfully.")
                # Store any new dataframes/views created
                if self._state:
                    self._state.adapter_code = code
                return True
            except Exception as e:
                error_msg = traceback.format_exc()
                if attempt < max_retries:
                    print(f"  Attempt {attempt + 1} failed: {e}")
                    print("  Asking LLM to fix...")
                    code = self._fix_adapter_code(code, error_msg)
                    print("  Retrying with corrected code...")
                else:
                    print(f"  Adapter failed after {1 + max_retries} attempts:")
                    print(f"  {error_msg}")
                    print("\n  Edit manually with session.set_adapter_code() and retry.")
                    return False

    def _fix_adapter_code(self, code: str, error: str) -> str:
        """Ask the LLM to fix adapter code given an error."""
        system = (
            "You are a data engineering expert. The following Python/PySpark code "
            "failed with an error. Fix the code and return ONLY the corrected "
            "executable Python code. Do not explain — just output the fixed code."
        )
        user = (
            f"CODE:\n```python\n{code}\n```\n\n"
            f"ERROR:\n{error}\n\n"
            "Fix the error and return the corrected code."
        )
        raw = self.llm.call(system, user, temperature=0.1)
        fixed = strip_markdown_fences(raw)
        if self._state:
            self._state.adapter_code = fixed
        return fixed

    # ── Pipeline execution ──────────────────────────────────────────────

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
            import ai2analytics.templates.detail_optimization as do
            if isinstance(config, do.DetailOptimizationConfig):
                self._template = do.DetailOptimizationPipeline

        if self._template is None:
            print("Could not determine template. Use session.select_template() first.")
            return None

        return self.run(config)
