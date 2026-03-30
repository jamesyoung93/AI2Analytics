"""Code generation for data adapters — transforms source data to template format."""

from __future__ import annotations

from ai2analytics.llm import LLMClient
from ai2analytics.discovery.surveyor import DataSurvey, profile_for_llm
from ai2analytics.templates.base import BaseTemplate


def generate_preprocessing_code(
    llm: LLMClient,
    survey: DataSurvey,
    template: type[BaseTemplate],
    config_dict: dict,
    brand_context: str = "",
) -> str:
    """Generate preprocessing code to adapt source data to template requirements.

    This handles the "last mile" problem: when a new brand/region has slightly
    different data than what the template expects, this generates the PySpark
    or Pandas code to bridge the gap.

    Args:
        llm: LLM client for code generation.
        survey: Data survey results.
        template: Target pipeline template.
        config_dict: Current config values.
        brand_context: Additional context about the brand/region differences.
    """
    schema_summary = template.get_schema_summary()
    data_summary = profile_for_llm(survey)

    system = (
        "You are a pharmaceutical data engineering expert working in Databricks. "
        "Generate clean, production-ready Python/PySpark code that transforms "
        "source data tables into the exact format required by a pipeline template.\n\n"
        "Rules:\n"
        "- Use PySpark for large table operations, Pandas for small reference files\n"
        "- Handle column renaming, type casting, and null filling\n"
        "- Add validation checks after each transformation\n"
        "- Write intermediate results to temp views or DBFS paths\n"
        "- Include clear comments explaining each transformation\n"
        "- If a required column doesn't exist, derive it if possible or flag it\n\n"
        "Output ONLY executable Python code."
    )

    user = (
        f"PIPELINE TEMPLATE REQUIREMENTS:\n{schema_summary}\n\n"
        f"AVAILABLE SOURCE DATA:\n{data_summary}\n\n"
        f"CURRENT CONFIG:\n{config_dict}\n\n"
    )

    if brand_context:
        user += f"BRAND-SPECIFIC CONTEXT:\n{brand_context}\n\n"

    user += (
        "Generate preprocessing code that transforms the source data to match "
        "the template requirements. Handle any column name differences, data type "
        "mismatches, or structural transformations needed."
    )

    from ai2analytics.llm import strip_markdown_fences
    raw = llm.call(system, user, temperature=0.1)
    return strip_markdown_fences(raw)
