"""Code generation for data adapters — transforms source data to template format."""

from __future__ import annotations

import ast

from ai2analytics.llm import LLMClient, strip_markdown_fences
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

    Handles the 'last mile' problem: when a new brand/region has slightly
    different data than what the template expects, this generates PySpark
    or Pandas code to bridge the gap.

    Args:
        llm: LLM client for code generation.
        survey: Data survey results.
        template: Target pipeline template.
        config_dict: Current config values.
        brand_context: Additional context about brand/region differences.
    """
    schema_summary = template.get_schema_summary()
    config_summary = template.get_config_summary()
    data_summary = profile_for_llm(survey)

    system = (
        "You are a data engineering expert working in Databricks. "
        "Generate clean, production-ready Python/PySpark code that transforms "
        "source data tables into the exact format required by a pipeline template.\n\n"
        "Rules:\n"
        "- Use PySpark for large table operations, Pandas for small reference files\n"
        "- Handle column renaming, type casting, and null filling\n"
        "- Add validation checks (row counts, null checks) after each step\n"
        "- Write intermediate results to temp views or DBFS paths\n"
        "- Include clear comments explaining each transformation\n"
        "- The variable `spark` is available in scope\n"
        "- If a required column doesn't exist, derive it if possible or raise ValueError\n\n"
        "Output ONLY executable Python code."
    )

    user = (
        f"PIPELINE TEMPLATE REQUIREMENTS:\n{schema_summary}\n\n"
        f"CONFIG FIELDS:\n{config_summary}\n\n"
        f"AVAILABLE SOURCE DATA:\n{data_summary}\n\n"
        f"CURRENT CONFIG:\n{config_dict}\n\n"
    )

    if brand_context:
        user += f"BRAND-SPECIFIC CONTEXT:\n{brand_context}\n\n"

    user += (
        "Generate preprocessing code that transforms the source data to match "
        "the template requirements. Handle column name differences, data type "
        "mismatches, and any structural transformations needed. "
        "If no transformation is needed, output only a comment saying so."
    )

    raw = llm.call(system, user, temperature=0.1)
    return strip_markdown_fences(raw)


def validate_adapter_code(code: str) -> list[str]:
    """Validate adapter code for syntax errors and basic safety.

    Returns a list of warning strings (empty if clean).
    """
    warnings = []

    # Syntax check
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        warnings.append(f"Syntax error: {e}")
        return warnings

    # Basic safety checks via AST
    for node in ast.walk(tree):
        # Warn on dangerous imports
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in ("os", "subprocess", "shutil"):
                    warnings.append(
                        f"Code imports '{alias.name}' — review for safety"
                    )
        elif isinstance(node, ast.ImportFrom):
            if node.module in ("os", "subprocess", "shutil"):
                warnings.append(
                    f"Code imports from '{node.module}' — review for safety"
                )
        # Warn on sys.exit / exit()
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in ("exit", "quit"):
                warnings.append("Code calls exit() — this would kill the session")
            elif isinstance(node.func, ast.Attribute) and node.func.attr == "exit":
                warnings.append("Code calls .exit() — this would kill the session")

    return warnings
