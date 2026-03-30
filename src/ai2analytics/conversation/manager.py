"""Conversation manager — structured AI-driven Q&A to fill config gaps."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

from ai2analytics.llm import LLMClient
from ai2analytics.discovery.surveyor import DataSurvey, profile_for_llm
from ai2analytics.templates.base import BaseTemplate


@dataclass
class Question:
    """A single question for the user."""
    key: str
    text: str
    field_name: str  # config field this maps to
    default: Any = None
    choices: list[str] | None = None
    required: bool = True
    answer: Any = None


@dataclass
class ConversationState:
    """Tracks the state of the configuration conversation."""
    template_name: str = ""
    questions: list[Question] = field(default_factory=list)
    answers: dict[str, Any] = field(default_factory=dict)
    config_dict: dict[str, Any] = field(default_factory=dict)
    adapter_code: str = ""
    is_complete: bool = False


class ConversationManager:
    """Manages structured back-and-forth to configure a pipeline template.

    The flow:
    1. Analyze survey data against template requirements
    2. Auto-map what can be inferred
    3. Generate questions for what can't
    4. Collect answers and produce a validated config
    5. If data needs transformation, generate adapter code
    """

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def analyze_fit(
        self,
        survey: DataSurvey,
        template: type[BaseTemplate],
        user_prompt: str,
    ) -> ConversationState:
        """Analyze how well available data fits a template and generate questions."""
        state = ConversationState(template_name=template.name)

        schema_summary = template.get_schema_summary()
        config_summary = template.get_config_summary()
        data_summary = profile_for_llm(survey)

        system = (
            "You are a data engineering assistant. You are helping configure an "
            "analytics pipeline by matching available data tables to a template's "
            "requirements.\n\n"
            "For each required table, identify the best matching available table "
            "and column mappings. Identify gaps.\n\n"
            "CRITICAL: Use the EXACT config field names from the CONFIG FIELDS "
            "section for all `field_name` values in questions and all keys in "
            "`auto_config`. For example, if the config has a field called "
            "`hcp_weekly_table`, use that exact string — do not invent names.\n\n"
            "Output JSON with this structure:\n"
            "{\n"
            '  "mappings": [\n'
            '    {"template_key": "...", "matched_table": "full.table.name", '
            '"column_map": {"template_col": "actual_col"}, "confidence": 0.9}\n'
            "  ],\n"
            '  "gaps": ["description of what is missing or unclear"],\n'
            '  "questions": [\n'
            '    {"key": "field_name", "text": "question", '
            '"field_name": "exact_config_field", "default": null, '
            '"choices": null, "required": true}\n'
            "  ],\n"
            '  "auto_config": {"exact_config_field": "auto-detected value"}\n'
            "}"
        )

        user = (
            f"USER INTENT: {user_prompt}\n\n"
            f"TEMPLATE DATA REQUIREMENTS:\n{schema_summary}\n\n"
            f"CONFIG FIELDS (use these exact names):\n{config_summary}\n\n"
            f"AVAILABLE DATA:\n{data_summary}\n\n"
            "Match the available data to the template requirements. "
            "For each table and column you can confidently map, put the value "
            "in auto_config using the exact config field name. "
            "For anything you cannot infer, generate a question with the "
            "config field name as the field_name."
        )

        try:
            result = self.llm.call_json(system, user, temperature=0.1)
        except Exception as e:
            print(f"  WARNING: LLM analysis failed ({e}), generating default questions")
            result = {"mappings": [], "gaps": [], "questions": [], "auto_config": {}}

        # Process auto-config — only accept keys that are actual config fields
        valid_fields = _get_config_field_names(template)
        for k, v in result.get("auto_config", {}).items():
            if not valid_fields or k in valid_fields:
                state.config_dict[k] = v

        # Process questions — validate field_name against config fields
        for q in result.get("questions", []):
            fn = q.get("field_name", "")
            if valid_fields and fn not in valid_fields:
                # LLM hallucinated a field name — try to find closest match
                fn = _closest_field(fn, valid_fields) or fn
            state.questions.append(Question(
                key=q.get("key", fn),
                text=q.get("text", ""),
                field_name=fn,
                default=q.get("default"),
                choices=q.get("choices"),
                required=q.get("required", True),
            ))

        # Ensure essential questions for fields that must be set
        self._ensure_essential_questions(state, template)

        return state

    def present_questions(self, state: ConversationState) -> str:
        """Format questions for display in a notebook cell."""
        lines = [
            f"Configuration for {state.template_name}",
            "=" * 60,
        ]

        if state.config_dict:
            lines.append("")
            lines.append("Auto-detected:")
            for k, v in state.config_dict.items():
                lines.append(f"  {k}: {v}")

        unanswered = [q for q in state.questions if q.answer is None]
        answered = [q for q in state.questions if q.answer is not None]

        if answered:
            lines.append("")
            lines.append("Answered:")
            for q in answered:
                lines.append(f"  {q.field_name}: {q.answer}")

        if unanswered:
            lines.append("")
            lines.append("Still needed:")
            lines.append("-" * 40)
            for i, q in enumerate(unanswered, 1):
                req = " *" if q.required else ""
                default = f" (default: {q.default})" if q.default is not None else ""
                choices = f"\n    Options: {', '.join(q.choices)}" if q.choices else ""
                lines.append(f"\n  {i}. [{q.field_name}] {q.text}{req}{default}{choices}")

        lines.append("")
        if unanswered:
            lines.append("Call session.answer({field_name: value, ...}) to provide answers.")
        else:
            lines.append("All questions answered. Call session.run() to execute.")
        return "\n".join(lines)

    def apply_answers(
        self,
        state: ConversationState,
        answers: dict[str, Any],
    ) -> ConversationState:
        """Apply user answers to the conversation state."""
        for q in state.questions:
            # Match by field_name or key
            val = answers.get(q.field_name, answers.get(q.key))
            if val is not None:
                q.answer = val
                state.config_dict[q.field_name] = val
            elif q.answer is None and q.default is not None:
                q.answer = q.default
                state.config_dict[q.field_name] = q.default

        # Also accept direct config field answers not tied to a question
        valid_fields = set()
        for q in state.questions:
            valid_fields.add(q.field_name)
            valid_fields.add(q.key)
        for k, v in answers.items():
            if k not in valid_fields:
                state.config_dict[k] = v

        unanswered = [q for q in state.questions if q.required and q.answer is None]
        state.is_complete = len(unanswered) == 0

        if unanswered:
            print(f"  Still need: {[q.field_name for q in unanswered]}")
        else:
            print("  All required fields set.")

        return state

    def generate_adapter(
        self,
        state: ConversationState,
        survey: DataSurvey,
        template: type[BaseTemplate],
    ) -> str:
        """Generate preprocessing/adapter code if data doesn't fit template directly."""
        schema_summary = template.get_schema_summary()
        config_summary = template.get_config_summary()
        data_summary = profile_for_llm(survey)

        system = (
            "You are a data engineering expert working in Databricks with PySpark. "
            "Generate Python code that transforms the available data into the exact "
            "format required by the pipeline template.\n\n"
            "The code should:\n"
            "1. Read from the source tables/files\n"
            "2. Rename/transform columns to match the config field expectations\n"
            "3. Handle data type conversions\n"
            "4. Write to intermediate tables or temp views the pipeline can consume\n"
            "5. Print validation summaries after each step\n\n"
            "The code will be executed with `spark` available in scope.\n"
            "Output ONLY executable Python code with comments."
        )

        user = (
            f"TEMPLATE REQUIREMENTS:\n{schema_summary}\n\n"
            f"CONFIG FIELDS:\n{config_summary}\n\n"
            f"AVAILABLE DATA:\n{data_summary}\n\n"
            f"CURRENT CONFIG:\n{state.config_dict}\n\n"
            "Generate adapter code. If no transformation is needed, "
            "output a comment saying so."
        )

        code = self.llm.call(system, user, temperature=0.1)
        from ai2analytics.llm import strip_markdown_fences
        state.adapter_code = strip_markdown_fences(code)
        return state.adapter_code

    def _ensure_essential_questions(
        self, state: ConversationState, template: type[BaseTemplate]
    ):
        """Add questions for config fields that must be set but aren't yet covered.

        Introspects the config dataclass to find fields with empty-string defaults
        (meaning 'must be user-supplied') that aren't already in auto_config or
        existing questions.
        """
        if template.config_class is None or not dataclasses.is_dataclass(template.config_class):
            return

        covered = {q.field_name for q in state.questions}
        covered.update(state.config_dict.keys())

        for f in dataclasses.fields(template.config_class):
            if f.name in covered:
                continue

            # A field needs user input if its default is empty string
            is_essential = (
                f.default is not dataclasses.MISSING
                and isinstance(f.default, str)
                and f.default == ""
            )
            if not is_essential:
                continue

            # Generate a human-readable question from the field name
            readable = f.name.replace("_", " ").replace("col ", "column name for ")
            text = f"What is the {readable}?"

            state.questions.append(Question(
                key=f.name,
                text=text,
                field_name=f.name,
                required=True,
            ))


def _get_config_field_names(template: type[BaseTemplate]) -> set[str]:
    """Get the set of valid config field names for a template."""
    if template.config_class is None or not dataclasses.is_dataclass(template.config_class):
        return set()
    return {f.name for f in dataclasses.fields(template.config_class)}


def _closest_field(name: str, valid: set[str]) -> str | None:
    """Find the closest matching field name (simple substring match)."""
    name_lower = name.lower().replace("-", "_")
    # Exact match first
    if name_lower in valid:
        return name_lower
    # Substring containment
    for v in sorted(valid):
        if name_lower in v or v in name_lower:
            return v
    return None
