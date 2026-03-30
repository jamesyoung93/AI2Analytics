"""Conversation manager — structured AI-driven Q&A to fill config gaps."""

from __future__ import annotations

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

        # Use LLM to match survey data to template requirements
        schema_summary = template.get_schema_summary()
        data_summary = profile_for_llm(survey)

        system = (
            "You are a data engineering assistant. Analyze how available data tables "
            "match a pipeline template's requirements. For each required table, identify "
            "the best matching available table and column mappings. Identify gaps.\n\n"
            "Output JSON with this structure:\n"
            '{\n'
            '  "mappings": [\n'
            '    {"template_key": "...", "matched_table": "...", "column_map": {"template_col": "actual_col", ...}, "confidence": 0.9},\n'
            '    ...\n'
            '  ],\n'
            '  "gaps": ["description of what is missing or unclear"],\n'
            '  "questions": [\n'
            '    {"key": "q1", "text": "question text", "field_name": "config_field", "default": null, "choices": null, "required": true},\n'
            '    ...\n'
            '  ],\n'
            '  "auto_config": {"field_name": "auto-detected value", ...}\n'
            '}'
        )

        user = (
            f"USER INTENT: {user_prompt}\n\n"
            f"TEMPLATE REQUIREMENTS:\n{schema_summary}\n\n"
            f"AVAILABLE DATA:\n{data_summary}\n\n"
            "Match the data to the template. Auto-fill what you can confidently map. "
            "Generate questions for anything you cannot infer."
        )

        try:
            result = self.llm.call_json(system, user, temperature=0.1)
        except Exception as e:
            print(f"  WARNING: LLM analysis failed ({e}), generating default questions")
            result = {"mappings": [], "gaps": [], "questions": [], "auto_config": {}}

        # Process auto-config
        state.config_dict = result.get("auto_config", {})

        # Process questions
        for q in result.get("questions", []):
            state.questions.append(Question(
                key=q.get("key", ""),
                text=q.get("text", ""),
                field_name=q.get("field_name", ""),
                default=q.get("default"),
                choices=q.get("choices"),
                required=q.get("required", True),
            ))

        # Always ask essential questions if not auto-detected
        self._ensure_essential_questions(state, template)

        return state

    def present_questions(self, state: ConversationState) -> str:
        """Format questions for display in a notebook cell."""
        lines = [
            f"Configuration Questions for {state.template_name}",
            "=" * 60,
            "",
            "Auto-detected configuration:",
        ]
        for k, v in state.config_dict.items():
            lines.append(f"  {k}: {v}")

        if state.questions:
            lines.append("")
            lines.append("Please answer the following questions:")
            lines.append("-" * 40)
            for i, q in enumerate(state.questions, 1):
                req = " *" if q.required else ""
                default = f" (default: {q.default})" if q.default is not None else ""
                choices = f"\n    Options: {', '.join(q.choices)}" if q.choices else ""
                lines.append(f"\n  {i}. [{q.key}] {q.text}{req}{default}{choices}")

        lines.append("")
        lines.append("Call session.answer({key: value, ...}) to provide answers.")
        return "\n".join(lines)

    def apply_answers(
        self,
        state: ConversationState,
        answers: dict[str, Any],
    ) -> ConversationState:
        """Apply user answers to the conversation state."""
        for q in state.questions:
            if q.key in answers:
                q.answer = answers[q.key]
                state.config_dict[q.field_name] = answers[q.key]
            elif q.default is not None:
                q.answer = q.default
                state.config_dict[q.field_name] = q.default

        # Check completeness
        unanswered = [q for q in state.questions if q.required and q.answer is None]
        state.is_complete = len(unanswered) == 0

        if unanswered:
            print(f"  Still need answers for: {[q.key for q in unanswered]}")

        return state

    def generate_adapter(
        self,
        state: ConversationState,
        survey: DataSurvey,
        template: type[BaseTemplate],
    ) -> str:
        """Generate preprocessing/adapter code if data doesn't fit template directly."""
        schema_summary = template.get_schema_summary()
        data_summary = profile_for_llm(survey)

        system = (
            "You are a data engineering expert. Generate Python/PySpark code "
            "that transforms the available data into the format required by the "
            "pipeline template. The code should:\n"
            "1. Read from the source tables\n"
            "2. Rename/transform columns as needed\n"
            "3. Handle any data type conversions\n"
            "4. Write to intermediate tables that the pipeline can consume\n\n"
            "Output ONLY executable Python code with comments."
        )

        user = (
            f"TEMPLATE REQUIREMENTS:\n{schema_summary}\n\n"
            f"AVAILABLE DATA:\n{data_summary}\n\n"
            f"CONFIG SO FAR:\n{state.config_dict}\n\n"
            "Generate the adapter code. If no transformation is needed, "
            "output a comment saying so."
        )

        code = self.llm.call(system, user, temperature=0.1)
        from ai2analytics.llm import strip_markdown_fences
        state.adapter_code = strip_markdown_fences(code)
        return state.adapter_code

    def _ensure_essential_questions(
        self, state: ConversationState, template: type[BaseTemplate]
    ):
        """Add essential questions if not already present."""
        existing_fields = {q.field_name for q in state.questions}
        existing_fields.update(state.config_dict.keys())

        essential = [
            ("drug_name", "What is the drug/brand name?", True),
            ("output_csv", "Where should the output CSV be written? (DBFS path)", True),
        ]

        for field_name, text, required in essential:
            if field_name not in existing_fields:
                state.questions.append(Question(
                    key=field_name,
                    text=text,
                    field_name=field_name,
                    required=required,
                ))
