"""Knowledge retriever — formats past decisions and context for LLM prompts."""

from __future__ import annotations

import json
from typing import Any

from ai2analytics.discovery.surveyor import DataSurvey, profile_for_llm
from ai2analytics.knowledge.context_store import ContextEntry, ContextStore
from ai2analytics.knowledge.decision_store import DecisionRecord, DecisionStore


class KnowledgeRetriever:
    """Retrieve and format past knowledge for injection into LLM prompts.

    Combines decision history and synthesized context into structured text
    that can be prepended to LLM prompts during analysis and adapter
    generation.

    Usage::

        retriever = KnowledgeRetriever(decision_store, context_store)
        knowledge = retriever.retrieve(template_name="hcp_overview")
        # Inject into LLM prompt ...
    """

    def __init__(
        self,
        decision_store: DecisionStore,
        context_store: ContextStore,
        max_decisions: int = 5,
        max_context: int = 5,
    ):
        self.decision_store = decision_store
        self.context_store = context_store
        self.max_decisions = max_decisions
        self.max_context = max_context

    def retrieve(
        self,
        template_name: str | None = None,
        scope: dict[str, str] | None = None,
        survey: DataSurvey | None = None,
    ) -> str:
        """Retrieve formatted knowledge for general LLM prompt injection.

        Returns a combined string with past decisions and context entries,
        ready to be included in a system or user prompt.
        """
        decisions = self.decision_store.query(
            template_name=template_name, limit=self.max_decisions,
        )
        context = self.context_store.query(
            scope=scope, template_name=template_name, limit=self.max_context,
        )

        if survey:
            decisions = self._rank_by_relevance(decisions, survey)

        sections: list[str] = []

        if decisions:
            sections.append("PAST DECISIONS:")
            sections.append(self._format_decisions(decisions))

        if context:
            sections.append("LEARNED PATTERNS:")
            sections.append(self._format_context(context))

        if not sections:
            return ""

        return (
            "--- KNOWLEDGE BASE ---\n"
            + "\n\n".join(sections)
            + "\n--- END KNOWLEDGE BASE ---"
        )

    def retrieve_for_analysis(
        self,
        template_name: str | None = None,
        survey: DataSurvey | None = None,
        scope: dict[str, str] | None = None,
    ) -> str:
        """Retrieve knowledge focused on column mappings for analysis.

        Emphasizes past column mappings, auto-detected values, and data
        quality patterns — useful when the LLM is deciding how to map
        available data to a template.
        """
        decisions = self.decision_store.query(
            template_name=template_name, limit=self.max_decisions,
        )
        context = self.context_store.query(
            scope=scope,
            category="column_mapping",
            template_name=template_name,
            limit=self.max_context,
        )

        if survey:
            decisions = self._rank_by_relevance(decisions, survey)

        sections: list[str] = []

        if decisions:
            sections.append("PAST COLUMN MAPPINGS:")
            mapping_lines: list[str] = []
            for d in decisions:
                mapping_lines.append(f"  Run {d.run_id} ({d.template_name}):")
                if d.auto_detected:
                    mapping_lines.append(
                        f"    Auto-detected: {json.dumps(d.auto_detected, default=str)}"
                    )
                if d.user_answers:
                    mapping_lines.append(
                        f"    User-provided: {json.dumps(d.user_answers, default=str)}"
                    )
                if d.outcome_notes:
                    mapping_lines.append(f"    Outcome: {d.outcome_notes}")
            sections.append("\n".join(mapping_lines))

        if context:
            sections.append("MAPPING PATTERNS:")
            sections.append(self._format_context(context))

        # Also include data quality context
        quality_context = self.context_store.query(
            scope=scope,
            category="data_quality",
            template_name=template_name,
            limit=3,
        )
        if quality_context:
            sections.append("DATA QUALITY NOTES:")
            sections.append(self._format_context(quality_context))

        if not sections:
            return ""

        return (
            "--- ANALYSIS KNOWLEDGE ---\n"
            + "\n\n".join(sections)
            + "\n--- END ANALYSIS KNOWLEDGE ---"
        )

    def retrieve_for_adapter(
        self,
        template_name: str | None = None,
        survey: DataSurvey | None = None,
        scope: dict[str, str] | None = None,
    ) -> str:
        """Retrieve knowledge focused on past adapter code.

        Emphasizes adapter code from previous runs and adapter patterns —
        useful when the LLM is generating data transformation code.
        """
        decisions = self.decision_store.query(
            template_name=template_name, limit=self.max_decisions,
        )
        context = self.context_store.query(
            scope=scope,
            category="adapter_pattern",
            template_name=template_name,
            limit=self.max_context,
        )

        if survey:
            decisions = self._rank_by_relevance(decisions, survey)

        # Only include decisions that have adapter code
        decisions_with_code = [d for d in decisions if d.adapter_code.strip()]

        sections: list[str] = []

        if decisions_with_code:
            sections.append("PAST ADAPTER CODE:")
            for d in decisions_with_code:
                sections.append(f"  --- Run {d.run_id} ({d.template_name}) ---")
                sections.append(f"  Config: {json.dumps(d.config_dict, default=str)}")
                if d.outcome_notes:
                    sections.append(f"  Outcome: {d.outcome_notes}")
                sections.append(f"  Code:\n{d.adapter_code}")
                sections.append("")

        if context:
            sections.append("ADAPTER PATTERNS:")
            sections.append(self._format_context(context))

        # Also include troubleshooting context
        troubleshooting = self.context_store.query(
            scope=scope,
            category="troubleshooting",
            template_name=template_name,
            limit=3,
        )
        if troubleshooting:
            sections.append("TROUBLESHOOTING NOTES:")
            sections.append(self._format_context(troubleshooting))

        if not sections:
            return ""

        return (
            "--- ADAPTER KNOWLEDGE ---\n"
            + "\n\n".join(sections)
            + "\n--- END ADAPTER KNOWLEDGE ---"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _format_decisions(self, decisions: list[DecisionRecord]) -> str:
        """Format decision records into a compact readable block."""
        lines: list[str] = []
        for d in decisions:
            lines.append(f"  Run {d.run_id} ({d.timestamp}):")
            lines.append(f"    Template: {d.template_name}")
            lines.append(f"    Config: {json.dumps(d.config_dict, default=str)}")
            if d.outcome_notes:
                lines.append(f"    Outcome: {d.outcome_notes}")
            if d.outcome_metrics:
                lines.append(
                    f"    Metrics: {json.dumps(d.outcome_metrics, default=str)}"
                )
            if d.tags:
                lines.append(f"    Tags: {', '.join(d.tags)}")
        return "\n".join(lines)

    def _format_context(self, entries: list[ContextEntry]) -> str:
        """Format context entries into a compact readable block."""
        lines: list[str] = []
        for e in entries:
            conf = f" ({e.confidence:.0%})" if e.confidence else ""
            lines.append(f"  [{e.category}] {e.title}{conf}")
            lines.append(f"    {e.content}")
        return "\n".join(lines)

    def _rank_by_relevance(
        self,
        decisions: list[DecisionRecord],
        survey: DataSurvey,
    ) -> list[DecisionRecord]:
        """Rank decisions by relevance to the current data survey.

        Uses a simple heuristic: decisions whose data_profile shares more
        table names with the current survey rank higher.
        """
        current_tables = {t.full_name.lower() for t in survey.tables}
        if not current_tables:
            return decisions

        scored: list[tuple[float, DecisionRecord]] = []
        for d in decisions:
            profile_lower = d.data_profile.lower()
            matches = sum(1 for t in current_tables if t in profile_lower)
            score = matches / len(current_tables) if current_tables else 0.0
            scored.append((score, d))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in scored]
