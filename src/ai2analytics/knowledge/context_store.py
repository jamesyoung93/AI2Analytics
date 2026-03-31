"""Context store — persists synthesized knowledge extracted from decisions."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from ai2analytics.knowledge.decision_store import DecisionRecord, DecisionStore
from ai2analytics.llm import LLMClient


@dataclass
class ContextEntry:
    """A piece of synthesized knowledge derived from past decisions."""
    entry_id: str = ""
    created: str = ""
    updated: str = ""
    scope: dict[str, str] = field(default_factory=dict)
    category: str = ""
    title: str = ""
    content: str = ""
    template_name: str = ""
    confidence: float = 0.0
    source_run_ids: list[str] = field(default_factory=list)


class ContextStore:
    """Persist and query synthesized context entries.

    Supports two backends:

    - ``"json"``: append-only JSONL file (default, no Spark needed).
    - ``"delta"``: Spark Delta table for production workloads.

    Usage::

        store = ContextStore(backend="json", path="/tmp/context.jsonl")
        store.add(entry)
        entries = store.query(scope={"domain": "pharma"}, category="column_mapping")
    """

    def __init__(
        self,
        backend: str = "json",
        path: str = "context.jsonl",
        spark: Any | None = None,
        table_name: str = "ai2analytics.knowledge.context",
    ):
        self.backend = backend
        self.path = path
        self.spark = spark
        self.table_name = table_name

        if backend == "json":
            parent = os.path.dirname(os.path.abspath(path))
            if not os.path.exists(parent):
                os.makedirs(parent, exist_ok=True)

    def add(self, entry: ContextEntry) -> str:
        """Persist a context entry. Returns the entry_id."""
        now = datetime.now(timezone.utc).isoformat()
        if not entry.entry_id:
            entry.entry_id = uuid.uuid4().hex[:12]
        if not entry.created:
            entry.created = now
        entry.updated = now

        if self.backend == "json":
            return self._add_json(entry)
        elif self.backend == "delta":
            return self._add_delta(entry)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def query(
        self,
        scope: dict[str, str] | None = None,
        category: str | None = None,
        template_name: str | None = None,
        limit: int = 10,
    ) -> list[ContextEntry]:
        """Query context entries with scope matching and optional filters."""
        if self.backend == "json":
            return self._query_json(scope, category, template_name, limit)
        elif self.backend == "delta":
            return self._query_delta(scope, category, template_name, limit)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def extract_from_decisions(
        self,
        decision_store: DecisionStore,
        llm: LLMClient,
        template_name: str | None = None,
        min_runs: int = 2,
    ) -> list[ContextEntry]:
        """Synthesize context entries from past decisions using an LLM.

        Queries the decision store, sends the history to the LLM to identify
        patterns and best practices, and returns new ContextEntry objects
        (automatically persisted).

        Args:
            decision_store: Source of past decision records.
            llm: LLM client for pattern synthesis.
            template_name: Optional filter for a specific template.
            min_runs: Minimum number of decisions required before extracting.
        """
        decisions = decision_store.query(template_name=template_name, limit=50)
        if len(decisions) < min_runs:
            return []

        decisions_text = _format_decisions_for_llm(decisions)

        system = (
            "You are a data engineering knowledge manager. Analyze the following "
            "pipeline configuration decisions and extract reusable patterns.\n\n"
            "For each pattern you find, output a JSON object with:\n"
            "- category: one of 'column_mapping', 'data_quality', 'adapter_pattern', "
            "'config_preference', 'troubleshooting'\n"
            "- title: short descriptive title\n"
            "- content: detailed description of the pattern or best practice\n"
            "- confidence: 0.0-1.0 how confident you are this is a real pattern\n"
            "- source_run_ids: list of run_ids that support this pattern\n\n"
            "Output a JSON array of these objects."
        )

        user = (
            f"PAST DECISIONS ({len(decisions)} runs):\n\n{decisions_text}\n\n"
            "Extract reusable patterns and best practices from these decisions."
        )

        try:
            raw = llm.call_json(system, user, temperature=0.2)
        except Exception as e:
            print(f"  WARNING: LLM extraction failed ({e})")
            return []

        if not isinstance(raw, list):
            raw = [raw]

        entries: list[ContextEntry] = []
        for item in raw:
            entry = ContextEntry(
                scope={"template": template_name} if template_name else {},
                category=item.get("category", "general"),
                title=item.get("title", ""),
                content=item.get("content", ""),
                template_name=template_name or "",
                confidence=float(item.get("confidence", 0.5)),
                source_run_ids=item.get("source_run_ids", []),
            )
            self.add(entry)
            entries.append(entry)

        return entries

    # ------------------------------------------------------------------
    # JSON backend
    # ------------------------------------------------------------------

    def _add_json(self, entry: ContextEntry) -> str:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(entry), default=str) + "\n")
        return entry.entry_id

    def _query_json(
        self,
        scope: dict[str, str] | None,
        category: str | None,
        template_name: str | None,
        limit: int,
    ) -> list[ContextEntry]:
        if not os.path.exists(self.path):
            return []

        entries: list[ContextEntry] = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                entries.append(ContextEntry(**data))

        # Filter by scope — entry matches if all entry scope keys match query
        if scope:
            entries = [e for e in entries if _scope_matches(e.scope, scope)]

        if category:
            entries = [e for e in entries if e.category == category]

        if template_name:
            entries = [e for e in entries if e.template_name == template_name]

        # Highest confidence first
        entries.sort(key=lambda e: e.confidence, reverse=True)
        return entries[:limit]

    # ------------------------------------------------------------------
    # Delta backend
    # ------------------------------------------------------------------

    def _add_delta(self, entry: ContextEntry) -> str:
        if self.spark is None:
            raise RuntimeError("Spark session required for delta backend")

        data = asdict(entry)
        # Serialize nested structures as JSON strings for Delta compatibility
        for key in ("scope", "source_run_ids"):
            data[key] = json.dumps(data[key], default=str)

        df = self.spark.createDataFrame([data])
        df.write.mode("append").saveAsTable(self.table_name)
        return entry.entry_id

    def _query_delta(
        self,
        scope: dict[str, str] | None,
        category: str | None,
        template_name: str | None,
        limit: int,
    ) -> list[ContextEntry]:
        if self.spark is None:
            raise RuntimeError("Spark session required for delta backend")

        where_clauses = []
        if category:
            where_clauses.append(f"category = '{category}'")
        if template_name:
            where_clauses.append(f"template_name = '{template_name}'")
        if scope:
            for k, v in scope.items():
                where_clauses.append(f"scope LIKE '%\"{k}\": \"{v}\"%'")

        where_sql = f" WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        query = (
            f"SELECT * FROM {self.table_name}{where_sql} "
            f"ORDER BY confidence DESC LIMIT {limit}"
        )

        rows = self.spark.sql(query).toPandas()
        entries: list[ContextEntry] = []
        for _, row in rows.iterrows():
            data = row.to_dict()
            if isinstance(data.get("scope"), str):
                data["scope"] = json.loads(data["scope"])
            if isinstance(data.get("source_run_ids"), str):
                data["source_run_ids"] = json.loads(data["source_run_ids"])
            entries.append(ContextEntry(**data))

        return entries


def _scope_matches(entry_scope: dict[str, str], query_scope: dict[str, str]) -> bool:
    """Check if an entry's scope matches a query scope.

    An entry matches if every key in the entry's scope has the same value
    in the query scope. An entry with an empty scope matches everything.
    """
    for k, v in entry_scope.items():
        if query_scope.get(k) != v:
            return False
    return True


def _format_decisions_for_llm(decisions: list[DecisionRecord]) -> str:
    """Format decision records into a compact string for LLM consumption."""
    lines: list[str] = []
    for d in decisions:
        lines.append(f"--- Run {d.run_id} ({d.timestamp}) ---")
        lines.append(f"Template: {d.template_name}")
        lines.append(f"Config: {json.dumps(d.config_dict, default=str)}")
        if d.user_answers:
            lines.append(f"User answers: {json.dumps(d.user_answers, default=str)}")
        if d.auto_detected:
            lines.append(f"Auto-detected: {json.dumps(d.auto_detected, default=str)}")
        if d.adapter_code:
            lines.append(f"Adapter code:\n{d.adapter_code}")
        if d.outcome_notes:
            lines.append(f"Outcome: {d.outcome_notes}")
        if d.outcome_metrics:
            lines.append(f"Metrics: {json.dumps(d.outcome_metrics, default=str)}")
        if d.tags:
            lines.append(f"Tags: {', '.join(d.tags)}")
        lines.append("")
    return "\n".join(lines)
