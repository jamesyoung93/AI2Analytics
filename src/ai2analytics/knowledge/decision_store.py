"""Decision store — logs and queries pipeline configuration decisions."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ai2analytics.conversation.manager import ConversationState
from ai2analytics.discovery.surveyor import DataSurvey, profile_for_llm


@dataclass
class DecisionRecord:
    """A single logged pipeline decision."""
    run_id: str = ""
    timestamp: str = ""
    template_name: str = ""
    config_dict: dict[str, Any] = field(default_factory=dict)
    data_profile: str = ""
    user_answers: dict[str, Any] = field(default_factory=dict)
    auto_detected: dict[str, Any] = field(default_factory=dict)
    adapter_code: str = ""
    outcome_notes: str = ""
    outcome_metrics: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)


class DecisionStore:
    """Persist and query pipeline configuration decisions.

    Supports two backends:

    - ``"json"``: append-only JSONL file (default, no Spark needed).
    - ``"delta"``: Spark Delta table for production workloads.

    Usage::

        store = DecisionStore(backend="json", path="/tmp/decisions.jsonl")
        store.log(record)
        past = store.query(template_name="hcp_overview")
    """

    def __init__(
        self,
        backend: str = "json",
        path: str = "decisions.jsonl",
        spark: Any | None = None,
        table_name: str = "ai2analytics.knowledge.decisions",
    ):
        self.backend = backend
        self.path = path
        self.spark = spark
        self.table_name = table_name

        if backend == "json":
            parent = Path(path).parent
            if not parent.exists():
                parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: DecisionRecord) -> str:
        """Persist a decision record. Returns the run_id."""
        if not record.run_id:
            record.run_id = uuid.uuid4().hex[:12]
        if not record.timestamp:
            record.timestamp = datetime.now(timezone.utc).isoformat()

        if self.backend == "json":
            return self._log_json(record)
        elif self.backend == "delta":
            return self._log_delta(record)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def log_from_session(
        self,
        template_name: str,
        config_dict: dict[str, Any],
        survey: DataSurvey,
        state: ConversationState,
        results: dict[str, Any] | None = None,
        outcome_notes: str = "",
        tags: list[str] | None = None,
    ) -> str:
        """Build a DecisionRecord from session artifacts and log it."""
        # Separate auto-detected vs user-answered config values
        user_answers: dict[str, Any] = {}
        auto_detected: dict[str, Any] = {}

        answered_fields = {q.field_name for q in state.questions if q.answer is not None}
        for k, v in config_dict.items():
            if k in answered_fields:
                user_answers[k] = v
            else:
                auto_detected[k] = v

        record = DecisionRecord(
            template_name=template_name,
            config_dict=config_dict,
            data_profile=profile_for_llm(survey),
            user_answers=user_answers,
            auto_detected=auto_detected,
            adapter_code=state.adapter_code,
            outcome_notes=outcome_notes,
            outcome_metrics=results or {},
            tags=tags or [],
        )
        return self.log(record)

    def query(
        self,
        template_name: str | None = None,
        tags: list[str] | None = None,
        limit: int = 10,
    ) -> list[DecisionRecord]:
        """Query past decisions, optionally filtered by template and tags."""
        if self.backend == "json":
            return self._query_json(template_name, tags, limit)
        elif self.backend == "delta":
            return self._query_delta(template_name, tags, limit)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    # ------------------------------------------------------------------
    # JSON backend
    # ------------------------------------------------------------------

    def _log_json(self, record: DecisionRecord) -> str:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(record), default=str) + "\n")
        return record.run_id

    def _query_json(
        self,
        template_name: str | None,
        tags: list[str] | None,
        limit: int,
    ) -> list[DecisionRecord]:
        if not os.path.exists(self.path):
            return []

        records: list[DecisionRecord] = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                records.append(DecisionRecord(**data))

        # Filter
        if template_name:
            records = [r for r in records if r.template_name == template_name]
        if tags:
            tag_set = set(tags)
            records = [r for r in records if tag_set.intersection(r.tags)]

        # Most recent first
        records.sort(key=lambda r: r.timestamp, reverse=True)
        return records[:limit]

    # ------------------------------------------------------------------
    # Delta backend
    # ------------------------------------------------------------------

    def _log_delta(self, record: DecisionRecord) -> str:
        if self.spark is None:
            raise RuntimeError("Spark session required for delta backend")

        data = asdict(record)
        # Serialize nested structures as JSON strings for Delta compatibility
        for key in ("config_dict", "user_answers", "auto_detected", "outcome_metrics", "tags"):
            data[key] = json.dumps(data[key], default=str)

        df = self.spark.createDataFrame([data])
        df.write.mode("append").saveAsTable(self.table_name)
        return record.run_id

    def _query_delta(
        self,
        template_name: str | None,
        tags: list[str] | None,
        limit: int,
    ) -> list[DecisionRecord]:
        if self.spark is None:
            raise RuntimeError("Spark session required for delta backend")

        where_clauses = []
        if template_name:
            where_clauses.append(f"template_name = '{template_name}'")
        if tags:
            # tags stored as JSON array string — use LIKE for simple matching
            for tag in tags:
                where_clauses.append(f"tags LIKE '%{tag}%'")

        where_sql = f" WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        query = (
            f"SELECT * FROM {self.table_name}{where_sql} "
            f"ORDER BY timestamp DESC LIMIT {limit}"
        )

        rows = self.spark.sql(query).toPandas()
        records: list[DecisionRecord] = []
        for _, row in rows.iterrows():
            data = row.to_dict()
            # Deserialize JSON string fields back to dicts/lists
            for key in ("config_dict", "user_answers", "auto_detected", "outcome_metrics"):
                if isinstance(data.get(key), str):
                    data[key] = json.loads(data[key])
            if isinstance(data.get("tags"), str):
                data["tags"] = json.loads(data["tags"])
            records.append(DecisionRecord(**data))

        return records
