# Building an Organizational Learning Loop for Analytics

There is a recurring failure mode in analytics organizations that no amount of model sophistication can fix: the person who set up Brand X's pipeline leaves, and the person picking up Brand Y starts from scratch. The column mappings, the data quality workarounds, the adapter code that fixed the type mismatch in the territory alignment file -- all of it lives in that person's head or buried in a Slack thread from nine months ago.

This is not a technology problem. It is a knowledge management problem. And it is the most expensive hidden cost in analytics teams.

The `ai2analytics` framework addresses this with a knowledge accumulation system that turns every pipeline run into a reusable organizational asset. This post describes the architecture, the privacy model, and the flywheel effect that makes each successive deployment faster than the last.

## The Two Stores

Knowledge in ai2analytics lives in two complementary stores, each serving a different purpose.

### Decision Store: The Audit Trail

Every pipeline run automatically logs a `DecisionRecord` containing:

- **Config dict:** The complete configuration that was used
- **Data profile:** A summary of the data environment (table schemas, row counts, column types)
- **User answers:** Which config fields the user explicitly set
- **Auto-detected values:** Which config fields the LLM inferred from the data
- **Adapter code:** Any data transformation code that was generated and executed
- **Outcome metrics:** Pipeline results (silhouette scores, R-squared, NPI counts)
- **Tags and notes:** Post-run annotations from the analyst

```python
# This happens automatically when you call session.run()
@dataclass
class DecisionRecord:
    run_id: str
    timestamp: str
    template_name: str
    config_dict: dict[str, Any]
    data_profile: str
    user_answers: dict[str, Any]
    auto_detected: dict[str, Any]
    adapter_code: str
    outcome_notes: str
    outcome_metrics: dict[str, Any]
    tags: list[str]
```

The decision store is append-only. Nothing is ever deleted. Every run is a complete, self-contained record of what was done, why, and what happened.

After a run, analysts can annotate results:

```python
session.annotate(
    notes="Silhouette of 0.61 -- good separation. 4 segments align with commercial team's tiering.",
    tags={"region": "US", "therapeutic_area": "immunology"},
)
```

### Context Store: The Pattern Library

While the decision store captures raw history, the context store holds **synthesized knowledge** -- patterns and best practices extracted from multiple runs. Each `ContextEntry` is a curated insight:

```python
@dataclass
class ContextEntry:
    entry_id: str
    scope: dict[str, str]       # e.g., {"region": "EU"}
    category: str               # column_mapping, data_quality, adapter_pattern, ...
    title: str
    content: str
    template_name: str
    confidence: float           # 0.0 - 1.0
    source_run_ids: list[str]   # which runs this was derived from
```

Context entries are categorized:

- **column_mapping:** "EU data consistently uses PRESCRIBER_ID as the entity identifier and MONTH_END for time periods"
- **data_quality:** "Territory alignment files typically cover 85-88% of entities; expect 12-15% unmatched"
- **adapter_pattern:** "When source data uses string NPIs, cast to int64 before merge to avoid silent zero-match joins"
- **config_preference:** "For account-level data, robust normalization outperforms standard normalization due to outlier distribution"
- **troubleshooting:** "If NPI counts drop significantly after the planning date filter, check that the filter column is properly cast to datetime"

These entries can be created manually by analysts or extracted automatically. The `extract_from_decisions()` method sends past decision records to the LLM, which identifies recurring patterns -- column naming conventions, parameter choices that worked well, common data quality issues -- and persists them as context entries for future reference.

## RAG Retrieval: Past Experience in Every Prompt

The knowledge stores are not passive archives. They actively improve the system through retrieval-augmented generation. The `KnowledgeRetriever` formats past decisions and context entries for injection into LLM prompts at three points:

### During Analysis (Data-to-Config Mapping)

When the LLM is mapping discovered data to template requirements, the retriever injects past column mappings and data quality notes. Instead of guessing whether `PRESCRIBER_ID` is a prescriber's personal ID or an account ID, the LLM has evidence from past runs showing how similar columns were mapped.

### During Adapter Generation

When generating data transformation code, the retriever injects past adapter code and troubleshooting notes. If a previous run discovered that string-to-int NPI type mismatches cause silent zero-row merges, that fix appears in every subsequent adapter -- without anyone remembering to add it.

### Relevance Ranking

Not all past decisions are equally relevant. The retriever ranks decisions by similarity to the current data environment using table name overlap between the current survey and past data profiles. A decision from a run that used similar tables ranks higher, keeping the LLM's context focused on the most useful precedents.

## The Flywheel Effect

Here is what this looks like in practice across successive deployments:

**Run 1 (US HCP segmentation, cold start):**
- LLM asks 15 questions (all config fields are unknown)
- User provides column mappings, file paths, clustering parameters
- Pipeline runs, decision is logged
- Analyst annotates: "4 segments, silhouette 0.61, good separation"

**Run 2 (EU account segmentation):**
- Retriever injects Run 1's decisions into the LLM prompt
- LLM auto-detects 8 config fields (it learned the pattern from Run 1)
- LLM asks 7 questions (only EU-specific values)
- Pipeline runs, decision is logged
- Context store now has entries for both US and EU naming conventions

**Run 3 (second US brand, different therapeutic area):**
- Retriever injects both prior decisions, ranked by relevance
- LLM auto-detects 12 config fields (US naming conventions are well-established)
- LLM asks 3 questions (only brand-specific paths and drug names)
- Adapter code is generated with NPI type-casting already included (learned from troubleshooting)

The question count drops from 15 to 7 to 3. Not because the pipeline is simpler -- because the system knows more.

## Enterprise Architecture

Both stores support two backends. For production Databricks environments, a Delta backend writes to shared catalog tables (`ai2analytics.knowledge.decisions` and `ai2analytics.knowledge.context`), queryable via SQL and governed by Unity Catalog permissions. For development, a JSON-lines backend works without Spark:

```python
# Production
session = AnalyticsSession(spark=spark, knowledge_backend="delta")

# Local development
session = AnalyticsSession(spark=None, knowledge_backend="json",
                           knowledge_path=".ai2analytics")
```

## Privacy and Security

A knowledge system in pharma must handle sensitive data carefully. The ai2analytics knowledge store captures **structure, not content**: config field names, data profiles (schemas, row counts, column types), adapter code, and outcome metrics. It never stores raw patient data, individual prescribing volumes, PHI, or row-level values.

The data profile records that a table has 5,000 unique NPIs across 52 weeks, not which NPIs they are. This boundary is enforced at the profiling layer: `profile_for_llm()` generates schema-level summaries only.

## From Tool to Platform

Most analytics tools are point solutions: they run a pipeline and produce output. The knowledge accumulation layer transforms ai2analytics from a tool into a platform -- one that gets better with use.

Every team member who runs a pipeline contributes to the collective knowledge. Every adapter code snippet that fixes a data quality issue becomes available to the next analyst who encounters the same problem. Every successful configuration becomes a template for the next deployment.

This is the difference between a tool that automates work and a platform that accumulates organizational capability. In pharma analytics, where team turnover is real and pipeline complexity is high, that difference compounds.

The knowledge loop does not replace analytical judgment. It preserves it -- across people, across time, and across the inevitable organizational changes that would otherwise erase it.

---

**ai2analytics** is open source: [github.com/jamesyoung93/AI2Analytics](https://github.com/jamesyoung93/AI2Analytics)

Install with: `pip install git+https://github.com/jamesyoung93/AI2Analytics.git`
