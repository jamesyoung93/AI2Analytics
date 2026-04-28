# ai2analytics (R)

Native R port of the [Python AI2Analytics](../../README.md) framework.

Reusable pipeline templates for **entity segmentation** and **market mix
modelling**, plus a JSONL-backed **knowledge store** for accumulating
decisions across runs. Every pipeline separates analytical logic from data
shape, so the same template handles different brands, regions, and schemas
via configuration.

## Install

```r
# install.packages("devtools")
devtools::install_github("jamesyoung93/AI2Analytics", subdir = "r-pkg/ai2analytics")
```

Required CRAN packages: R6, cluster, dplyr, glmnet, jsonlite, rlang, tibble.

## Quickstart

```r
library(ai2analytics)

dfs <- generate_us_hcp(seed = 42)
cfg <- SegmentationConfig(
  analysis_name   = "demo",
  col_entity_id   = "npi",
  feature_columns = c("IL_17_TRX_L12M", "IL_23_TRX_L12M"),
  n_segments      = 4
)
out <- SegmentationPipeline$new()$run(cfg, dataframes = list(entity_data = dfs$hcp_reference))
head(out$assignments)
```

## Templates

| Template               | Class                  | Reading |
|------------------------|------------------------|---------|
| Entity segmentation    | `SegmentationPipeline` | KMeans / hierarchical / auto-method, optional auto-k, normalisation, missing-value handling. |
| Market mix model       | `MarketMixPipeline`    | Adstock + saturation + ridge/lasso/OLS, contribution decomposition, response curves, ROI. |

## Knowledge

| Class                | Purpose |
|----------------------|---------|
| `DecisionStore`      | Append-only JSONL log of pipeline runs. |
| `ContextStore`       | Synthesised patterns and best practices. |
| `KnowledgeRetriever` | Format both for prompt injection or human reference. |

## Synthetic data

Three generators produce realistic fake commercial datasets matching the
schemas the pipelines expect (no PII):

- `generate_us_hcp()` — US HCP-level, weekly cadence, 7 tables.
- `generate_eu_account()` — EU account-level, monthly, 5 tables.
- `generate_bric()` — BRIC account-level, quarterly, 4 tables.

## Vignettes

```r
browseVignettes("ai2analytics")
```

- `00_quickstart.Rmd` — minimal end-to-end.
- `01_one_template_three_regions.Rmd` — same pipeline, three schemas.
- `02_market_mix_budget_reallocation.Rmd` — fit MMM, then simulate budget shifts.
- `03_knowledge_accumulates.Rmd` — multi-quarter rollout & retriever output.

## Differences from the Python package

This R port covers segmentation, market mix, knowledge, and synthetic data.
The Python `detail_optimization` template (HCP call allocation with PuLP) is
not yet ported — that template is large and tightly coupled to the PuLP LP
solver. Recommend using R `ompr` or `lpSolveAPI` for an R port.

The Python `AnalyticsSession` LLM-orchestrator is also not ported. The R
package is designed to be called directly; LLM integration is left to the
caller (`httr` / `ellmer` / `chattr` etc.).
