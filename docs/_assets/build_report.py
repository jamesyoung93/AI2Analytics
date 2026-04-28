"""Assemble docs/AI2Analytics_Report.html from the static template plus
embedded base64 images extracted from the executed notebooks.

Run from repo root:
    python docs/_assets/build_report.py
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ASSETS = REPO / "docs" / "_assets"
OUT = REPO / "docs" / "AI2Analytics_Report.html"


def b64(stem: str) -> str:
    """Read a base64 payload file and return a data: URL."""
    return "data:image/png;base64," + (ASSETS / f"{stem}.b64").read_text().strip()


HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AI2Analytics &mdash; A reusable analytics framework for pharma commercial teams</title>
<style>
  :root {
    --bg: #ffffff;
    --ink: #1a202c;
    --muted: #4a5568;
    --soft: #f7fafc;
    --line: #e2e8f0;
    --accent: #2b6cb0;
    --accent-soft: #ebf4fb;
    --teal: #2c7a7b;
    --teal-soft: #e6fffa;
    --plum: #6b46c1;
    --plum-soft: #f3eeff;
    --amber: #b7791f;
    --amber-soft: #fffaf0;
    --green: #2f855a;
    --green-soft: #f0fff4;
    --code-bg: #f7fafc;
    --code-ink: #2d3748;
    --shadow: 0 1px 3px rgba(0,0,0,0.05), 0 4px 14px rgba(0,0,0,0.04);
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                 "Helvetica Neue", Arial, sans-serif;
    color: var(--ink);
    background: var(--bg);
    line-height: 1.6;
    font-size: 16px;
  }
  .layout {
    display: grid;
    grid-template-columns: 230px 1fr;
    max-width: 1280px;
    margin: 0 auto;
  }
  nav.toc {
    position: sticky;
    top: 0;
    align-self: start;
    height: 100vh;
    overflow-y: auto;
    padding: 36px 18px 36px 28px;
    background: var(--soft);
    border-right: 1px solid var(--line);
  }
  nav.toc .brand {
    font-size: 17px;
    font-weight: 700;
    letter-spacing: -0.01em;
    margin-bottom: 4px;
  }
  nav.toc .brand-sub {
    font-size: 12px;
    color: var(--muted);
    margin-bottom: 22px;
  }
  nav.toc h2 {
    margin: 0 0 12px;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--muted);
    font-weight: 600;
  }
  nav.toc ol { list-style: none; padding: 0; margin: 0; counter-reset: section; }
  nav.toc li { counter-increment: section; margin: 0 0 4px; }
  nav.toc a {
    display: block;
    padding: 6px 8px;
    color: var(--muted);
    text-decoration: none;
    border-radius: 4px;
    font-size: 14px;
  }
  nav.toc a:hover { background: #edf2f7; color: var(--ink); }
  nav.toc a.active { background: #fff; color: var(--accent); font-weight: 600; }

  main { padding: 48px 56px 80px; max-width: 980px; }

  /* Hero */
  header.hero {
    margin: 0 0 56px;
  }
  .eyebrow {
    display: inline-block;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--accent);
    background: var(--accent-soft);
    padding: 4px 10px;
    border-radius: 12px;
    font-weight: 600;
    margin-bottom: 16px;
  }
  header.hero h1 {
    font-size: 38px;
    line-height: 1.15;
    margin: 0 0 14px;
    letter-spacing: -0.02em;
  }
  header.hero .lede {
    font-size: 17px;
    color: var(--muted);
    max-width: 70ch;
    margin: 0 0 28px;
    line-height: 1.6;
  }
  .hero-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 14px;
  }
  .what-tile {
    background: #fff;
    border: 1px solid var(--line);
    border-radius: 10px;
    padding: 16px 16px 14px;
    display: flex;
    gap: 12px;
    align-items: flex-start;
  }
  .what-tile .icon {
    flex: 0 0 32px;
    height: 32px;
    border-radius: 8px;
    background: var(--accent-soft);
    color: var(--accent);
    display: flex;
    align-items: center;
    justify-content: center;
  }
  .what-tile .icon.teal  { background: var(--teal-soft);  color: var(--teal);  }
  .what-tile .icon.plum  { background: var(--plum-soft);  color: var(--plum);  }
  .what-tile .icon.amber { background: var(--amber-soft); color: var(--amber); }
  .what-tile .label {
    font-weight: 600; font-size: 14px; margin-bottom: 2px;
  }
  .what-tile .desc {
    font-size: 13.5px; color: var(--muted); line-height: 1.45;
  }

  /* Sections */
  section { margin: 0 0 64px; scroll-margin-top: 12px; }
  section > h2 {
    font-size: 24px;
    margin: 0 0 8px;
    letter-spacing: -0.01em;
  }
  section > .section-lede {
    color: var(--muted);
    margin: 0 0 24px;
    max-width: 70ch;
  }
  h3 {
    font-size: 17px;
    margin: 26px 0 10px;
  }

  /* Capability cards */
  .cap {
    background: #fff;
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 26px 26px 22px;
    margin: 0 0 22px;
    box-shadow: var(--shadow);
  }
  .cap-head {
    display: flex;
    align-items: flex-start;
    gap: 16px;
    margin-bottom: 8px;
  }
  .cap-head .badge {
    flex: 0 0 44px;
    height: 44px;
    border-radius: 10px;
    background: var(--accent-soft);
    color: var(--accent);
    display: flex;
    align-items: center;
    justify-content: center;
  }
  .cap.teal  .cap-head .badge { background: var(--teal-soft);  color: var(--teal);  }
  .cap.plum  .cap-head .badge { background: var(--plum-soft);  color: var(--plum);  }
  .cap.amber .cap-head .badge { background: var(--amber-soft); color: var(--amber); }
  .cap-head .title {
    font-size: 19px; font-weight: 600; letter-spacing: -0.01em; margin: 0;
  }
  .cap-head .sub {
    font-size: 13px; color: var(--muted); margin-top: 2px;
  }
  .cap-grid {
    display: grid;
    grid-template-columns: 1.1fr 1fr;
    gap: 24px;
    margin-top: 14px;
    align-items: start;
  }
  .cap-grid.single { grid-template-columns: 1fr; }
  .cap-grid p { margin: 0 0 12px; }
  .qa { margin: 0; padding: 0; list-style: none; }
  .qa li {
    font-size: 14px; padding: 6px 0 6px 22px; position: relative; color: var(--muted);
  }
  .qa li::before {
    content: ""; position: absolute; left: 6px; top: 14px;
    width: 6px; height: 6px; border-radius: 3px; background: var(--accent);
  }
  .cap.teal .qa li::before  { background: var(--teal); }
  .cap.plum .qa li::before  { background: var(--plum); }
  .cap.amber .qa li::before { background: var(--amber); }

  /* Code blocks */
  pre {
    background: var(--code-bg);
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 16px 18px;
    overflow-x: auto;
    font-size: 13px;
    line-height: 1.55;
    color: var(--code-ink);
    margin: 0;
  }
  pre code { background: none; padding: 0; font-size: inherit; }
  code {
    font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
    font-size: 13.5px;
    background: var(--code-bg);
    color: var(--code-ink);
    padding: 1px 6px;
    border-radius: 3px;
  }

  /* Demo cards */
  .demos { display: grid; grid-template-columns: 1fr; gap: 22px; }
  .demo {
    background: #fff;
    border: 1px solid var(--line);
    border-radius: 12px;
    overflow: hidden;
    box-shadow: var(--shadow);
  }
  .demo-body {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0;
  }
  .demo-text {
    padding: 22px 24px 22px;
  }
  .demo-art {
    background: var(--soft);
    padding: 18px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    border-left: 1px solid var(--line);
    min-height: 240px;
  }
  .demo-art img { max-width: 100%; height: auto; border-radius: 6px;
                  box-shadow: 0 1px 4px rgba(0,0,0,0.06); }
  .demo-tag {
    display: inline-block;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: var(--muted);
    background: var(--soft);
    padding: 2px 8px;
    border-radius: 10px;
    margin-bottom: 8px;
    border: 1px solid var(--line);
  }
  .demo h3 { margin: 6px 0 8px; font-size: 18px; }
  .demo .demo-meta {
    font-size: 12.5px; color: var(--muted); margin: 0 0 14px;
  }
  .demo .demo-meta strong { color: var(--ink); font-weight: 600; }
  .demo p { font-size: 14.5px; color: var(--ink); margin: 0 0 12px; }
  .insight {
    background: var(--accent-soft);
    color: var(--accent);
    padding: 10px 14px;
    border-radius: 6px;
    font-size: 13.5px;
    margin: 12px 0 12px;
    border-left: 3px solid var(--accent);
  }
  .insight.teal { background: var(--teal-soft); color: var(--teal); border-color: var(--teal); }
  .insight.plum { background: var(--plum-soft); color: var(--plum); border-color: var(--plum); }
  .insight.amber{ background: var(--amber-soft); color: var(--amber); border-color: var(--amber); }
  .insight strong { font-weight: 700; }
  .demo-art .caption {
    font-size: 11.5px; color: var(--muted); margin-top: 8px; text-align: center;
  }
  .demo-art .stat {
    font-size: 28px;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: var(--accent);
    line-height: 1;
  }
  .demo-art .stat-sub {
    font-size: 13px; color: var(--muted); margin-top: 6px; text-align: center;
  }
  .open {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 13px;
    color: var(--accent);
    text-decoration: none;
    padding: 6px 12px;
    border: 1px solid var(--accent);
    border-radius: 6px;
    margin-top: 4px;
  }
  .open:hover { background: var(--accent-soft); }
  .open svg { width: 12px; height: 12px; }

  /* Capability matrix */
  table { border-collapse: collapse; width: 100%; margin: 12px 0 22px; font-size: 14px; }
  th, td {
    border-bottom: 1px solid var(--line);
    padding: 10px 14px;
    text-align: left;
    vertical-align: middle;
  }
  thead th {
    background: var(--soft);
    color: var(--muted);
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    font-weight: 600;
    border-bottom: 2px solid var(--line);
  }
  td.center { text-align: center; }
  .check { color: var(--green); font-weight: 600; }
  .dash  { color: #cbd5e0; }
  .later { color: var(--amber); font-size: 12px; font-style: italic; }

  .callout {
    background: var(--soft);
    border-left: 3px solid var(--accent);
    padding: 14px 18px;
    border-radius: 0 6px 6px 0;
    margin: 18px 0 8px;
    font-size: 14px;
  }
  .callout p { margin: 0 0 8px; }
  .callout p:last-child { margin-bottom: 0; }

  /* Quick-start */
  .qs {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 18px;
    margin-top: 12px;
  }
  .qs .panel {
    background: #fff;
    border: 1px solid var(--line);
    border-radius: 10px;
    padding: 20px 22px;
  }
  .qs .panel h3 {
    font-size: 15px;
    margin: 0 0 12px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--muted);
    font-weight: 600;
  }

  footer {
    margin-top: 60px;
    padding-top: 18px;
    border-top: 1px solid var(--line);
    font-size: 12px;
    color: var(--muted);
  }

  @media (max-width: 880px) {
    .layout { grid-template-columns: 1fr; }
    nav.toc { position: static; height: auto; border-right: none; border-bottom: 1px solid var(--line); }
    main { padding: 24px; }
    .cap-grid, .demo-body, .qs { grid-template-columns: 1fr; }
    .demo-art { border-left: none; border-top: 1px solid var(--line); }
  }
</style>
</head>
<body>
<div class="layout">
<nav class="toc" aria-label="Table of contents">
  <div class="brand">AI2Analytics</div>
  <div class="brand-sub">v0.2.1 &middot; Python &amp; R</div>
  <h2>On this page</h2>
  <ol>
    <li><a href="#overview">Overview</a></li>
    <li><a href="#capabilities">Capabilities</a></li>
    <li><a href="#demos">Demos</a></li>
    <li><a href="#cross-language">Python &amp; R</a></li>
    <li><a href="#start">Get started</a></li>
  </ol>
</nav>

<main>

<!-- ===========================================================
     HERO
=========================================================== -->
<header class="hero" id="overview">
  <span class="eyebrow">Pharma commercial analytics</span>
  <h1>One pipeline, every brand.</h1>
  <p class="lede">
    AI2Analytics is a Python &amp; R framework for pharmaceutical
    commercial teams who keep rebuilding the same analytical pipeline
    &mdash; once per brand, once per region, once per data source.
    It separates pipeline <em>logic</em> from <em>data shape</em>:
    write the model once, configure it many times. The reference
    templates cover HCP segmentation, market mix modelling, and
    detail-call optimisation, with an LLM-assisted layer for
    onboarding new datasets.
  </p>

  <div class="hero-grid">

    <div class="what-tile">
      <div class="icon">
        <svg viewBox="0 0 20 20" width="18" height="18" fill="none" stroke="currentColor" stroke-width="1.6">
          <circle cx="6" cy="6" r="2.5"/><circle cx="14" cy="14" r="2.5"/>
          <circle cx="14" cy="6" r="2.5"/><circle cx="6" cy="14" r="2.5"/>
        </svg>
      </div>
      <div>
        <div class="label">Reusable templates</div>
        <div class="desc">Segmentation, MMM, and call optimisation, all driven by a single typed config.</div>
      </div>
    </div>

    <div class="what-tile">
      <div class="icon teal">
        <svg viewBox="0 0 20 20" width="18" height="18" fill="none" stroke="currentColor" stroke-width="1.6">
          <path d="M3 14V8m4 6V4m4 10v-7m4 7V6"/>
        </svg>
      </div>
      <div>
        <div class="label">Plain DataFrames in</div>
        <div class="desc">Pipelines accept pandas / tibbles directly. Spark optional, never required.</div>
      </div>
    </div>

    <div class="what-tile">
      <div class="icon plum">
        <svg viewBox="0 0 20 20" width="18" height="18" fill="none" stroke="currentColor" stroke-width="1.6">
          <rect x="3" y="4" width="14" height="12" rx="2"/>
          <path d="M3 8h14M7 4v12"/>
        </svg>
      </div>
      <div>
        <div class="label">Knowledge that compounds</div>
        <div class="desc">Every run logs config, data profile, and metrics. New runs draw on past ones.</div>
      </div>
    </div>

    <div class="what-tile">
      <div class="icon amber">
        <svg viewBox="0 0 20 20" width="18" height="18" fill="none" stroke="currentColor" stroke-width="1.6">
          <path d="M10 2v4m0 8v4M2 10h4m8 0h4M4.5 4.5l3 3m5 5l3 3m-3-11l-3 3m-5 5l-3 3"/>
        </svg>
      </div>
      <div>
        <div class="label">Available in two languages</div>
        <div class="desc">Native Python and R packages with the same API shape and verified numerical agreement.</div>
      </div>
    </div>

  </div>
</header>


<!-- ===========================================================
     CAPABILITIES
=========================================================== -->
<section id="capabilities">
  <h2>Capabilities</h2>
  <p class="section-lede">
    Each template is a thin layer over standard libraries (scikit-learn,
    glmnet, PuLP), wrapped in a config-first API so the same code runs
    across schemas. Outputs are plain DataFrames you can keep working with.
  </p>

  <!-- Segmentation -->
  <div class="cap">
    <div class="cap-head">
      <div class="badge">
        <svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="1.8">
          <circle cx="7" cy="9" r="3"/><circle cx="16" cy="8" r="2.5"/>
          <circle cx="10" cy="17" r="2.5"/><circle cx="18" cy="16" r="2"/>
        </svg>
      </div>
      <div>
        <h3 class="title">Entity segmentation</h3>
        <div class="sub"><code>SegmentationPipeline</code> &middot; KMeans, hierarchical, or auto-method with optional auto-k.</div>
      </div>
    </div>
    <div class="cap-grid">
      <div>
        <p>
          Cluster any entity table by any numeric features. Handles
          missing values (median/mean/zero/drop), normalisation
          (standard / minmax / robust), and optional PCA. Returns a
          tidy assignment table plus per-segment feature profiles and
          a silhouette score.
        </p>
        <ul class="qa">
          <li>Which HCPs are heavy IL-17 prescribers vs. mixed-class writers?</li>
          <li>Which accounts are low-volume but high-tier?</li>
          <li>Should we use 3, 4, or 5 segments &mdash; let the data decide?</li>
        </ul>
      </div>
<pre><code>from ai2analytics.templates.segmentation import (
    SegmentationConfig, SegmentationPipeline,
)

cfg = SegmentationConfig(
    col_entity_id   = "npi",
    feature_columns = ["IL_17_TRX_L12M", "IL_23_TRX_L12M"],
    n_segments      = 4,
    method          = "auto",   # picks kmeans vs hierarchical
    output_csv      = "segments.csv",
)
out = SegmentationPipeline().run(cfg, dataframes={"entity_data": hcps})
out.profiles    # per-segment feature means
out.assignments # one row per entity</code></pre>
    </div>
  </div>

  <!-- Market mix -->
  <div class="cap teal">
    <div class="cap-head">
      <div class="badge">
        <svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="1.8">
          <path d="M3 18l5-7 4 3 6-9"/><path d="M14 5h5v5"/>
        </svg>
      </div>
      <div>
        <h3 class="title">Market mix modelling</h3>
        <div class="sub"><code>MarketMixPipeline</code> &middot; adstock + saturation + ridge / lasso / OLS, with contribution decomposition and response curves.</div>
      </div>
    </div>
    <div class="cap-grid">
      <div>
        <p>
          Geometric adstock per channel, choice of Hill or log
          saturation, optional positivity constraint. Reports R&sup2;
          and MAPE, decomposes weekly sales into base / channel /
          control contributions, and emits a per-channel response curve
          you can use for budget what-if analyses.
        </p>
        <ul class="qa">
          <li>How much of last quarter's sales did Digital actually drive?</li>
          <li>What's the marginal ROI of a $1 increase in TV?</li>
          <li>If we shifted 5&nbsp;pp from Print to Digital, what happens?</li>
        </ul>
      </div>
<pre><code>from ai2analytics.templates.market_mix import (
    MarketMixConfig, MarketMixPipeline,
)

cfg = MarketMixConfig(
    col_date           = "WEEK_ENDING",
    col_response       = "SALES",
    media_columns      = ["TV_SPEND", "DIGITAL_SPEND", "PRINT_SPEND"],
    control_columns    = ["PRICE_INDEX", "DISTRIBUTION_PCT"],
    default_saturation = "log",
    model_type         = "ridge",
)
out = MarketMixPipeline().run(cfg, dataframes={"time_series": ts})
out.channel_summary  # spend, contribution, ROI, coefficient per channel
out.response_curves  # 50-pt grid per channel</code></pre>
    </div>
  </div>

  <!-- Call optimizer -->
  <div class="cap plum">
    <div class="cap-head">
      <div class="badge">
        <svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="1.8">
          <rect x="4" y="4" width="16" height="16" rx="2"/>
          <path d="M8 12l3 3 5-6"/>
        </svg>
      </div>
      <div>
        <h3 class="title">HCP call allocation</h3>
        <div class="sub"><code>DetailOptimizationPipeline</code> &middot; probability + depth + look-alike models, then a PuLP LP under territory budget constraints.</div>
      </div>
    </div>
    <div class="cap-grid">
      <div>
        <p>
          End-to-end pipeline: lag/rolling feature engineering, three
          gradient-boosting models trained with walk-forward backtesting,
          vectorised scenario scoring, and a linear programme that
          allocates calls across HCPs and territories under per-team
          budget caps, priority-target rules, and slack penalties.
          Outputs a long-format portfolio ready for the field team.
        </p>
        <ul class="qa">
          <li>Given 80 calls per territory per quarter, who should each rep visit, and how often?</li>
          <li>Which non-target HCPs look like high-value writers (the look-alike score)?</li>
          <li>How does the planned plan compare to last cycle's actuals?</li>
        </ul>
      </div>
<pre><code>from ai2analytics.templates.detail_optimization import (
    DetailOptimizationConfig, DetailOptimizationPipeline,
)

cfg = DetailOptimizationConfig(
    drug_name      = "BRAND_X",
    il_rx_columns  = [("IL_17_TRX_L12M", "IL_17_DECILE"),
                      ("IL_23_TRX_L12M", "IL_23_DECILE")],
    team_a_budget_per_territory = 80,
    output_csv     = "call_plan.csv",
)
results = DetailOptimizationPipeline().run(
    cfg, dataframes={...}  # 7 input tables
)
results.portfolio  # one row per HCP x team</code></pre>
    </div>
  </div>

  <!-- Knowledge -->
  <div class="cap amber">
    <div class="cap-head">
      <div class="badge">
        <svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="1.8">
          <path d="M4 5a2 2 0 012-2h12v18H6a2 2 0 01-2-2V5z"/>
          <path d="M8 7h8M8 11h8M8 15h5"/>
        </svg>
      </div>
      <div>
        <h3 class="title">Knowledge that accumulates</h3>
        <div class="sub"><code>DecisionStore</code> + <code>ContextStore</code> + <code>KnowledgeRetriever</code> &middot; JSONL or Spark Delta backend.</div>
      </div>
    </div>
    <div class="cap-grid">
      <div>
        <p>
          Every pipeline run is logged: config, data profile,
          auto-detected vs. user-answered fields, adapter code,
          outcome metrics, free-form tags. A separate context store
          holds curated patterns ("US tables use <code>npi</code>",
          "BRIC quarterly performance must be aggregated to one row
          per account before clustering"). The retriever combines
          both into a structured block for prompt injection or human
          reference.
        </p>
        <ul class="qa">
          <li>What did we configure last quarter that worked?</li>
          <li>What's the standard column mapping for EU data?</li>
          <li>What adapter code patterns have we used for this template before?</li>
        </ul>
      </div>
<pre><code>from ai2analytics.knowledge import (
    DecisionStore, DecisionRecord, ContextStore, KnowledgeRetriever,
)

decisions = DecisionStore(backend="json", path="decisions.jsonl")
context   = ContextStore(backend="json", path="context.jsonl")

decisions.log(DecisionRecord(
    template_name="segmentation",
    config_dict={"col_entity_id": "npi", "n_segments": 4},
    outcome_metrics={"silhouette_score": 0.61},
    tags=["us", "hcp", "q4-2025"],
))

retriever = KnowledgeRetriever(decisions, context)
print(retriever.retrieve_for_analysis(
    template_name="segmentation", scope={"region": "us"},
))</code></pre>
    </div>
  </div>

  <!-- Bundled datasets -->
  <h3 style="margin-top: 36px;">Bundled synthetic datasets</h3>
  <p>
    Three generators produce realistic fake commercial datasets matching the
    schemas the templates expect. No PII, deterministic given a seed,
    everything in memory by default.
  </p>
  <table>
    <thead>
      <tr><th>Generator</th><th>Market style</th><th>Cadence</th><th>Tables produced</th></tr>
    </thead>
    <tbody>
      <tr><td><code>us_hcp.generate_all()</code></td><td>US</td><td>Weekly</td><td>HCP reference, weekly activity, calls, two team alignments, portfolio decile, priority targets</td></tr>
      <tr><td><code>eu_account.generate_all()</code></td><td>EU</td><td>Monthly</td><td>Account reference, monthly performance, multi-channel visits, KAM &amp; medical alignment</td></tr>
      <tr><td><code>bric.generate_all()</code></td><td>BRIC</td><td>Quarterly</td><td>Account master, quarterly performance, engagement, sales alignment</td></tr>
    </tbody>
  </table>

  <!-- AnalyticsSession -->
  <h3 style="margin-top: 36px;">LLM-assisted onboarding (Python)</h3>
  <p>
    For a brand-new dataset where you don't yet know the column mappings,
    <code>AnalyticsSession</code> surveys your Spark catalog, profiles
    each table, asks an LLM to match discovered columns to the template's
    declared requirements, and generates adapter code where shapes don't
    line up. Conversation is structured around exact config field names,
    so anything the LLM doesn't auto-fill becomes a targeted question.
  </p>
<pre><code>from ai2analytics import AnalyticsSession
session = AnalyticsSession(spark=spark, llm_endpoint="your-endpoint")
session.discover(schemas=["commercial_data"], prompt="Optimize HCP call allocation")
session.show_questions()       # auto-detected vs. needs-answer
session.answer({"drug_name": "BRAND_X", "output_csv": "/dbfs/.../plan.csv"})
session.generate_adapter()     # LLM writes preprocessing code
session.run_adapter()          # executes; auto-retries with LLM fix on error
session.run()                  # runs the full pipeline</code></pre>
</section>


<!-- ===========================================================
     DEMOS
=========================================================== -->
<section id="demos">
  <h2>Demo notebooks</h2>
  <p class="section-lede">
    Five executable Jupyter notebooks under <code>demos/notebooks/</code>.
    Each opens with a self-installing setup cell &mdash; one click in
    Colab, no checkout required, all synthetic data bundled in the
    package.
  </p>

  <div class="demos">

    <!-- Demo 0: Quickstart -->
    <article class="demo">
      <div class="demo-body">
        <div class="demo-text">
          <span class="demo-tag">Hello world</span>
          <h3>00 &middot; Quickstart</h3>
          <p class="demo-meta"><strong>Dataset:</strong> 5,000 synthetic US HCPs &middot; <strong>Time to run:</strong> &lt;30s</p>
          <p>
            Five cells, end-to-end: generate synthetic prescribers,
            cluster them on two Rx-volume features, inspect the
            resulting segments, and view a PCA scatter. The
            shortest-possible "convince me in 60 seconds" tour.
          </p>
          <div class="insight">
            <strong>Result:</strong> 4 segments on 5,000 HCPs in &lt;1s.
            Silhouette&nbsp;0.6142 &mdash; clean separation between
            heavy IL-17 writers (mean 35 Rx) and the mixed group
            (16 IL-17, 18 IL-23).
          </div>
          <a class="open" href="https://colab.research.google.com/github/jamesyoung93/AI2Analytics/blob/master/demos/notebooks/00_quickstart.ipynb" target="_blank" rel="noopener">
            Open in Colab
            <svg viewBox="0 0 12 12" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M3 9l6-6M5 3h4v4"/></svg>
          </a>
        </div>
        <div class="demo-art">
          <img src="QUICKSTART_IMG" alt="Segment scatter and size chart from the quickstart notebook">
          <div class="caption">PCA scatter plus segment sizes &mdash; output of <code>pipeline.dashboard()</code>.</div>
        </div>
      </div>
    </article>

    <!-- Demo 1: Three regions -->
    <article class="demo">
      <div class="demo-body">
        <div class="demo-text">
          <span class="demo-tag">Schema portability</span>
          <h3>01 &middot; One pipeline, three regions</h3>
          <p class="demo-meta"><strong>Dataset:</strong> US HCPs (5,000), EU accounts (3,000), BRIC accounts (2,000) &middot; <strong>Time:</strong> ~30s</p>
          <p>
            The thesis demo. The <em>identical</em>
            <code>SegmentationPipeline</code> class clusters US
            individual prescribers, EU institutional accounts, and a
            joined BRIC quarterly view &mdash; three different schemas,
            three different entity types &mdash; with all differences
            absorbed into <code>SegmentationConfig</code>. The
            <code>auto</code> method picks KMeans for US/EU and
            hierarchical for BRIC.
          </p>
          <div class="insight teal">
            <strong>Result:</strong> US silhouette 0.61 (kmeans),
            EU 0.57 (kmeans), BRIC 0.52 (hierarchical) &mdash; all three
            in a healthy range with one pipeline class.
          </div>
          <a class="open" href="https://colab.research.google.com/github/jamesyoung93/AI2Analytics/blob/master/demos/notebooks/01_one_template_three_regions.ipynb" target="_blank" rel="noopener">
            Open in Colab
            <svg viewBox="0 0 12 12" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M3 9l6-6M5 3h4v4"/></svg>
          </a>
        </div>
        <div class="demo-art">
          <img src="THREE_REGIONS_IMG" alt="Side-by-side segment-size charts for US, EU, BRIC">
          <div class="caption">Segment sizes by region &mdash; same pipeline, three schemas.</div>
        </div>
      </div>
    </article>

    <!-- Demo 2: MMM -->
    <article class="demo">
      <div class="demo-body">
        <div class="demo-text">
          <span class="demo-tag">Actionable analytics</span>
          <h3>02 &middot; MMM with budget reallocation</h3>
          <p class="demo-meta"><strong>Dataset:</strong> 156 weeks of synthetic media + sales &middot; <strong>Time:</strong> ~10s</p>
          <p>
            Fits a market mix model on three years of weekly TV /
            Digital / Print spend, then runs a small reallocation grid
            using <em>only</em> quantities the pipeline already
            returns: per-channel coefficients and adstock decay from
            <code>channel_summary</code>, saturation shape from
            <code>response_curves</code>. No bespoke math &mdash; the
            answer falls out of the model.
          </p>
          <div class="insight teal">
            <strong>Result:</strong> Model reaches R&sup2; 0.81, MAPE
            5.05%. Reallocation search recommends moving Digital share
            from 27.5% to 25% and TV from 48.4% to 55% &mdash;
            <strong>+~48 sales/yr at the same total spend</strong>.
          </div>
          <a class="open" href="https://colab.research.google.com/github/jamesyoung93/AI2Analytics/blob/master/demos/notebooks/02_market_mix_budget_reallocation.ipynb" target="_blank" rel="noopener">
            Open in Colab
            <svg viewBox="0 0 12 12" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M3 9l6-6M5 3h4v4"/></svg>
          </a>
        </div>
        <div class="demo-art">
          <img src="MMM_IMG" alt="Actual vs. predicted sales over 156 weeks">
          <div class="caption">Actual (solid) vs. predicted (dashed) sales over 156 weeks.</div>
        </div>
      </div>
    </article>

    <!-- Demo 3: Call optimizer -->
    <article class="demo">
      <div class="demo-body">
        <div class="demo-text">
          <span class="demo-tag">End-to-end pipeline</span>
          <h3>03 &middot; Call optimizer end-to-end</h3>
          <p class="demo-meta"><strong>Dataset:</strong> 7-table synthetic US commercial dataset &middot; <strong>Time:</strong> ~60s</p>
          <p>
            The most ambitious demo. Runs the full
            <code>DetailOptimizationPipeline</code>: load &rarr; lag /
            rolling features &rarr; train probability + depth +
            look-alike models with walk-forward backtest &rarr; score
            every HCP at every call-count scenario &rarr; solve a PuLP
            linear programme under per-territory budgets and
            priority-target rules &rarr; long-format portfolio with
            audit columns. Plots calls per territory by team and the
            decile-uplift of the budget.
          </p>
          <div class="insight plum">
            <strong>Result:</strong> 4,266-row call plan covering 2,133
            unique HCPs across two teams. Allocation concentrates on
            higher Rx-volume deciles &mdash; the LP correctly
            prioritises high-value writers while satisfying territory
            budget caps.
          </div>
          <a class="open" href="https://colab.research.google.com/github/jamesyoung93/AI2Analytics/blob/master/demos/notebooks/03_call_optimizer_endtoend.ipynb" target="_blank" rel="noopener">
            Open in Colab
            <svg viewBox="0 0 12 12" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M3 9l6-6M5 3h4v4"/></svg>
          </a>
        </div>
        <div class="demo-art">
          <img src="OPTIMIZER_IMG" alt="Decile-uplift chart showing average calls per HCP by Rx-volume decile">
          <div class="caption">Average allocated calls per HCP by Rx-volume decile.</div>
        </div>
      </div>
    </article>

    <!-- Demo 4: Knowledge -->
    <article class="demo">
      <div class="demo-body">
        <div class="demo-text">
          <span class="demo-tag">Operational learning</span>
          <h3>04 &middot; Knowledge accumulates over runs</h3>
          <p class="demo-meta"><strong>Dataset:</strong> Four simulated quarterly rollouts (US Q3, EU Q3, US Q4, BRIC Q4) &middot; <strong>Time:</strong> &lt;5s</p>
          <p>
            Walks through the knowledge layer: log four pipeline runs
            into a JSONL <code>DecisionStore</code>, curate a handful
            of context entries, then call
            <code>KnowledgeRetriever.retrieve_for_analysis()</code>
            scoped to a fresh region. The retriever surfaces the
            BRIC-specific aggregation pattern that emerged from the
            one prior BRIC run &mdash; the kind of tribal knowledge
            that usually lives in one analyst's head.
          </p>
          <div class="insight amber">
            <strong>Result:</strong> 4 logged runs + 4 curated patterns
            yield a region-aware knowledge block ready for an LLM
            prompt or a new analyst's first read.
          </div>
          <a class="open" href="https://colab.research.google.com/github/jamesyoung93/AI2Analytics/blob/master/demos/notebooks/04_knowledge_accumulates.ipynb" target="_blank" rel="noopener">
            Open in Colab
            <svg viewBox="0 0 12 12" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M3 9l6-6M5 3h4v4"/></svg>
          </a>
        </div>
        <div class="demo-art">
          <div class="stat">4&nbsp;runs</div>
          <div class="stat-sub">across US / EU / BRIC,<br>4 curated context entries,<br>fed into one retriever block.</div>
        </div>
      </div>
    </article>

  </div>
</section>


<!-- ===========================================================
     CROSS-LANGUAGE
=========================================================== -->
<section id="cross-language">
  <h2>Use it from Python or R</h2>
  <p class="section-lede">
    Both packages are first-class citizens. The R port (<code>r-pkg/ai2analytics/</code>)
    uses idiomatic R6 + tibble &mdash; no Python interop required.
    Constructor names mirror the Python API, and the
    <code>dataframes = list(entity_data = df)</code> pattern is the
    same shape as Python's <code>dataframes={"entity_data": df}</code>.
  </p>

  <table>
    <thead>
      <tr>
        <th>Capability</th>
        <th class="center">Python</th>
        <th class="center">R</th>
        <th>Notes</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Entity segmentation</td>
        <td class="center"><span class="check">&#10003;</span></td>
        <td class="center"><span class="check">&#10003;</span></td>
        <td>Same constructor names, same options.</td>
      </tr>
      <tr>
        <td>Market mix modelling</td>
        <td class="center"><span class="check">&#10003;</span></td>
        <td class="center"><span class="check">&#10003;</span></td>
        <td>R uses <code>glmnet</code>; <code>alpha</code> argument is sklearn-equivalent.</td>
      </tr>
      <tr>
        <td>Knowledge stores &amp; retriever</td>
        <td class="center"><span class="check">&#10003;</span></td>
        <td class="center"><span class="check">&#10003;</span></td>
        <td>JSONL backend in both. Spark Delta backend Python-only.</td>
      </tr>
      <tr>
        <td>Synthetic datasets (US / EU / BRIC)</td>
        <td class="center"><span class="check">&#10003;</span></td>
        <td class="center"><span class="check">&#10003;</span></td>
        <td>Native generators in each language; equivalent shape, different RNG.</td>
      </tr>
      <tr>
        <td>HCP call optimisation</td>
        <td class="center"><span class="check">&#10003;</span></td>
        <td class="center"><span class="dash">&mdash;</span></td>
        <td><span class="later">Python only for now &mdash; the LP + RF stack is large; R port deferred.</span></td>
      </tr>
      <tr>
        <td>LLM-assisted session</td>
        <td class="center"><span class="check">&#10003;</span></td>
        <td class="center"><span class="dash">&mdash;</span></td>
        <td><span class="later">Python only &mdash; recommend wiring <code>httr2</code>/<code>ellmer</code> at the application layer in R.</span></td>
      </tr>
    </tbody>
  </table>

  <div class="callout">
    <p>
      <strong>Numerical agreement.</strong> The two implementations were
      tested on the same shared CSV running the same canonical Market
      Mix configuration: <strong>R&sup2; agrees to within 0.003</strong>,
      per-channel ROI to within 10%, and channel ranking is identical
      (TV &gt; Digital &gt; Print). A regression test in the R suite
      pins this so future changes can't drift unnoticed.
    </p>
  </div>

  <p style="margin-top: 18px;">The R API for the same MMM looks like this:</p>
<pre><code>library(ai2analytics)

cfg &lt;- MarketMixConfig(
  col_date           = "WEEK_ENDING",
  col_response       = "SALES",
  media_columns      = c("TV_SPEND", "DIGITAL_SPEND", "PRINT_SPEND"),
  control_columns    = c("PRICE_INDEX", "DISTRIBUTION_PCT"),
  default_saturation = "log",
  model_type         = "ridge"
)

out &lt;- MarketMixPipeline$new()$run(cfg, dataframes = list(time_series = ts))
out$channel_summary   # tibble with spend, contribution, ROI per channel</code></pre>

  <p>
    Four RMarkdown vignettes mirror the first, second, third, and fifth
    Python notebooks. Render any of them with
    <code>rmarkdown::render("vignettes/a_quickstart.Rmd")</code>.
  </p>
</section>


<!-- ===========================================================
     QUICK START
=========================================================== -->
<section id="start">
  <h2>Get started</h2>
  <p class="section-lede">
    Install one of the two packages, generate a bundled dataset, and run
    a pipeline. Both code paths fit on one screen.
  </p>

  <div class="qs">
    <div class="panel">
      <h3>Python</h3>
<pre><code>pip install git+https://github.com/jamesyoung93/AI2Analytics.git</code></pre>
<pre style="margin-top:10px;"><code>from ai2analytics.datasets import us_hcp
from ai2analytics.templates.segmentation import (
    SegmentationConfig, SegmentationPipeline,
)

dfs = us_hcp.generate_all(seed=42)
cfg = SegmentationConfig(
    col_entity_id   = "npi",
    feature_columns = ["IL_17_TRX_L12M", "IL_23_TRX_L12M"],
    n_segments      = 4,
    output_csv      = "segments.csv",
)
out = SegmentationPipeline().run(
    cfg, dataframes={"entity_data": dfs["hcp_reference"]}
)
out.profiles  # 4 segments x 2 features</code></pre>
    </div>

    <div class="panel">
      <h3>R</h3>
<pre><code>devtools::install_github("jamesyoung93/AI2Analytics",
                         subdir = "r-pkg/ai2analytics")</code></pre>
<pre style="margin-top:10px;"><code>library(ai2analytics)

dfs &lt;- generate_us_hcp(seed = 42)
cfg &lt;- SegmentationConfig(
  col_entity_id   = "npi",
  feature_columns = c("IL_17_TRX_L12M", "IL_23_TRX_L12M"),
  n_segments      = 4,
  output_csv      = "segments.csv"
)
out &lt;- SegmentationPipeline$new()$run(
  cfg, dataframes = list(entity_data = dfs$hcp_reference)
)
out$profiles  # tibble: 4 segments x 2 features</code></pre>
    </div>
  </div>

  <div class="callout" style="margin-top: 22px;">
    <p>
      <strong>Where to go next.</strong> If you have data and want to
      try the AI-assisted workflow, see
      <code>AnalyticsSession</code> in the README. If you want to add
      a new template, subclass <code>BaseTemplate</code> and declare
      <code>required_tables</code> with <code>config_field</code>
      links &mdash; the discovery and conversation layers will wire your
      data automatically.
    </p>
  </div>
</section>

<footer>
  <p>
    AI2Analytics is open source under the MIT license.
    Source: <code>github.com/jamesyoung93/AI2Analytics</code>.
    This page is built from the executed demo notebooks; numbers and
    images are real outputs, not mockups.
  </p>
</footer>

</main>
</div>

<script>
const links = document.querySelectorAll('nav.toc a');
const sections = [...links].map(l => document.querySelector(l.getAttribute('href')));
const observer = new IntersectionObserver(entries => {
  for (const e of entries) {
    if (e.isIntersecting) {
      const id = '#' + e.target.id;
      links.forEach(a => a.classList.toggle('active', a.getAttribute('href') === id));
    }
  }
}, { rootMargin: '-25% 0px -65% 0px', threshold: 0 });
sections.forEach(s => s && observer.observe(s));
</script>
</body>
</html>
"""


def main() -> None:
    html = HTML
    html = html.replace("QUICKSTART_IMG", b64("00_quickstart_img1"))
    html = html.replace("THREE_REGIONS_IMG", b64("01_one_template_three_regions_img1"))
    html = html.replace("MMM_IMG", b64("02_market_mix_budget_reallocation_img1"))
    html = html.replace("OPTIMIZER_IMG", b64("03_call_optimizer_endtoend_img2"))
    OUT.write_text(html, encoding="utf-8")
    print(f"Wrote {OUT} ({OUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
