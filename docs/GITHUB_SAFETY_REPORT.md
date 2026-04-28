# GitHub safety scan

## Scope

Verified that nothing in the working tree or git history exposes secrets,
PII, customer/client data, or personal paths.

## Working tree

Searched all tracked files for:

| Pattern class                                                   | Hits |
|-----------------------------------------------------------------|------|
| `password`, `secret`, `token`, `api_key`, `BEGIN PRIVATE KEY`   | 0 (only `max_tokens` in LLM client) |
| `sk-…`, `AKIA…`, `ghp_…`, `xox[baprs]-…` (key prefixes)         | 0 |
| Personal paths: `C:\Users\Admin`, `/Users/Admin`, `/home/...`   | 0 |
| Personal email: `@gmail`, `@yahoo`, `james.young`, `sdstate`    | 0 |
| Customer/client names                                           | 0 |

The working tree is clean.

## Git history

`git log --all --pretty="" --name-only | sort -u` shows that **only source
files were ever committed** (R/Python source, README, demos, blog markdown,
sanity-check scripts). No `.csv`, `.env`, `.pem`, `.key`, `.json` data
files have ever entered the history.

`git log --all -p -S '<token>'` searched for:
- `sk-`, `BEGIN PRIVATE`, `AKIA`, `C:\Users`, `james.young`, `ZS`

All returned zero matches in any historical blob.

## Files intentionally excluded by `.gitignore`

These exist in the working tree of the parent repo but are **never tracked**:

| Path                              | Why it's excluded                          | Verified ignored |
|-----------------------------------|--------------------------------------------|------------------|
| `generic_detail_approach.txt`     | Original 1500-line working notebook; mentions "ZS" (real consultancy name) and contains pre-refactor specifics | yes |
| `iterative_html_builder_v4.py`    | Internal Databricks tool, not part of this package | yes |
| `dist/`                           | Built wheel/tarball artifacts | yes |
| `demos/data/`                     | Generated CSV outputs (regenerable from synthetic generators) | yes |
| `.ai2analytics/`                  | Local knowledge store JSONL files | yes |
| `__pycache__/`, `*.pyc`, `.venv/` | Standard Python cruft | yes |

`git check-ignore` confirms each pattern is in effect.

### Note on `generic_detail_approach.txt`

This file mentions "in prep for hand-off to ZS" -- a real consulting firm.
Even though it is properly gitignored, **be careful never to add it
explicitly via `git add -f`** or similar. If you ever need to commit any
content from it, redact "ZS" first.

## Findings

- **No sensitive content was found in the working tree.**
- **No sensitive content has ever been committed.**
- `.gitignore` covers the four categories of excluded files (working
  notebooks, build artifacts, generated data, knowledge stores).
- No history rewrites are needed.

## Recommendations

1. Before publishing release artifacts, double-check that the `dist/`
   contents (built from `pyproject.toml`) do not accidentally include
   notebook files, generated data, or local knowledge stores. The
   `[tool.setuptools.packages.find]` setting points at `src` so the package
   contents should be limited to source code.
2. Consider adding a pre-commit hook that scans for the same patterns this
   report checked. (E.g. `gitleaks`, `detect-secrets`, or a custom regex
   list.)
