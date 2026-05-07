# GitHub safety scan

## Scope

Verified that nothing in the working tree or git history exposes secrets,
PII, customer or client data, or hardcoded user-specific paths.

## Working tree

Searched all tracked files for:

| Pattern class                                                              | Hits |
|----------------------------------------------------------------------------|------|
| `password`, `secret`, `token`, `api_key`, `BEGIN PRIVATE KEY`              | 0 (only `max_tokens` in the LLM client) |
| Common credential prefixes (`sk-…`, `AKIA…`, `ghp_…`, `gho_…`, `xox[baprs]-…`) | 0 |
| Hardcoded home directory paths (Windows, macOS, Linux)                     | 0 |
| Author or affiliation strings                                              | 0 |
| Customer or client names                                                   | 0 |

The working tree is clean.

## Git history

`git log --all --pretty="" --name-only | sort -u` shows that only source
files were ever committed (R and Python source, README, demos, blog
markdown, sanity-check scripts). No `.csv`, `.env`, `.pem`, `.key`, or
`.json` data file has ever entered history.

`git log --all -p -S '<token>'` was run for the same credential prefixes,
home-directory paths, and author strings listed above. Every search
returned zero matches in any historical blob.

## Files intentionally excluded by `.gitignore`

These exist in the working tree of the parent repo but are never tracked:

| Path                              | Why it is excluded                                           | Verified ignored |
|-----------------------------------|--------------------------------------------------------------|------------------|
| `generic_detail_approach.txt`     | Original working notebook with pre-refactor references that should not be public. | yes |
| `iterative_html_builder_v4.py`    | Internal Databricks utility, not part of this package.       | yes |
| `dist/`                           | Built wheel and tarball artifacts.                           | yes |
| `demos/data/`                     | Generated CSV outputs, regenerable from the synthetic generators. | yes |
| `.ai2analytics/`                  | Local knowledge-store JSONL files.                           | yes |
| `parity/{shared_mmm.csv, _*.csv, *.json}` | Regenerable parity-check artifacts.                  | yes |
| `__pycache__/`, `*.pyc`, `.venv/` | Standard Python cruft.                                       | yes |

`git check-ignore` confirms each pattern is in effect.

### Operational note

Two of the gitignored files (`generic_detail_approach.txt` and
`iterative_html_builder_v4.py`) contain references that should never be
exposed publicly. Do not bypass `.gitignore` for these files
(`git add -f`, force-add via tooling, etc.). If any portion of their
content needs to be reused in this package, copy and redact first.

## Findings

- No sensitive content was found in the working tree.
- No sensitive content has ever been committed.
- `.gitignore` covers working notebooks, build artifacts, generated data,
  local knowledge stores, and parity-check artifacts.
- No history rewrites are needed.

## Recommendations

1. Before publishing release artifacts, double-check that the `dist/`
   contents (built from `pyproject.toml`) do not accidentally include
   notebook files, generated data, or local knowledge stores. The
   `[tool.setuptools.packages.find]` setting points at `src/`, so the
   wheel should be source-only by construction.
2. Consider adding a pre-commit secret scanner (`gitleaks`,
   `detect-secrets`, or an equivalent) to run the same checks
   automatically.
