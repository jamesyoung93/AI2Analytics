# Demo stress-test report (Colab / Jupyter portability)

## Scope

Verified that every demo in `demos/notebooks/` runs end-to-end in a fresh
Python 3.13 environment, and identified what would break in Colab or local
Jupyter for an outside user.

## Existing demos (`demos/notebooks/*.py`)

All five legacy `.py` "notebook scripts" execute end-to-end with no errors
when run as `python demos/notebooks/<name>.py` from the repo root:

| Script                       | Result | Notes |
|------------------------------|--------|-------|
| `demo_segmentation_us.py`    | passes | KMeans + auto-k both work; CSVs written |
| `demo_segmentation_eu.py`    | passes | `method="auto"` correctly picks KMeans |
| `demo_market_mix.py`         | passes | R² 0.79, ROI ranking sensible |
| `demo_cross_region.py`       | passes | Knowledge store wires up |
| `demo_knowledge.py`          | passes | All retrievers produce non-empty blocks |

### Colab/Jupyter portability gaps in the legacy scripts

These would have prevented the legacy `.py` scripts from running in Colab
"out of the box":

1. **`__file__` not defined in Jupyter cells.** Every legacy script does
   `_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))`.
   This is a `NameError` when pasted into a notebook.
2. **`from demos.synthetic_us_hcp import generate_all`.** The `demos`
   directory is not packaged. After `pip install ai2analytics`, this import
   fails. A user would need to `git clone` and then `cd` into the repo first.
3. **Hard-coded `demos/data/...` output paths.** Assume a writable directory
   relative to the repo root. Fine when running scripts; brittle in Colab.

### Fixes applied

- **Bundled synthetic generators inside the package** at
  `ai2analytics/datasets/{us_hcp,eu_account,bric}.py`. Now any environment
  with `pip install ai2analytics` can `from ai2analytics.datasets import
  us_hcp`. The original `demos/synthetic_*.py` files are kept (they still
  work as standalone scripts).
- **Fixed a real bug in
  `src/ai2analytics/templates/detail_optimization/output.py`** where
  `os.makedirs(os.path.dirname(cfg.output_csv))` would raise
  `FileNotFoundError: [WinError 3] The system cannot find the path
  specified: ''` when `output_csv` was a bare filename. Changed to
  `os.makedirs(os.path.dirname(cfg.output_csv) or ".", exist_ok=True)`,
  matching the pattern already used by the market_mix output.

## New richer demos (`demos/notebooks/*.ipynb`)

Five new Jupyter notebooks built from `demos/build_notebooks.py`:

| Notebook                                          | Story | Status |
|---------------------------------------------------|-------|--------|
| `00_quickstart.ipynb`                             | 5-cell hello world | executes ✓ |
| `01_one_template_three_regions.ipynb`             | One pipeline across US, EU, BRIC schemas | executes ✓ |
| `02_market_mix_budget_reallocation.ipynb`         | Fit MMM, then simulate budget reallocations using only quantities the pipeline already returns | executes ✓ |
| `03_call_optimizer_endtoend.ipynb`                | Full `DetailOptimizationPipeline` on synthetic US data with charts | executes ✓ |
| `04_knowledge_accumulates.ipynb`                  | Multi-quarter rollout simulation; retriever output | executes ✓ |

All notebooks were executed in-place via `jupyter nbconvert --execute`.

### Colab compatibility design

Each notebook starts with a self-installing setup cell:

```python
import sys, subprocess
try:
    import ai2analytics
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
                           'git+https://github.com/jamesyoung93/AI2Analytics.git'])
```

This is a no-op locally and pulls from GitHub in Colab. No `__file__`, no
hard-coded repo paths, no out-of-package imports. All synthetic data comes
from `ai2analytics.datasets`.

## Outputs

- 5 new `.ipynb` files (replaces no existing files; complements the legacy
  `.py` scripts).
- 1 helper `demos/build_notebooks.py` to regenerate them.
- 1 source bug fix in `detail_optimization/output.py`.
- 1 new `ai2analytics.datasets` subpackage bundling the three synthetic
  generators.
