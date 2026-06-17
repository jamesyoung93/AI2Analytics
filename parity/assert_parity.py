"""Compare Python and R outputs for the canonical Market Mix demo.

Run after both run_python.py and run_r.R have written their JSON files::

    python parity/assert_parity.py

Exits 0 on parity, 1 on failure. Prints a side-by-side report.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    here = Path(__file__).parent
    p = json.loads((here / "python_outputs.json").read_text())
    r = json.loads((here / "r_outputs.json").read_text())

    print("\n" + "=" * 64)
    print("R <-> Python parity report -- canonical Market Mix demo")
    print("=" * 64)

    failures: list[str] = []

    def cmp_scalar(name, py_val, r_val, abs_tol=None, rel_tol=None):
        diff = r_val - py_val
        rel = abs(diff) / max(abs(py_val), 1e-9)
        ok = True
        if abs_tol is not None and abs(diff) > abs_tol:
            ok = False
        if rel_tol is not None and rel > rel_tol:
            ok = False
        flag = "ok" if ok else "FAIL"
        print(f"  {name:<28s} py={py_val:>12.4f}  r={r_val:>12.4f}  "
              f"|d|={abs(diff):>10.4f}  rel={rel*100:>6.2f}%  [{flag}]")
        if not ok:
            failures.append(name)

    print("\n[Diagnostics]")
    cmp_scalar("R^2",          p["r_squared"],     r["r_squared"],     abs_tol=0.05)
    cmp_scalar("Adj. R^2",     p["adj_r_squared"], r["adj_r_squared"], abs_tol=0.05)
    cmp_scalar("MAPE (%)",     p["mape"],          r["mape"],          abs_tol=1.0)
    cmp_scalar("predicted_sum",p["predicted_sum"], r["predicted_sum"], rel_tol=0.001)

    print("\n[Media coefficients]")
    for ch in ("TV_SPEND", "DIGITAL_SPEND", "PRINT_SPEND"):
        cmp_scalar(ch, p["coefficients"][ch], r["coefficients"][ch],
                   rel_tol=0.20)  # 20% tolerance: solver differences are real

    print("\n[Per-channel ROI]")
    for ch in ("TV_SPEND", "DIGITAL_SPEND", "PRINT_SPEND"):
        cmp_scalar(ch, p["channel_roi"][ch], r["channel_roi"][ch],
                   rel_tol=0.20)

    # Channel ranking by ROI
    py_rank = sorted(p["channel_roi"], key=p["channel_roi"].get, reverse=True)
    r_rank  = sorted(r["channel_roi"], key=r["channel_roi"].get, reverse=True)
    print("\n[Channel ranking by ROI]")
    print(f"  python: {' > '.join(py_rank)}")
    print(f"  R     : {' > '.join(r_rank)}")
    if py_rank != r_rank:
        failures.append("channel_ranking")
        print("  [FAIL] rankings differ")
    else:
        print("  [ok] rankings match")

    print("\n" + "=" * 64)
    if failures:
        print(f"FAILURES ({len(failures)}): {', '.join(failures)}")
        print("=" * 64)
        return 1
    print("All parity checks passed.")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
