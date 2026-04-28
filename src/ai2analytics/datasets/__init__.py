"""Bundled synthetic pharma datasets for examples and tests.

Three generators produce realistic fake commercial datasets that match
the schemas the ai2analytics pipelines expect. All identifiers are
randomly generated -- no real PII.

Usage::

    from ai2analytics.datasets import us_hcp, eu_account, bric

    tables = us_hcp.generate_all()        # dict of pandas DataFrames
    print(tables["hcp_reference"].head())

The data is generated in memory by default; pass ``output_dir=`` to
also write CSVs to disk.
"""

from ai2analytics.datasets import bric, eu_account, us_hcp

__all__ = ["us_hcp", "eu_account", "bric"]
