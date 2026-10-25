"""Find the most transferred ERC-20 token per day from decoded Ethereum logs.

Each metric is written to ``<output-dir>/<metric>.csv`` with the columns
``ADDRESS``, ``BLOCK_DATE`` and the metric, ordered by the metric descending
and limited to `TOP_N` rows:

``count_of_transfers``
    Number of log rows per token address per day.
``quantity_transferred``
    Sum of the transferred ``value`` per token address per day.

Run it with the ``ethereum-transfers`` command (or ``python -m
ethereum_transfers``), or call `generate_output` from Python.
"""

from importlib.metadata import version

from ethereum_transfers.files import generate_output, write_new_output
from ethereum_transfers.metrics import (
    TOP_N,
    Metric,
    compute_metric,
    count_of_transfers,
    join_tokens,
    quantity_transferred,
)

__version__ = version("ethereum-transfers")

__all__ = [
    "TOP_N",
    "Metric",
    "__version__",
    "compute_metric",
    "count_of_transfers",
    "generate_output",
    "join_tokens",
    "quantity_transferred",
    "write_new_output",
]
