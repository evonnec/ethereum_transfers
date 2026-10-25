"""Find the most transferred ERC-20 token per day from decoded Ethereum logs.

A transfer is an ERC-20 ``Transfer`` log (``TOPIC0`` is the ``Transfer``
signature, ``TOPIC3`` is null and ``DATA`` is one 32-byte word) emitted by a
token listed in ``tokens.csv``. Each metric is written to
``<output-dir>/<metric>.csv`` with the columns ``day``, ``token_address`` and
the metric: the top token of each day, ordered by the metric descending and
limited to `TOP_N` rows.

``count_of_transfers``
    Number of transfers per token per day.
``quantity_transferred``
    Exact sum of the transferred amounts per token per day, in raw base units.

Run it with the ``ethereum-transfers`` command (or ``python -m
ethereum_transfers``), or call `generate_output` from Python.
"""

from importlib.metadata import version

from ethereum_transfers.aggregate import aggregate_transfers
from ethereum_transfers.files import (
    read_token_addresses,
    scan_transfers,
    write_head_sample,
)
from ethereum_transfers.metrics import TOP_N, Metric, compute_metric
from ethereum_transfers.pipeline import aggregate_inputs, generate_output
from ethereum_transfers.rank import rank_top

__version__ = version("ethereum-transfers")

__all__ = [
    "TOP_N",
    "Metric",
    "__version__",
    "aggregate_inputs",
    "aggregate_transfers",
    "compute_metric",
    "generate_output",
    "rank_top",
    "read_token_addresses",
    "scan_transfers",
    "write_head_sample",
]
