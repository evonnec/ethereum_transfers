"""
Compute the metrics from the input files and write one CSV per metric.

One streamed pass over the logs produces the per-``(day, token)`` aggregate,
and every metric is ranked from that single aggregate.
"""

import logging
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from ethereum_transfers.aggregate import aggregate_transfers
from ethereum_transfers.files import (
    read_token_addresses,
    scan_transfers,
    write_metric_csv,
)
from ethereum_transfers.metrics import TOP_N, Metric, compute_metric
from ethereum_transfers.rank import DEFAULT_PER_DAY

logger = logging.getLogger(__name__)


def aggregate_inputs(logs_path: Path, tokens_path: Path) -> pd.DataFrame:
    """
    Aggregate the ERC-20 transfers of the listed tokens per day and token.

    Parameters
    ----------
    logs_path : Path
        Decoded logs parquet file or directory.
    tokens_path : Path
        Token list CSV file.

    Returns
    -------
    pd.DataFrame
        See `ethereum_transfers.aggregate.combine_partials`.
    """
    token_addresses = read_token_addresses(tokens_path)
    logger.info("%d distinct token addresses in %s", len(token_addresses), tokens_path)
    aggregate = aggregate_transfers(scan_transfers(logs_path, token_addresses))
    logger.info(
        "%d ERC-20 transfers of %d tokens over %d days (%d day/token groups)",
        int(aggregate["transfer_count"].sum()),
        aggregate["token_address"].nunique(),
        aggregate["day"].nunique(),
        len(aggregate),
    )
    return aggregate


def generate_output(  # noqa: PLR0913 - keyword-only options of one pipeline run
    *,
    logs_path: Path,
    tokens_path: Path,
    output_dir: Path,
    metrics: Iterable[Metric],
    per_day: int | None = DEFAULT_PER_DAY,
    limit: int | None = TOP_N,
) -> list[Path]:
    """
    Generate one output CSV per metric from the input files.

    Parameters
    ----------
    logs_path : Path
        Decoded logs parquet file or directory.
    tokens_path : Path
        Token list CSV file.
    output_dir : Path
        Directory to write ``<metric>.csv`` files into; created if missing.
    metrics : Iterable[Metric]
        Metrics to generate, in order.
    per_day : int or None, default 1
        Tokens kept per day; ``None`` ranks all ``(day, token)`` pairs together.
    limit : int or None, default `TOP_N`
        Maximum rows per CSV; ``None`` for no limit.

    Returns
    -------
    list[Path]
        Paths of the CSV files written, in the order of `metrics`.
    """
    aggregate = aggregate_inputs(logs_path, tokens_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for metric in metrics:
        result = compute_metric(aggregate, metric, per_day=per_day, limit=limit)
        output_file = output_dir / f"{metric}.csv"
        write_metric_csv(result, output_file)
        logger.info("wrote %d rows to %s", len(result), output_file)
        written.append(output_file)
    return written
