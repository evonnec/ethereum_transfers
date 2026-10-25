"""Read the input files, write the metric CSVs, and sample large inputs."""

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Final

import pandas as pd

from ethereum_transfers.metrics import Metric, compute_metric, join_tokens

logger = logging.getLogger(__name__)

SAMPLE_ROWS: Final = 1_000_000


def write_new_output(
    input_file: Path, output_file: Path, *, n_rows: int = SAMPLE_ROWS
) -> None:
    """
    Make a smaller file for testing purposes.

    Parameters
    ----------
    input_file : Path
        Parquet file or directory to read.
    output_file : Path
        Gzip-compressed parquet file to write.
    n_rows : int, default 1_000_000
        Number of leading rows of `input_file` to keep.

    Notes
    -----
    Reads all of `input_file` into memory before keeping the first rows.
    """
    sample = pd.read_parquet(input_file).head(n_rows)
    sample.to_parquet(output_file, engine="pyarrow", compression="gzip")
    logger.info("wrote first %d rows of %s to %s", len(sample), input_file, output_file)


def generate_output(
    *,
    logs_path: Path,
    tokens_path: Path,
    output_dir: Path,
    metrics: Iterable[Metric],
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

    Returns
    -------
    list[Path]
        Paths of the CSV files written.
    """
    df = join_tokens(pd.read_parquet(logs_path), pd.read_csv(tokens_path))
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for metric in metrics:
        result = compute_metric(df, metric)
        logger.info("top token address per day by %s:\n%s", metric, result)
        output_file = output_dir / f"{metric}.csv"
        result.to_csv(output_file, index=False)
        logger.info("wrote %d rows to %s", len(result), output_file)
        written.append(output_file)
    return written
