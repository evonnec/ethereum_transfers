"""
Read the inputs, stream the transfer logs, write the metric CSVs and samples.

All parquet reading is streamed with column projection and filters pushed
down to the scan, so memory stays bounded by roughly one parquet row group no
matter how large the input is.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Final

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from ethereum_transfers.aggregate import INPUT_COLUMNS
from ethereum_transfers.transfers import erc20_transfer_filter

logger = logging.getLogger(__name__)

SAMPLE_ROWS: Final = 1_000_000
"""Default number of rows kept by `write_head_sample`."""

BATCH_ROWS: Final = 262_144
"""Maximum rows per record batch when streaming parquet input."""

REQUIRED_COLUMNS: Final = (*INPUT_COLUMNS, "TOPIC0", "TOPIC3")
"""Columns the logs must have: the filter reads two more than are returned."""

_ADDRESS_PATTERN: Final = r"^0x[0-9a-f]{40}$"


def read_token_addresses(path: Path) -> pa.Array[Any]:
    """
    Read the token list and return its distinct contract addresses.

    Every address listed in the file counts as an ERC-20 token, whether or
    not its ``name``, ``symbol`` or ``decimals`` are filled in (tokens such as
    MKR publish ``bytes32`` metadata that token lists often leave empty).

    Parameters
    ----------
    path : Path
        CSV file with an ``address`` column (``tokens.csv``).

    Returns
    -------
    pa.Array
        Distinct lower-case addresses, in first-seen order.

    Raises
    ------
    ValueError
        If the file has no ``address`` column or an address is not ``0x``
        followed by 40 hex digits.
    """
    try:
        table = pacsv.read_csv(
            path,
            convert_options=pacsv.ConvertOptions(
                include_columns=["address"],
                column_types={"address": pa.string()},
                strings_can_be_null=False,
            ),
        )
    except pa.ArrowKeyError as error:
        raise ValueError(f"{path}: the token list has no 'address' column") from error
    except pa.ArrowInvalid as error:
        raise ValueError(f"{path}: cannot read the token list: {error}") from error
    addresses = pc.utf8_lower(table["address"]).combine_chunks()
    valid = pc.match_substring_regex(addresses, _ADDRESS_PATTERN)
    if not pc.all(valid, min_count=0).as_py():
        first_bad = pc.index(valid, value=False).as_py()
        raise ValueError(
            f"{path}: record {first_bad + 1} has an invalid token address "
            f"{addresses[first_bad].as_py()!r}"
        )
    return pc.unique(addresses)


def open_logs(path: Path) -> ds.Dataset:
    """
    Open a parquet file or directory of parquet parts as a dataset.

    Raises
    ------
    FileNotFoundError
        If `path` does not exist.
    ValueError
        If `path` holds no parquet files (Spark markers do not count).
    """
    if not path.exists():
        raise FileNotFoundError(f"{path}: no such file or directory")
    dataset = ds.dataset(path, format="parquet")
    if not list(dataset.files):
        raise ValueError(f"{path}: no parquet files found")
    return dataset


def scan_transfers(
    logs_path: Path, token_addresses: pa.Array[Any], *, batch_rows: int = BATCH_ROWS
) -> Iterator[pa.RecordBatch]:
    """
    Stream the ERC-20 ``Transfer`` logs of the listed tokens.

    Only the columns ``BLOCK_DATE``, ``ADDRESS`` and ``DATA`` are returned,
    and the transfer filter is pushed down to the parquet scan; one file is
    read at a time.

    Parameters
    ----------
    logs_path : Path
        Decoded logs: a parquet file or a directory of parquet parts. Files
        whose names start with ``_`` or ``.`` (Spark markers) are ignored.
    token_addresses : pa.Array
        Lower-case token addresses, as returned by `read_token_addresses`.
    batch_rows : int, default `BATCH_ROWS`
        Maximum rows per batch.

    Yields
    ------
    pa.RecordBatch
        Matching logs, with the columns ``BLOCK_DATE``, ``ADDRESS``, ``DATA``.
    """
    dataset = open_logs(logs_path)
    missing = [name for name in REQUIRED_COLUMNS if name not in dataset.schema.names]
    if missing:
        raise ValueError(f"{logs_path}: missing column(s): {', '.join(missing)}")
    yield from dataset.to_batches(
        columns=list(INPUT_COLUMNS),
        filter=erc20_transfer_filter(token_addresses),
        batch_size=batch_rows,
        batch_readahead=1,
        fragment_readahead=1,
    )


def write_metric_csv(result: pd.DataFrame, path: Path) -> None:
    """
    Write a metric result as CSV.

    Parameters
    ----------
    result : pd.DataFrame
        Result of `ethereum_transfers.metrics.compute_metric`.
    path : Path
        Output file; its directory is created if missing.

    Notes
    -----
    No index is written. Amounts are written as plain base-10 integers of any
    size, so read them back as strings or `decimal.Decimal`, not floats.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(path, index=False, lineterminator="\n")


def write_head_sample(
    input_path: Path,
    output_path: Path,
    *,
    n_rows: int = SAMPLE_ROWS,
    batch_rows: int = BATCH_ROWS,
) -> int:
    """
    Write the first rows of a parquet dataset to a smaller file, streaming.

    Rows are taken in file-name order. Only one batch is held in memory, so
    this works on inputs far larger than memory.

    Parameters
    ----------
    input_path : Path
        Parquet file or directory of parquet parts.
    output_path : Path
        Gzip-compressed parquet file to write.
    n_rows : int, default 1_000_000
        Number of leading rows to keep.
    batch_rows : int, default `BATCH_ROWS`
        Maximum rows per batch (and per row group written).

    Returns
    -------
    int
        Number of rows written (fewer than `n_rows` if the input is smaller).
    """
    dataset = open_logs(input_path)
    written = 0
    with pq.ParquetWriter(output_path, dataset.schema, compression="gzip") as writer:
        batches = dataset.to_batches(
            batch_size=batch_rows, batch_readahead=1, fragment_readahead=1
        )
        for batch in batches:
            if written >= n_rows:
                break
            head = batch.slice(0, n_rows - written)
            writer.write_batch(head)
            written += head.num_rows
    logger.info("wrote the first %d rows of %s to %s", written, input_path, output_path)
    return written
