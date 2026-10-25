"""Find the most transferred ERC-20 token per day from decoded Ethereum logs.

Each metric is written to ``<output-dir>/<metric>.csv`` with the columns
``ADDRESS``, ``BLOCK_DATE`` and the metric, ordered by the metric descending
and limited to `TOP_N` rows:

``count_of_transfers``
    Number of log rows per token address per day.
``quantity_transferred``
    Sum of the transferred ``value`` per token address per day.

Examples
--------
Compute both metrics from the default input files into the current directory::

    python main.py

Choose the input files and output directory, or compute a single metric::

    python main.py run --logs logs.parquet --tokens tokens.csv --output-dir out
    python main.py run --metric count_of_transfers

Write a smaller parquet file for testing::

    python main.py sample --input eth_logs_decoded_2023_sample --rows 1000000
"""

import argparse
import json
import logging
from collections.abc import Iterable, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Final, assert_never

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_SOURCE_LOGS: Final = Path("eth_logs_decoded_2023_sample")
DEFAULT_LOGS: Final = Path("eth_logs_decoded_2023_sample_100.gzip")
DEFAULT_TOKENS: Final = Path("tokens.csv")
DEFAULT_OUTPUT_DIR: Final = Path()
SAMPLE_ROWS: Final = 1_000_000
TOP_N: Final = 100


class Metric(StrEnum):
    """Transfer metric used to rank token addresses per day."""

    COUNT_OF_TRANSFERS = "count_of_transfers"
    QUANTITY_TRANSFERRED = "quantity_transferred"


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
    """
    sample = pd.read_parquet(input_file).head(n_rows)
    sample.to_parquet(output_file, engine="pyarrow", compression="gzip")
    logger.info("wrote first %d rows of %s to %s", len(sample), input_file, output_file)


def join_tokens(logs: pd.DataFrame, tokens: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only the log rows emitted by a named token from the token list.

    Parameters
    ----------
    logs : pd.DataFrame
        Decoded logs with at least ``ADDRESS``, ``PARAMS`` and ``BLOCK_DATE``.
    tokens : pd.DataFrame
        Token list as read from ``tokens.csv`` (``address``, ``name``,
        ``symbol``, ...).

    Returns
    -------
    pd.DataFrame
        `logs` joined to `tokens` on the token address, with a fresh
        ``RangeIndex``.

    Notes
    -----
    Assumes every token in the list that has both a ``name`` and a ``symbol``
    is an ERC-20 token. Rows missing ``ADDRESS``, ``PARAMS``, ``BLOCK_DATE`` or
    a token ``name`` are dropped.
    """
    named_tokens = tokens.dropna(subset=["name", "symbol"], ignore_index=True).rename(
        columns={"address": "ADDRESS"}
    )
    return logs.merge(named_tokens, how="left", on="ADDRESS").dropna(
        subset=["ADDRESS", "PARAMS", "BLOCK_DATE", "name"], ignore_index=True
    )


def count_of_transfers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Find the token address with the most transfers on each day.

    Parameters
    ----------
    df : pd.DataFrame
        Token logs as returned by `join_tokens`.

    Returns
    -------
    pd.DataFrame
        Columns ``ADDRESS``, ``BLOCK_DATE`` and ``count_of_transfers``, one row
        per day, ordered by ``count_of_transfers`` descending, at most `TOP_N`
        rows.
    """
    count_df = (
        df.groupby(["ADDRESS", "BLOCK_DATE"]).size().to_frame("count_of_transfers")
    )
    # number the addresses within each day by count of transfers, descending
    count_df["row_num"] = (
        count_df.sort_values(by=["count_of_transfers", "BLOCK_DATE"], ascending=False)
        .groupby("BLOCK_DATE")
        .cumcount()
        + 1
    )
    top_transferred_address_per_day = count_df[count_df["row_num"] == 1].drop(
        columns="row_num"
    )
    result = top_transferred_address_per_day.sort_values(
        by="count_of_transfers", ascending=False
    ).head(TOP_N)
    return result.reset_index()


def quantity_transferred(df: pd.DataFrame) -> pd.DataFrame:
    """
    Find the token address with the largest quantity transferred on each day.

    Parameters
    ----------
    df : pd.DataFrame
        Token logs as returned by `join_tokens`.

    Returns
    -------
    pd.DataFrame
        Columns ``ADDRESS``, ``BLOCK_DATE`` and ``quantity_transferred``, one
        row per day, ordered by ``quantity_transferred`` descending, at most
        `TOP_N` rows.
    """
    transfers = df[["ADDRESS", "BLOCK_DATE", "PARAMS"]]
    # PARAMS is the JSON string of the decoded event arguments; for a transfer
    # its keys are ``from``, ``to`` and ``value`` (the amount transferred).
    # https://medium.com/@eiki1212/ethereum-transaction-structure-explained-aa5a94182ad6
    transfers["PARAMS"] = transfers["PARAMS"][1]
    transfers["PARAMS"] = transfers["PARAMS"].apply(json.loads)
    transfers["value_transferred"] = transfers["PARAMS"].apply(
        lambda params: params["value"]
    )
    # sum the value transferred per address per day and broadcast it to the rows
    # as a categorical (per-row codes into the per-group sums). Under pandas 3 a
    # ``str`` column -- what ``transform("sum")`` returns, and what sorting turns
    # an object column into -- stores its own copy of the group's sum in every
    # row. The categories are sorted, so the ordering below is unchanged.
    quantity = (
        transfers.groupby(["ADDRESS", "BLOCK_DATE"])["value_transferred"]
        .sum()
        .astype("category")
        .rename("quantity_transferred")
    )
    transfers = transfers.join(quantity, on=["ADDRESS", "BLOCK_DATE"])
    # number the rows within each day by quantity transferred, descending
    transfers["row_num"] = (
        transfers.sort_values(
            by=["BLOCK_DATE", "quantity_transferred"], ascending=False
        )
        .groupby("BLOCK_DATE")
        .cumcount()
        + 1
    )
    # back to plain values for the final ordering of the (one per day) top rows
    top_transferred_address_per_day = (
        transfers[transfers["row_num"] == 1]
        .drop(columns=["row_num", "PARAMS", "value_transferred"])
        .astype({"quantity_transferred": object})
    )
    result = top_transferred_address_per_day.sort_values(
        by="quantity_transferred", ascending=False
    ).head(TOP_N)
    return result.reset_index(drop=True)


def compute_metric(df: pd.DataFrame, metric: Metric) -> pd.DataFrame:
    """
    Compute the top token address per day for the given metric.

    Parameters
    ----------
    df : pd.DataFrame
        Token logs as returned by `join_tokens`.
    metric : Metric
        Metric to rank token addresses by.

    Returns
    -------
    pd.DataFrame
        Columns ``ADDRESS``, ``BLOCK_DATE`` and a column named after `metric`.
    """
    match metric:
        case Metric.COUNT_OF_TRANSFERS:
            return count_of_transfers(df)
        case Metric.QUANTITY_TRANSFERRED:
            return quantity_transferred(df)
        case _:
            assert_never(metric)


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


def build_parser() -> argparse.ArgumentParser:
    """
    Build the command-line parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser with ``run`` (the default when no command is given) and
        ``sample`` subcommands.
    """
    parser = argparse.ArgumentParser(
        description="Find the most transferred ERC-20 token per day.",
    )
    subparsers = parser.add_subparsers(
        dest="command", metavar="COMMAND", title="commands"
    )

    run = subparsers.add_parser(
        "run",
        help="compute the metrics and write one CSV per metric (default)",
        description="Compute the metrics and write one CSV per metric.",
    )
    run.add_argument(
        "--logs",
        type=Path,
        default=DEFAULT_LOGS,
        metavar="PATH",
        help="decoded logs parquet file or directory (default: %(default)s)",
    )
    run.add_argument(
        "--tokens",
        type=Path,
        default=DEFAULT_TOKENS,
        metavar="PATH",
        help="token list CSV file (default: %(default)s)",
    )
    run.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        metavar="DIR",
        help="directory to write the CSV files into (default: current directory)",
    )
    run.add_argument(
        "-m",
        "--metric",
        dest="metrics",
        nargs="+",
        choices=[metric.value for metric in Metric],
        default=list(Metric),
        metavar="METRIC",
        help="metrics to compute, from: %(choices)s (default: all)",
    )

    sample = subparsers.add_parser(
        "sample",
        help="write the first rows of a parquet dataset to a smaller file",
        description="Write the first rows of a parquet dataset to a smaller file.",
    )
    sample.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_SOURCE_LOGS,
        metavar="PATH",
        help="parquet file or directory to read (default: %(default)s)",
    )
    sample.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_LOGS,
        metavar="PATH",
        help="gzip-compressed parquet file to write (default: %(default)s)",
    )
    sample.add_argument(
        "-n",
        "--rows",
        type=int,
        default=SAMPLE_ROWS,
        help="number of leading rows to keep (default: %(default)s)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """
    Run the command-line interface.

    Parameters
    ----------
    argv : Sequence[str] or None, optional
        Command-line arguments without the program name; defaults to
        ``sys.argv[1:]``.

    Returns
    -------
    int
        Process exit status.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        args = parser.parse_args(["run"])
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    match args.command:
        case "run":
            generate_output(
                logs_path=args.logs,
                tokens_path=args.tokens,
                output_dir=args.output_dir,
                metrics=[Metric(metric) for metric in args.metrics],
            )
        case "sample":
            write_new_output(args.input, args.output, n_rows=args.rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
