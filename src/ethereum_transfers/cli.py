"""Command-line interface: ``ethereum-transfers`` / ``python -m ethereum_transfers``.

Examples
--------
Compute both metrics from the default input paths into ``output/``:

.. code-block:: sh

    ethereum-transfers

Choose the input files and output directory, or compute a single metric:

.. code-block:: sh

    ethereum-transfers run --logs logs_dir --tokens tokens.csv --output-dir out
    ethereum-transfers run --metric count_of_transfers

Rank every (day, token) pair instead of keeping the top token per day:

.. code-block:: sh

    ethereum-transfers run --per-day all --limit 100

Write a smaller parquet file with the first rows of the dataset:

.. code-block:: sh

    ethereum-transfers sample --input eth_logs_decoded_2023_sample --rows 1000000
"""

import argparse
import logging
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Final

import pyarrow as pa

from ethereum_transfers import __version__
from ethereum_transfers.files import SAMPLE_ROWS, write_head_sample
from ethereum_transfers.metrics import TOP_N, Metric
from ethereum_transfers.pipeline import generate_output
from ethereum_transfers.rank import DEFAULT_PER_DAY

DEFAULT_LOGS: Final = Path("eth_logs_decoded_2023_sample")
DEFAULT_SAMPLE: Final = Path("eth_logs_decoded_2023_sample_100.gzip")
DEFAULT_TOKENS: Final = Path("tokens.csv")
DEFAULT_OUTPUT_DIR: Final = Path("output")
ALL: Final = "all"


def parse_count(text: str, minimum: int = 0) -> int:
    """
    Parse an integer option value that must be at least `minimum`.

    Parameters
    ----------
    text : str
        Command-line value.
    minimum : int, default 0
        Smallest accepted value.

    Returns
    -------
    int
        The parsed value.

    Raises
    ------
    argparse.ArgumentTypeError
        If `text` is not an integer or is below `minimum`.
    """
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected an integer, got {text!r}") from None
    if value < minimum:
        raise argparse.ArgumentTypeError(f"must be at least {minimum}, got {value}")
    return value


def count_or_all(minimum: int) -> Callable[[str], int | None]:
    """
    Build an argparse type for "an integer >= `minimum`, or ``all``".

    Parameters
    ----------
    minimum : int
        Smallest accepted integer.

    Returns
    -------
    Callable[[str], int | None]
        Converter returning the integer, or ``None`` for ``all``.
    """

    def convert(text: str) -> int | None:
        if text.strip().lower() == ALL:
            return None
        return parse_count(text, minimum)

    return convert


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
        prog="ethereum-transfers",
        description="Find the most transferred ERC-20 token per day.",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    subparsers = parser.add_subparsers(
        dest="command", metavar="COMMAND", title="commands"
    )

    run = subparsers.add_parser(
        "run",
        help="compute the metrics and write one CSV per metric (default)",
        description=(
            "Compute the metrics and write <output-dir>/<metric>.csv with the "
            "columns day, token_address and the metric."
        ),
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
        help="directory to write the CSV files into (default: %(default)s)",
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
    run.add_argument(
        "--per-day",
        type=count_or_all(1),
        default=DEFAULT_PER_DAY,
        metavar="N|all",
        help=(
            "tokens kept per day; 'all' ranks every (day, token) pair "
            "(default: %(default)s)"
        ),
    )
    run.add_argument(
        "--limit",
        type=count_or_all(0),
        default=TOP_N,
        metavar="N|all",
        help="maximum rows per CSV (default: %(default)s)",
    )

    sample = subparsers.add_parser(
        "sample",
        help="write the first rows of a parquet dataset to a smaller file",
        description="Write the first rows of a parquet dataset to a smaller file.",
    )
    sample.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_LOGS,
        metavar="PATH",
        help="parquet file or directory to read (default: %(default)s)",
    )
    sample.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_SAMPLE,
        metavar="PATH",
        help="gzip-compressed parquet file to write (default: %(default)s)",
    )
    sample.add_argument(
        "-n",
        "--rows",
        type=parse_count,
        default=SAMPLE_ROWS,
        metavar="N",
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
        Process exit status: 0 on success, 1 if the inputs or outputs are
        unusable (the reason is printed to stderr), 2 for invalid arguments.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        args = parser.parse_args(["run"])
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    inputs = (
        {"--logs": args.logs, "--tokens": args.tokens}
        if args.command == "run"
        else {"--input": args.input}
    )
    for flag, path in inputs.items():
        if not path.exists():
            print(
                f"{parser.prog}: error: {flag} {path} does not exist", file=sys.stderr
            )
            return 1

    try:
        match args.command:
            case "run":
                generate_output(
                    logs_path=args.logs,
                    tokens_path=args.tokens,
                    output_dir=args.output_dir,
                    metrics=[Metric(metric) for metric in args.metrics],
                    per_day=args.per_day,
                    limit=args.limit,
                )
            case "sample":
                write_head_sample(args.input, args.output, n_rows=args.rows)
    except (OSError, ValueError, OverflowError, pa.ArrowException) as error:
        print(f"{parser.prog}: error: {error}", file=sys.stderr)
        return 1
    return 0
