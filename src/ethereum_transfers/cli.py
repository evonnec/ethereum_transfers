"""Command-line interface: ``ethereum-transfers`` / ``python -m ethereum_transfers``.

Examples
--------
Compute both metrics from the default input files into the current directory:

.. code-block:: sh

    ethereum-transfers

Choose the input files and output directory, or compute a single metric:

.. code-block:: sh

    ethereum-transfers run --logs logs.parquet --tokens tokens.csv --output-dir out
    ethereum-transfers run --metric count_of_transfers

Write a smaller parquet file for testing:

.. code-block:: sh

    ethereum-transfers sample --input eth_logs_decoded_2023_sample --rows 1000000
"""

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from ethereum_transfers import __version__
from ethereum_transfers.files import SAMPLE_ROWS, generate_output, write_new_output
from ethereum_transfers.metrics import Metric

DEFAULT_SOURCE_LOGS: Final = Path("eth_logs_decoded_2023_sample")
DEFAULT_LOGS: Final = Path("eth_logs_decoded_2023_sample_100.gzip")
DEFAULT_TOKENS: Final = Path("tokens.csv")
DEFAULT_OUTPUT_DIR: Final = Path()


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
