"""Run the CLI on the real input data.

Skipped unless ``ETHEREUM_TRANSFERS_LOGS`` (parquet file or directory) and
``ETHEREUM_TRANSFERS_TOKENS`` (``tokens.csv``) are set. ``run`` loads the whole
parquet input into memory, so point it at a sample (``ethereum-transfers
sample``), not the full dataset::

    ETHEREUM_TRANSFERS_LOGS=eth_logs_decoded_2023_sample_100.gzip \\
    ETHEREUM_TRANSFERS_TOKENS=tokens.csv uv run pytest -m integration
"""

import os
from pathlib import Path

import pandas as pd
import pytest

from ethereum_transfers.cli import main
from ethereum_transfers.metrics import TOP_N, Metric

pytestmark = pytest.mark.integration


@pytest.fixture
def real_data() -> tuple[Path, Path]:
    logs = os.environ.get("ETHEREUM_TRANSFERS_LOGS")
    tokens = os.environ.get("ETHEREUM_TRANSFERS_TOKENS")
    if not (logs and tokens):
        pytest.skip("set ETHEREUM_TRANSFERS_LOGS and ETHEREUM_TRANSFERS_TOKENS")
    return Path(logs), Path(tokens)


def test_run_on_the_real_data(real_data: tuple[Path, Path], tmp_path: Path) -> None:
    logs, tokens = real_data

    argv = ["run", "--logs", str(logs), "--tokens", str(tokens), "-o", str(tmp_path)]
    assert main(argv) == 0

    listed = set(pd.read_csv(tokens, usecols=["address"])["address"])
    for metric in Metric:
        result = pd.read_csv(tmp_path / f"{metric}.csv", dtype=str)
        assert list(result.columns) == ["ADDRESS", "BLOCK_DATE", metric]
        assert 0 < len(result) <= TOP_N
        assert result["BLOCK_DATE"].is_unique
        assert set(result["ADDRESS"]) <= listed
    counts = pd.read_csv(tmp_path / "count_of_transfers.csv")["count_of_transfers"]
    assert counts.is_monotonic_decreasing
