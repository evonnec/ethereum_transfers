"""Run the CLI on the real input data.

Skipped unless ``ETHEREUM_TRANSFERS_LOGS`` (parquet file or directory) and
``ETHEREUM_TRANSFERS_TOKENS`` (``tokens.csv``) are set. The pipeline streams the
parquet input, so the full dataset is fine (about 1.2 GB of memory)::

    ETHEREUM_TRANSFERS_LOGS=eth_logs_decoded_2023_sample \\
    ETHEREUM_TRANSFERS_TOKENS=tokens.csv uv run pytest -m integration --no-cov

`test_run_on_the_real_data` holds for any input, such as a sample made with
``ethereum-transfers sample``. The golden and funnel tests pin the results for
the full 2023 sample dataset (18,200,863 logs) and skip on any other input.
"""

import os
from pathlib import Path
from typing import Final

import pandas as pd
import pyarrow.dataset as ds
import pytest

from ethereum_transfers.cli import main
from ethereum_transfers.metrics import TOP_N, Metric
from ethereum_transfers.pipeline import aggregate_inputs

pytestmark = pytest.mark.integration

GOLDEN_DIR: Final = Path(__file__).parent / "golden"
FULL_DATASET_ROWS: Final = 18_200_863


@pytest.fixture(scope="module")
def real_data() -> tuple[Path, Path]:
    logs = os.environ.get("ETHEREUM_TRANSFERS_LOGS")
    tokens = os.environ.get("ETHEREUM_TRANSFERS_TOKENS")
    if not (logs and tokens):
        pytest.skip("set ETHEREUM_TRANSFERS_LOGS and ETHEREUM_TRANSFERS_TOKENS")
    return Path(logs), Path(tokens)


@pytest.fixture(scope="module")
def full_dataset(real_data: tuple[Path, Path]) -> tuple[Path, Path]:
    logs, _ = real_data
    rows = ds.dataset(logs, format="parquet").count_rows()  # parquet metadata only
    if rows != FULL_DATASET_ROWS:
        pytest.skip(f"golden results are for the full dataset, got {rows} rows")
    return real_data


def run_cli(real_data: tuple[Path, Path], output_dir: Path) -> None:
    logs, tokens = real_data
    argv = ["run", "--logs", str(logs), "--tokens", str(tokens), "-o", str(output_dir)]
    assert main(argv) == 0


def test_run_on_the_real_data(real_data: tuple[Path, Path], tmp_path: Path) -> None:
    _, tokens = real_data

    run_cli(real_data, tmp_path)

    listed = set(pd.read_csv(tokens, usecols=["address"])["address"].str.lower())
    for metric in Metric:
        result = pd.read_csv(tmp_path / f"{metric}.csv", dtype=str)
        assert list(result.columns) == ["day", "token_address", metric]
        assert 0 < len(result) <= TOP_N
        assert result["day"].is_unique
        assert set(result["token_address"]) <= listed
        values = [int(value) for value in result[metric]]
        assert values == sorted(values, reverse=True)


@pytest.fixture(scope="module")
def full_outputs(
    full_dataset: tuple[Path, Path], tmp_path_factory: pytest.TempPathFactory
) -> Path:
    output_dir = tmp_path_factory.mktemp("full")
    run_cli(full_dataset, output_dir)
    return output_dir


@pytest.mark.parametrize("metric", list(Metric))
def test_full_dataset_matches_the_golden_csvs(
    full_outputs: Path, metric: Metric
) -> None:
    golden = (GOLDEN_DIR / f"{metric}.csv").read_text()
    assert (full_outputs / f"{metric}.csv").read_text() == golden


def test_full_dataset_funnel(full_dataset: tuple[Path, Path]) -> None:
    aggregate = aggregate_inputs(*full_dataset)

    assert int(aggregate["transfer_count"].sum()) == 3_359_694
    assert aggregate["token_address"].nunique() == 1_306
    assert len(aggregate) == 4_375
    assert aggregate["day"].unique().tolist() == [
        f"2023-04-{day}" for day in range(23, 31)
    ]
    assert max(aggregate["transfer_amount"]).bit_length() == 257
