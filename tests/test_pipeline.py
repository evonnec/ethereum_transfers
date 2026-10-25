"""End-to-end metric results: parquet + token CSV in, metric CSVs out.

These go through `generate_output` and read the CSVs back, so they hold however
the computation is split between functions. The ``xfail`` tests assert the
correct results for known bugs; strict xfail makes them fail loudly once a bug
is fixed, so the fix must remove the marker.
"""

from collections.abc import Callable
from datetime import date
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest

from ethereum_transfers.files import generate_output
from ethereum_transfers.metrics import Metric

type MakeLogs = Callable[..., pd.DataFrame]
type WriteInputs = Callable[[pd.DataFrame, pd.DataFrame], tuple[Path, Path]]

# the addresses listed by the `tokens` fixture (see conftest.py)
A, B, C = ("0x" + c * 40 for c in "abc")
DAY_1, DAY_2, DAY_3 = date(2023, 4, 1), date(2023, 4, 2), date(2023, 4, 3)


@pytest.fixture
def run_metrics(
    tmp_path: Path, tokens: pd.DataFrame, write_inputs: WriteInputs
) -> Callable[[pd.DataFrame], dict[Metric, pd.DataFrame]]:
    """Return ``run(logs)``: write the inputs, run every metric, read the CSVs."""

    def run(logs: pd.DataFrame) -> dict[Metric, pd.DataFrame]:
        logs_path, tokens_path = write_inputs(logs, tokens)
        written = generate_output(
            logs_path=logs_path,
            tokens_path=tokens_path,
            output_dir=tmp_path / "out",
            metrics=list(Metric),
        )
        return {
            metric: pd.read_csv(path, dtype=str)
            for metric, path in zip(Metric, written, strict=True)
        }

    return run


def winners(result: pd.DataFrame, metric: Metric) -> dict[str, tuple[str, Decimal]]:
    """Map each ISO day in a metric CSV to its (token address, metric value)."""
    columns = result[["BLOCK_DATE", "ADDRESS", metric]]
    return {
        day: (address, Decimal(value))
        for day, address, value in columns.itertuples(index=False, name=None)
    }


def test_count_of_transfers_end_to_end(
    make_logs: MakeLogs,
    run_metrics: Callable[[pd.DataFrame], dict[Metric, pd.DataFrame]],
) -> None:
    logs = make_logs(
        (A, DAY_1, 1),
        (A, DAY_1, 1),
        (B, DAY_1, 1),
        (B, DAY_2, 1),
        (C, DAY_3, 1),
        (C, DAY_3, 1),
        (C, DAY_3, 1),
    )

    result = run_metrics(logs)[Metric.COUNT_OF_TRANSFERS]

    assert result.to_dict("list") == {
        "ADDRESS": [C, A, B],
        "BLOCK_DATE": ["2023-04-03", "2023-04-01", "2023-04-02"],
        "count_of_transfers": ["3", "2", "1"],
    }


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="amounts are summed as strings (concatenated); fixed in the review PR",
)
@pytest.mark.parametrize(
    ("amounts", "total"), [((2, 10), 12), ((5, 5), 10)], ids=["2+10", "5+5"]
)
def test_quantity_sums_amounts_numerically(
    make_logs: MakeLogs,
    run_metrics: Callable[[pd.DataFrame], dict[Metric, pd.DataFrame]],
    amounts: tuple[int, int],
    total: int,
) -> None:
    logs = make_logs(*[(A, DAY_1, amount) for amount in amounts])

    result = run_metrics(logs)[Metric.QUANTITY_TRANSFERRED]

    assert winners(result, Metric.QUANTITY_TRANSFERRED) == {
        "2023-04-01": (A, Decimal(total))
    }


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="every row takes the PARAMS of row 1; fixed in the review PR",
)
def test_quantity_uses_each_rows_own_amount(
    make_logs: MakeLogs,
    run_metrics: Callable[[pd.DataFrame], dict[Metric, pd.DataFrame]],
) -> None:
    logs = make_logs((A, DAY_1, 100), (B, DAY_2, 7), (C, DAY_3, 3))

    result = run_metrics(logs)[Metric.QUANTITY_TRANSFERRED]

    assert winners(result, Metric.QUANTITY_TRANSFERRED) == {
        "2023-04-01": (A, Decimal(100)),
        "2023-04-02": (B, Decimal(7)),
        "2023-04-03": (C, Decimal(3)),
    }


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="all decoded events are counted, not just Transfer; fixed in the review PR",
)
@pytest.mark.parametrize(
    ("metric", "value"),
    [(Metric.COUNT_OF_TRANSFERS, 2), (Metric.QUANTITY_TRANSFERRED, 10)],
)
def test_non_transfer_events_are_excluded(
    make_logs: MakeLogs,
    run_metrics: Callable[[pd.DataFrame], dict[Metric, pd.DataFrame]],
    metric: Metric,
    value: int,
) -> None:
    # A has one Transfer and two (much larger) Approvals; B has two Transfers
    logs = make_logs(
        (A, DAY_1, 1),
        (A, DAY_1, 1000, "Approval"),
        (A, DAY_1, 1000, "Approval"),
        (B, DAY_1, 5),
        (B, DAY_1, 5),
    )

    result = run_metrics(logs)[metric]

    assert winners(result, metric) == {"2023-04-01": (B, Decimal(value))}


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="quantity is not the day's numeric total; fixed in the review PR",
)
def test_quantity_reports_one_row_per_day_with_the_days_total(
    make_logs: MakeLogs,
    run_metrics: Callable[[pd.DataFrame], dict[Metric, pd.DataFrame]],
) -> None:
    # the winner has the largest total, not the largest single transfer
    logs = make_logs(
        (A, DAY_1, 3),
        (A, DAY_1, 4),
        (B, DAY_1, 5),
        (B, DAY_2, 1),
        (B, DAY_2, 1),
        (B, DAY_2, 1),
        (A, DAY_2, 2),
    )

    result = run_metrics(logs)[Metric.QUANTITY_TRANSFERRED]

    assert len(result) == 2
    assert winners(result, Metric.QUANTITY_TRANSFERRED) == {
        "2023-04-01": (A, Decimal(7)),
        "2023-04-02": (B, Decimal(3)),
    }
