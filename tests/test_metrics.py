from datetime import date, timedelta
from typing import cast

import pandas as pd
import pytest

from ethereum_transfers.metrics import TOP_N, Metric, aggregate_column, compute_metric

A, B, C = ("0x" + c * 40 for c in "abc")
D1, D2, D3 = "2023-04-01", "2023-04-02", "2023-04-03"


def aggregate(rows: list[tuple[str, str, int, int]]) -> pd.DataFrame:
    """Build an aggregate frame from ``(day, token, count, amount)`` rows."""
    return pd.DataFrame(
        {
            "day": [row[0] for row in rows],
            "token_address": [row[1] for row in rows],
            "transfer_count": pd.Series([row[2] for row in rows], dtype="int64"),
            "transfer_amount": pd.Series([row[3] for row in rows], dtype=object),
        }
    )


def rows(result: pd.DataFrame) -> list[tuple[object, ...]]:
    """Return the rows of a metric result as plain tuples, in order."""
    return list(result.itertuples(index=False, name=None))


def test_metric_values_name_the_output_columns_and_files() -> None:
    assert [str(metric) for metric in Metric] == [
        "count_of_transfers",
        "quantity_transferred",
    ]
    assert Metric("quantity_transferred") is Metric.QUANTITY_TRANSFERRED


def test_top_n_is_100() -> None:
    assert TOP_N == 100


@pytest.mark.parametrize(
    ("metric", "column"),
    [
        (Metric.COUNT_OF_TRANSFERS, "transfer_count"),
        (Metric.QUANTITY_TRANSFERRED, "transfer_amount"),
    ],
)
def test_aggregate_column(metric: Metric, column: str) -> None:
    assert aggregate_column(metric) == column


def test_aggregate_column_rejects_an_unknown_metric() -> None:
    bogus = cast("Metric", "bogus")
    with pytest.raises(AssertionError, match="unreachable"):
        aggregate_column(bogus)


class TestCountOfTransfers:
    def test_keeps_the_top_token_of_each_day(self) -> None:
        agg = aggregate([(D1, A, 3, 1), (D1, B, 1, 100), (D2, B, 2, 1), (D2, C, 1, 9)])

        result = compute_metric(agg, Metric.COUNT_OF_TRANSFERS)

        assert list(result.columns) == ["day", "token_address", "count_of_transfers"]
        assert rows(result) == [(D1, A, 3), (D2, B, 2)]

    def test_orders_days_by_count_descending(self) -> None:
        agg = aggregate([(D1, A, 1, 0), (D2, A, 5, 0), (D3, A, 3, 0)])

        result = compute_metric(agg, Metric.COUNT_OF_TRANSFERS)

        assert result["day"].tolist() == [D2, D3, D1]

    def test_keeps_the_top_n_days(self) -> None:
        start = date(2023, 1, 1)
        agg = aggregate(
            [((start + timedelta(days=i)).isoformat(), A, i + 1, 0) for i in range(150)]
        )

        result = compute_metric(agg, Metric.COUNT_OF_TRANSFERS)

        assert len(result) == TOP_N
        assert result["count_of_transfers"].tolist() == list(range(150, 50, -1))


class TestQuantityTransferred:
    def test_keeps_the_top_token_of_each_day_by_exact_amount(self) -> None:
        agg = aggregate(
            [(D1, A, 9, 2**256), (D1, B, 1, 2**256 + 1), (D2, A, 1, 5), (D2, C, 7, 4)]
        )

        result = compute_metric(agg, Metric.QUANTITY_TRANSFERRED)

        assert list(result.columns) == ["day", "token_address", "quantity_transferred"]
        assert rows(result) == [(D1, B, 2**256 + 1), (D2, A, 5)]

    def test_returns_one_row_per_day(self) -> None:
        agg = aggregate([(D1, A, 1, 1), (D1, B, 1, 2), (D2, A, 1, 3), (D3, C, 1, 0)])

        result = compute_metric(agg, Metric.QUANTITY_TRANSFERRED)

        assert result["day"].is_unique
        assert len(result) == 3


@pytest.mark.parametrize("metric", list(Metric))
def test_per_day_and_limit_are_passed_through(metric: Metric) -> None:
    agg = aggregate([(D1, A, 3, 3), (D1, B, 2, 2), (D2, C, 1, 1)])

    result = compute_metric(agg, metric, per_day=None, limit=2)

    assert rows(result) == [(D1, A, 3), (D1, B, 2)]


@pytest.mark.parametrize("metric", list(Metric))
def test_compute_metric_does_not_modify_its_input(metric: Metric) -> None:
    agg = aggregate([(D1, A, 3, 3), (D1, B, 2, 2)])
    before = agg.copy()

    compute_metric(agg, metric)

    pd.testing.assert_frame_equal(agg, before)
