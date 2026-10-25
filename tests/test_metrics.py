from collections.abc import Callable
from datetime import date, timedelta
from typing import cast

import pandas as pd
import pytest

from ethereum_transfers.metrics import (
    TOP_N,
    Metric,
    compute_metric,
    count_of_transfers,
    join_tokens,
    quantity_transferred,
)

type MakeLogs = Callable[..., pd.DataFrame]

# the addresses listed by the `tokens` fixture (see conftest.py)
A, B, C, UNNAMED, NO_SYMBOL, UNLISTED = ("0x" + c * 40 for c in "abcdef")
DAY_1, DAY_2, DAY_3 = date(2023, 4, 1), date(2023, 4, 2), date(2023, 4, 3)


def rows(result: pd.DataFrame) -> list[tuple[object, ...]]:
    """Return the rows of a metric result as plain tuples, in order."""
    return list(result.itertuples(index=False, name=None))


def test_metric_values_name_the_output_columns_and_files() -> None:
    assert [str(metric) for metric in Metric] == [
        "count_of_transfers",
        "quantity_transferred",
    ]
    assert Metric("quantity_transferred") is Metric.QUANTITY_TRANSFERRED


class TestJoinTokens:
    def test_keeps_only_logs_of_listed_tokens(
        self, make_logs: MakeLogs, tokens: pd.DataFrame
    ) -> None:
        logs = make_logs((A, DAY_1, 1), (UNLISTED, DAY_1, 1), (B, DAY_2, 1))

        joined = join_tokens(logs, tokens)

        assert joined["ADDRESS"].tolist() == [A, B]
        assert joined["name"].tolist() == ["Token A", "Token B"]
        assert joined["symbol"].tolist() == ["AAA", "BBB"]
        assert "address" not in joined.columns
        pd.testing.assert_index_equal(joined.index, pd.RangeIndex(2))

    @pytest.mark.parametrize(
        "address", [UNNAMED, NO_SYMBOL], ids=["no-name", "no-symbol"]
    )
    def test_drops_tokens_without_name_or_symbol(
        self, make_logs: MakeLogs, tokens: pd.DataFrame, address: str
    ) -> None:
        logs = make_logs((address, DAY_1, 1), (A, DAY_1, 1))

        assert join_tokens(logs, tokens)["ADDRESS"].tolist() == [A]

    @pytest.mark.parametrize("column", ["ADDRESS", "PARAMS", "BLOCK_DATE"])
    def test_drops_rows_with_a_null_key(
        self, make_logs: MakeLogs, tokens: pd.DataFrame, column: str
    ) -> None:
        logs = make_logs((A, DAY_1, 1), (B, DAY_1, 1), (C, DAY_1, 1))
        logs.loc[1, column] = None

        joined = join_tokens(logs, tokens)

        assert joined["ADDRESS"].tolist() == [A, C]
        pd.testing.assert_index_equal(joined.index, pd.RangeIndex(2))


class TestCountOfTransfers:
    def test_keeps_the_top_token_of_each_day(self, make_logs: MakeLogs) -> None:
        logs = make_logs(
            (A, DAY_1, 1),
            (B, DAY_1, 1),
            (A, DAY_1, 1),
            (A, DAY_1, 1),
            (C, DAY_2, 1),
            (B, DAY_2, 1),
            (B, DAY_2, 1),
        )

        result = count_of_transfers(logs)

        assert list(result.columns) == ["ADDRESS", "BLOCK_DATE", "count_of_transfers"]
        assert rows(result) == [(A, DAY_1, 3), (B, DAY_2, 2)]

    def test_orders_days_by_count_descending(self, make_logs: MakeLogs) -> None:
        logs = make_logs(
            (A, DAY_1, 1),
            *[(B, DAY_2, 1)] * 3,
            *[(C, DAY_3, 1)] * 2,
        )

        result = count_of_transfers(logs)

        assert rows(result) == [(B, DAY_2, 3), (C, DAY_3, 2), (A, DAY_1, 1)]

    def test_keeps_the_top_n_days(self, make_logs: MakeLogs) -> None:
        # day i has i transfers, so the TOP_N busiest days are the last ones
        n_days = TOP_N + 5
        days = [DAY_1 + timedelta(days=i) for i in range(n_days)]
        logs = make_logs(
            *[(A, day, 1) for i, day in enumerate(days, start=1) for _ in range(i)]
        )

        result = count_of_transfers(logs)

        assert len(result) == TOP_N
        assert result["count_of_transfers"].tolist() == list(range(n_days, 5, -1))
        assert result["BLOCK_DATE"].tolist() == days[:4:-1]


class TestQuantityTransferred:
    def test_returns_one_row_per_day(self, make_logs: MakeLogs) -> None:
        logs = make_logs(
            (A, DAY_1, 1),
            (B, DAY_1, 1),
            (B, DAY_2, 1),
            (C, DAY_2, 1),
            (C, DAY_3, 1),
        )

        result = quantity_transferred(logs)

        assert list(result.columns) == [
            "ADDRESS",
            "BLOCK_DATE",
            "quantity_transferred",
        ]
        assert sorted(result["BLOCK_DATE"]) == [DAY_1, DAY_2, DAY_3]

    def test_keeps_the_top_token_of_each_day(self, make_logs: MakeLogs) -> None:
        # equal amounts: the top token is the one with the most transfers
        logs = make_logs(
            (A, DAY_1, 1),
            (A, DAY_1, 1),
            (B, DAY_1, 1),
            (A, DAY_2, 1),
            (B, DAY_2, 1),
            (B, DAY_2, 1),
            (B, DAY_2, 1),
        )

        result = quantity_transferred(logs)

        winners = dict(zip(result["BLOCK_DATE"], result["ADDRESS"], strict=True))
        assert winners == {DAY_1: A, DAY_2: B}

    def test_keeps_at_most_top_n_days(self, make_logs: MakeLogs) -> None:
        days = [DAY_1 + timedelta(days=i) for i in range(TOP_N + 5)]
        logs = make_logs(*[(A, day, 1) for day in days])

        result = quantity_transferred(logs)

        assert len(result) == TOP_N
        assert result["BLOCK_DATE"].is_unique


@pytest.mark.parametrize(
    ("metric", "function"),
    [
        (Metric.COUNT_OF_TRANSFERS, count_of_transfers),
        (Metric.QUANTITY_TRANSFERRED, quantity_transferred),
    ],
)
def test_compute_metric_dispatches_to_the_metric_function(
    logs: pd.DataFrame,
    tokens: pd.DataFrame,
    metric: Metric,
    function: Callable[[pd.DataFrame], pd.DataFrame],
) -> None:
    joined = join_tokens(logs, tokens)

    result = compute_metric(joined, metric)

    pd.testing.assert_frame_equal(result, function(joined))
    assert list(result.columns) == ["ADDRESS", "BLOCK_DATE", metric]


def test_compute_metric_rejects_an_unknown_metric(
    logs: pd.DataFrame, tokens: pd.DataFrame
) -> None:
    with pytest.raises(AssertionError, match="unreachable"):
        compute_metric(join_tokens(logs, tokens), cast("Metric", "bogus"))


@pytest.mark.parametrize(
    "function",
    [count_of_transfers, quantity_transferred],
    ids=lambda function: function.__name__,
)
def test_metric_functions_do_not_modify_their_input(
    logs: pd.DataFrame,
    tokens: pd.DataFrame,
    function: Callable[[pd.DataFrame], pd.DataFrame],
) -> None:
    joined = join_tokens(logs, tokens)
    before = joined.copy()

    function(joined)

    pd.testing.assert_frame_equal(joined, before)


def test_join_tokens_does_not_modify_its_inputs(
    logs: pd.DataFrame, tokens: pd.DataFrame
) -> None:
    logs_before, tokens_before = logs.copy(), tokens.copy()

    join_tokens(logs, tokens)

    pd.testing.assert_frame_equal(logs, logs_before)
    pd.testing.assert_frame_equal(tokens, tokens_before)
