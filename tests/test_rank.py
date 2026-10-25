import random
from typing import TYPE_CHECKING

import pandas as pd
import pytest

from ethereum_transfers.rank import rank_top

if TYPE_CHECKING:
    from conftest import Oracle

A, B, C = "0x" + "a" * 40, "0x" + "b" * 40, "0x" + "c" * 40
D1, D2, D3 = "2023-04-23", "2023-04-24", "2023-04-25"


def frame(rows: list[tuple[str, str, int]], column: str = "value") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "day": [r[0] for r in rows],
            "token_address": [r[1] for r in rows],
            column: pd.Series([r[2] for r in rows], dtype=object),
        }
    )


def rows_of(result: pd.DataFrame, column: str = "value") -> list[tuple[str, str, int]]:
    return list(
        result[["day", "token_address", column]].itertuples(index=False, name=None)
    )


# --- R1-R8 ---------------------------------------------------------------------


def test_top_token_per_day_ordered_by_metric() -> None:  # R1
    agg = frame([(D1, A, 5), (D1, B, 3), (D2, A, 1), (D2, B, 9)])
    assert rows_of(rank_top(agg, "value")) == [(D2, B, 9), (D1, A, 5)]


def test_tie_within_day_goes_to_lower_address() -> None:  # R2
    agg = frame([(D1, B, 5), (D1, A, 5), (D1, C, 4)])
    assert rows_of(rank_top(agg, "value")) == [(D1, A, 5)]


def test_tie_across_days_orders_by_day() -> None:  # R3
    agg = frame([(D2, B, 5), (D1, A, 5)])
    assert rows_of(rank_top(agg, "value")) == [(D1, A, 5), (D2, B, 5)]


def test_limit_keeps_largest_daily_winners() -> None:  # R4
    agg = frame([(D1, A, 1), (D2, A, 3), (D3, A, 2)])
    assert rows_of(rank_top(agg, "value", limit=2)) == [(D2, A, 3), (D3, A, 2)]


def test_per_day_none_is_a_global_ranking() -> None:  # R5
    agg = frame([(D1, A, 9), (D1, B, 8), (D2, A, 1), (D2, B, 7)])
    result = rank_top(agg, "value", per_day=None, limit=3)
    assert rows_of(result) == [(D1, A, 9), (D1, B, 8), (D2, B, 7)]


def test_per_day_two() -> None:  # R6
    agg = frame([(D1, A, 1), (D1, B, 2), (D1, C, 3), (D2, A, 4)])
    result = rank_top(agg, "value", per_day=2, limit=None)
    assert rows_of(result) == [(D2, A, 4), (D1, C, 3), (D1, B, 2)]


def test_big_integers_rank_exactly() -> None:  # R7
    # 2**256 and 2**256 + 1 are equal as float64; ranking must not round.
    agg = frame([(D1, A, 2**256), (D1, B, 2**256 + 1), (D1, C, 10)])
    assert rows_of(rank_top(agg, "value", per_day=None, limit=None)) == [
        (D1, B, 2**256 + 1),
        (D1, A, 2**256),
        (D1, C, 10),
    ]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        pytest.param({"per_day": 0}, "per_day", id="R8-per-day-zero"),
        pytest.param({"limit": -1}, "limit", id="R8-negative-limit"),
    ],
)
def test_rejects_invalid_arguments(kwargs: dict[str, int], message: str) -> None:
    agg = frame([(D1, A, 1)])
    with pytest.raises(ValueError, match=message):
        rank_top(agg, "value", **kwargs)


# --- contract details ------------------------------------------------------------


def test_limit_zero_and_unlimited() -> None:
    agg = frame([(D1, A, 1), (D2, A, 2)])
    assert rank_top(agg, "value", limit=0).empty
    assert len(rank_top(agg, "value", limit=None)) == 2


def test_defaults_are_top_one_and_limit_100() -> None:
    agg = frame(
        [(f"2023-{1 + i // 28:02d}-{1 + i % 28:02d}", A, i) for i in range(150)]
    )
    result = rank_top(agg, "value")
    assert len(result) == 100
    assert result["value"].tolist() == list(range(149, 49, -1))


def test_output_columns_index_and_input_untouched() -> None:
    agg = frame([(D1, B, 2), (D1, A, 1)]).assign(extra="ignored")
    before = agg.copy()
    result = rank_top(agg, "value", per_day=None)
    assert list(result.columns) == ["day", "token_address", "value"]
    assert list(result.index) == [0, 1]
    pd.testing.assert_frame_equal(agg, before)


def test_works_with_int64_columns() -> None:
    agg = pd.DataFrame(
        {"day": [D1, D1, D2], "token_address": [A, B, A], "transfer_count": [3, 4, 1]}
    )
    result = rank_top(agg, "transfer_count")
    assert result["transfer_count"].tolist() == [4, 1]
    assert result["transfer_count"].dtype == agg["transfer_count"].dtype


def test_rejects_missing_column() -> None:
    agg = frame([(D1, A, 1)])
    with pytest.raises(ValueError, match="missing column"):
        rank_top(agg, "transfer_amount")


def test_rejects_missing_values() -> None:
    agg = frame([(D1, A, 1), (D2, A, 2)])
    agg.loc[0, "value"] = None
    with pytest.raises(ValueError, match="missing values"):
        rank_top(agg, "value")


def test_empty_aggregate() -> None:
    agg = frame([])
    result = rank_top(agg, "value")
    assert result.empty
    assert list(result.columns) == ["day", "token_address", "value"]


# --- property test against the brute-force oracle --------------------------------


@pytest.mark.parametrize(
    ("per_day", "limit"),
    [(1, 100), (1, 3), (2, 5), (None, 10), (None, None), (3, None)],
)
def test_matches_oracle(
    oracle: "Oracle", rng: random.Random, per_day: int | None, limit: int | None
) -> None:
    days = [f"2023-04-{d:02d}" for d in range(20, 30)]
    tokens = [f"0x{t:040x}" for t in range(12)]
    # Small value range so ties are frequent; a few huge values for exactness.
    values = {
        (day, token): rng.choice([*range(6), 2**200, 2**200 + 1])
        for day in days
        for token in rng.sample(tokens, rng.randint(1, len(tokens)))
    }
    agg = frame([(day, token, value) for (day, token), value in values.items()])
    expected = oracle.rank(values, per_day=per_day, limit=limit)
    assert rows_of(rank_top(agg, "value", per_day=per_day, limit=limit)) == expected
