import random
from datetime import date, timedelta
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from ethereum_transfers.aggregate import (
    AGGREGATE_COLUMNS,
    aggregate_transfers,
    combine_partials,
    partial_aggregate,
)
from ethereum_transfers.transfers import erc20_transfer_filter

if TYPE_CHECKING:
    from conftest import AggregateKey, Log, LogFactory, Oracle


def as_dict(frame: pd.DataFrame) -> dict[tuple[str, str], tuple[int, int]]:
    return {
        (day, token): (count, amount)
        for day, token, count, amount in frame.itertuples(index=False, name=None)
    }


def run(factory: "LogFactory", rows: list["Log"], batches: int = 1) -> pd.DataFrame:
    table = factory.table(rows)
    size = -(-table.num_rows // batches) if table.num_rows else 1
    return aggregate_transfers(table.to_batches(max_chunksize=size))


# --- A1-A9 ---------------------------------------------------------------------


def test_sums_and_counts_one_group(factory: "LogFactory") -> None:  # A1
    a, d1 = factory.TOKEN_A, factory.DAY_1
    result = run(factory, [factory.erc20(a, d1, 1), factory.erc20(a, d1, 2)])
    assert as_dict(result) == {(d1.isoformat(), a): (2, 3)}


def test_groups_by_day_and_token(factory: "LogFactory") -> None:  # A2
    a, b, d1, d2 = factory.TOKEN_A, factory.TOKEN_B, factory.DAY_1, factory.DAY_2
    result = run(
        factory,
        [factory.erc20(b, d2, 5), factory.erc20(a, d2, 1), factory.erc20(a, d1, 7)],
    )
    assert result.to_dict("records") == [
        {
            "day": d1.isoformat(),
            "token_address": a,
            "transfer_count": 1,
            "transfer_amount": 7,
        },
        {
            "day": d2.isoformat(),
            "token_address": a,
            "transfer_count": 1,
            "transfer_amount": 1,
        },
        {
            "day": d2.isoformat(),
            "token_address": b,
            "transfer_count": 1,
            "transfer_amount": 5,
        },
    ]


def test_sum_beyond_uint256_is_exact(factory: "LogFactory") -> None:  # A3
    a, d1 = factory.TOKEN_A, factory.DAY_1
    big = 2**256 - 1
    result = run(factory, [factory.erc20(a, d1, big), factory.erc20(a, d1, big)])
    amount = result.loc[0, "transfer_amount"]
    assert amount == 2**257 - 2
    assert isinstance(amount, int)
    assert amount.bit_length() == 257


def test_sum_beyond_int64_is_exact(factory: "LogFactory") -> None:  # A4
    a, d1 = factory.TOKEN_A, factory.DAY_1
    result = run(factory, [factory.erc20(a, d1, 2**63)] * 3)
    assert as_dict(result) == {(d1.isoformat(), a): (3, 3 * 2**63)}


def test_groups_split_across_batches_match_one_batch(
    factory: "LogFactory", rng: random.Random
) -> None:  # A5
    tokens, days = [factory.TOKEN_A, factory.TOKEN_B], [factory.DAY_1, factory.DAY_2]
    rows = [
        factory.erc20(rng.choice(tokens), rng.choice(days), rng.getrandbits(256))
        for _ in range(50)
    ]
    one = run(factory, rows, batches=1)
    many = run(factory, rows, batches=7)
    pd.testing.assert_frame_equal(one, many)


def test_zero_value_transfer_counts(factory: "LogFactory") -> None:  # A6
    a, d1 = factory.TOKEN_A, factory.DAY_1
    result = run(factory, [factory.erc20(a, d1, 0), factory.erc20(a, d1, 4)])
    assert as_dict(result) == {(d1.isoformat(), a): (2, 4)}


def test_self_transfer_counts_end_to_end(factory: "LogFactory") -> None:  # A7
    a, d1, me = factory.TOKEN_A, factory.DAY_1, factory.SENDER
    table = factory.table([factory.erc20(a, d1, 9, sender=me, recipient=me)])
    selected = table.filter(erc20_transfer_filter(pa.array([a])))
    assert as_dict(aggregate_transfers([selected])) == {(d1.isoformat(), a): (1, 9)}


def test_empty_input_has_documented_columns_and_dtypes(factory: "LogFactory") -> None:
    # A8: no matching rows (no batches, or only empty ones)
    for batches in ([], [factory.table([])], factory.table([]).to_batches()):
        result = aggregate_transfers(batches)
        assert list(result.columns) == list(AGGREGATE_COLUMNS)
        assert result.empty
        assert result["transfer_count"].dtype == np.int64
        assert result["transfer_amount"].dtype == object
        assert pd.api.types.is_string_dtype(result["day"])
        assert pd.api.types.is_string_dtype(result["token_address"])


def test_group_size_guard(factory: "LogFactory") -> None:  # A9
    a, d1 = factory.TOKEN_A, factory.DAY_1
    partials = [partial_aggregate(factory.table([factory.erc20(a, d1, 1)] * 3))]
    with pytest.raises(OverflowError, match="3 transfers"):
        combine_partials(partials, max_group_rows=2)
    assert combine_partials(partials, max_group_rows=3).loc[0, "transfer_count"] == 3


# --- partial_aggregate ----------------------------------------------------------


def test_partial_aggregate_schema(factory: "LogFactory") -> None:
    a, d1 = factory.TOKEN_A, factory.DAY_1
    partial = partial_aggregate(factory.table([factory.erc20(a, d1, 2**32 + 3)] * 2))
    assert sorted(partial.column_names) == sorted(
        ["BLOCK_DATE", "ADDRESS", "n", *(f"limb{i}" for i in range(8))]
    )
    row = partial.to_pylist()[0]
    assert row["n"] == 2
    assert [row[f"limb{i}"] for i in range(8)] == [0, 0, 0, 0, 0, 0, 2, 6]
    assert all(partial.schema.field(f"limb{i}").type == pa.uint64() for i in range(8))


def test_partial_aggregate_accepts_record_batches(factory: "LogFactory") -> None:
    a, d1 = factory.TOKEN_A, factory.DAY_1
    (batch,) = factory.table([factory.erc20(a, d1, 5)]).to_batches()
    assert partial_aggregate(batch).num_rows == 1


def test_partial_aggregate_requires_columns(factory: "LogFactory") -> None:
    table = factory.table(
        [factory.erc20(factory.TOKEN_A, factory.DAY_1, 1)]
    ).drop_columns(["DATA", "ADDRESS"])
    with pytest.raises(ValueError, match="ADDRESS, DATA"):
        partial_aggregate(table)


def test_partial_aggregate_rejects_bad_data(factory: "LogFactory") -> None:
    table = factory.table([factory.erc721(factory.TOKEN_A, factory.DAY_1, 1)])
    with pytest.raises(ValueError, match="uint256 word"):
        partial_aggregate(table)


def test_input_is_not_mutated(factory: "LogFactory") -> None:
    table = factory.table([factory.erc20(factory.TOKEN_A, factory.DAY_1, 1)])
    before = table.to_pylist()
    aggregate_transfers([table])
    assert table.to_pylist() == before


# --- property test against the plain-Python oracle -------------------------------


def test_matches_oracle_on_random_transfers(
    factory: "LogFactory", oracle: "Oracle", rng: random.Random
) -> None:
    tokens = [factory.TOKEN_A, factory.TOKEN_B, *(f"0x{i:040x}" for i in range(1, 6))]
    days = [date(2023, 4, 23) + timedelta(days=i) for i in range(4)]
    rows = [
        factory.erc20(
            rng.choice(tokens),
            rng.choice(days),
            rng.choice([0, 1, 2**63, 2**256 - 1, rng.getrandbits(256)]),
        )
        for _ in range(10_000)
    ]
    expected: dict[AggregateKey, tuple[int, int]] = oracle.aggregate(rows)
    result = run(factory, rows, batches=3)
    assert as_dict(result) == expected
    assert list(result.columns) == list(AGGREGATE_COLUMNS)
    keys = list(zip(result["day"], result["token_address"], strict=True))
    assert keys == sorted(keys)
