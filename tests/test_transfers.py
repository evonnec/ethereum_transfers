import random
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pytest

from ethereum_transfers.transfers import (
    TRANSFER_TOPIC0,
    decode_uint256_words,
    erc20_transfer_filter,
    limbs_to_ints,
)

if TYPE_CHECKING:
    from conftest import LogFactory


def decode(values: list[str | None]) -> list[int]:
    limbs = decode_uint256_words(pa.array(values, type=pa.string()))
    return limbs_to_ints(limbs.astype(np.uint64))


# --- decode_uint256_words / limbs_to_ints (review section 7.4, T1-T12) -------


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(0, id="T1-zero"),
        pytest.param(1, id="T2-one"),
        pytest.param(2**256 - 1, id="T3-max-uint256"),
        pytest.param(2**63, id="T4-int64-boundary"),
        pytest.param(2**32, id="limb-boundary"),
        pytest.param(2**224, id="top-limb"),
    ],
)
def test_decode_round_trips_exact_values(factory: "LogFactory", value: int) -> None:
    assert decode([factory.word(value)]) == [value]


def test_decode_returns_big_endian_uint32_limbs(factory: "LogFactory") -> None:
    limbs = decode_uint256_words(pa.array([factory.word(2**224 + 5)]))
    assert limbs.dtype == np.uint32
    assert limbs.shape == (1, 8)
    assert limbs.tolist() == [[1, 0, 0, 0, 0, 0, 0, 5]]


def test_decode_accepts_upper_case_hex() -> None:  # T5
    upper = "0x" + "ABCDEF" * 10 + "ABCD"
    assert decode([upper]) == decode([upper.lower()]) == [int(upper, 16)]


@pytest.mark.parametrize(
    "bad",
    [
        pytest.param("0x" + "0" * 62, id="T6-short"),
        pytest.param("0x" + "0" * 128, id="T6-two-words"),
        pytest.param("0x", id="T6-empty-data"),
        pytest.param("", id="T6-empty-string"),
        pytest.param("0x" + "g" * 64, id="T7-non-hex"),
        pytest.param("0x" + "0" * 62 + " 1", id="T7-whitespace"),
        pytest.param("0x" + "0" * 62 + "  ", id="T7-trailing-spaces"),
        pytest.param("0x" + "0" * 63 + "é", id="T7-non-ascii"),
        pytest.param("00" + "0" * 64, id="T8-no-prefix"),
        pytest.param(None, id="T9-null"),
    ],
)
def test_decode_rejects_malformed_words(factory: "LogFactory", bad: str | None) -> None:
    array = pa.array([factory.word(1), bad], type=pa.string())
    with pytest.raises(ValueError, match="uint256 word"):
        decode_uint256_words(array)


def test_decode_rejects_non_string_arrays() -> None:
    array = pa.array([1, 2])
    with pytest.raises(ValueError, match="string array"):
        decode_uint256_words(array)


def test_decode_empty_array() -> None:  # T10
    limbs = decode_uint256_words(pa.array([], type=pa.string()))
    assert limbs.shape == (0, 8)
    assert limbs_to_ints(limbs.astype(np.uint64)) == []


def test_decode_respects_slices_and_chunks(factory: "LogFactory") -> None:  # T11
    values = list(range(10))
    array = pa.array([factory.word(v) for v in values])
    sliced = array.slice(3, 4)
    assert limbs_to_ints(decode_uint256_words(sliced).astype(np.uint64)) == [3, 4, 5, 6]
    chunked = pa.chunked_array([array.slice(0, 2), array.slice(7, 3)])
    assert limbs_to_ints(decode_uint256_words(chunked).astype(np.uint64)) == [
        0,
        1,
        7,
        8,
        9,
    ]


def test_decode_accepts_large_string(factory: "LogFactory") -> None:
    array = pa.array([factory.word(7)], type=pa.large_string())
    assert limbs_to_ints(decode_uint256_words(array).astype(np.uint64)) == [7]


def test_decode_property_matches_int_parsing(
    factory: "LogFactory", rng: random.Random
) -> None:  # T12
    values = [rng.getrandbits(rng.choice([8, 64, 128, 255, 256])) for _ in range(1000)]
    assert decode([factory.word(v) for v in values]) == values


def test_limbs_to_ints_handles_carries_beyond_32_bits() -> None:
    # Summed limbs exceed 32 bits and must carry into the more significant limb.
    sums = np.array([[0, 0, 0, 0, 0, 0, 2**33, 2**40]], dtype=np.uint64)
    assert limbs_to_ints(sums) == [(2**33 << 32) + 2**40]


def test_limbs_to_ints_can_exceed_uint256() -> None:
    sums = np.array([[2 * (2**32 - 1)] * 8], dtype=np.uint64)
    assert limbs_to_ints(sums) == [2 * (2**256 - 1)]


@pytest.mark.parametrize("shape", [(2, 7), (8,), (1, 8, 1)])
def test_limbs_to_ints_rejects_wrong_shape(shape: tuple[int, ...]) -> None:
    sums = np.zeros(shape, dtype=np.uint64)
    with pytest.raises(ValueError, match="shape"):
        limbs_to_ints(sums)


# --- erc20_transfer_filter (F1-F8) --------------------------------------------


def test_topic0_constant_matches_fixture(factory: "LogFactory") -> None:
    assert TRANSFER_TOPIC0 == factory.TRANSFER


def test_filter_keeps_listed_erc20_transfer(factory: "LogFactory") -> None:  # F1
    table = factory.table([factory.erc20(factory.TOKEN_A, factory.DAY_1, 5)])
    kept = table.filter(erc20_transfer_filter(pa.array([factory.TOKEN_A])))
    assert kept.num_rows == 1
    assert decode(kept["DATA"].to_pylist()) == [5]


@pytest.mark.parametrize(
    "kind",
    [
        pytest.param("erc721", id="F2-erc721"),
        pytest.param("approval", id="F3-approval"),
        pytest.param("weth_deposit", id="F4-weth-deposit"),
        pytest.param("ens_transfer", id="F5-transfer-named-other-signature"),
        pytest.param("unlisted", id="F6-unlisted-token"),
        pytest.param("non_indexed", id="F7-non-indexed-transfer"),
    ],
)
def test_filter_drops_everything_else(factory: "LogFactory", kind: str) -> None:
    a, day = factory.TOKEN_A, factory.DAY_1
    log = {
        "erc721": factory.erc721(a, day, token_id=42),
        "approval": factory.approval(a, day, 5),
        "weth_deposit": factory.weth_deposit(a, day, 5),
        "ens_transfer": factory.ens_transfer(a, day),
        "unlisted": factory.erc20(factory.UNLISTED, day, 5),
        "non_indexed": factory.non_indexed_transfer(a, day, 5),
    }[kind]
    table = factory.table([log])
    assert table.filter(erc20_transfer_filter(pa.array([a]))).num_rows == 0


@pytest.mark.parametrize(
    "params",
    [
        pytest.param('{"from": "0x1", "to": "0x2", "tokenId": "9"}', id="tokenId"),
        pytest.param('{"_from": "0x1", "_to": "0x2", "amount": "9"}', id="amount"),
        pytest.param("{}", id="empty"),
        pytest.param("not json", id="malformed"),
        pytest.param(None, id="null"),
    ],
)
def test_filter_ignores_params_labels(
    factory: "LogFactory", params: str | None
) -> None:
    # F8: the decoder's PARAMS labels vary per row; the amount comes from DATA.
    log = replace(factory.erc20(factory.TOKEN_A, factory.DAY_1, 77), PARAMS=params)
    table = factory.table([log])
    kept = table.filter(erc20_transfer_filter(pa.array([factory.TOKEN_A])))
    assert decode(kept["DATA"].to_pylist()) == [77]


def test_filter_without_tokens_keeps_all_erc20_transfers(factory: "LogFactory") -> None:
    table = factory.table(
        [
            factory.erc20(factory.TOKEN_A, factory.DAY_1, 1),
            factory.erc20(factory.UNLISTED, factory.DAY_1, 2),
            factory.erc721(factory.TOKEN_A, factory.DAY_1, 3),
        ]
    )
    assert table.filter(erc20_transfer_filter())["ADDRESS"].to_pylist() == [
        factory.TOKEN_A,
        factory.UNLISTED,
    ]


def test_filter_keeps_self_and_zero_value_transfers(factory: "LogFactory") -> None:
    a, day = factory.TOKEN_A, factory.DAY_1
    table = factory.table(
        [
            factory.erc20(a, day, 0),
            factory.erc20(a, day, 3, sender=factory.SENDER, recipient=factory.SENDER),
        ]
    )
    assert table.filter(erc20_transfer_filter(pa.array([a]))).num_rows == 2


def test_filter_pushes_down_into_a_spark_style_dataset(
    factory: "LogFactory", tmp_path: Path
) -> None:
    a, b, day = factory.TOKEN_A, factory.TOKEN_B, factory.DAY_2
    path = factory.write_dataset(
        tmp_path / "logs",
        [
            factory.erc20(a, day, 1),
            factory.erc721(a, day, 2),
            factory.approval(b, day, 3),
            factory.erc20(b, day, 4),
            factory.erc20(factory.UNLISTED, day, 5),
        ],
    )
    dataset = ds.dataset(path, format="parquet")
    table = dataset.to_table(
        columns=["BLOCK_DATE", "ADDRESS", "DATA"],
        filter=erc20_transfer_filter(pa.array([a, b])),
    )
    assert sorted(decode(table["DATA"].to_pylist())) == [1, 4]
    assert table.column_names == ["BLOCK_DATE", "ADDRESS", "DATA"]
