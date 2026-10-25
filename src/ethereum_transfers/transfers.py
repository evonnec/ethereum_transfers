"""
Identify ERC-20 ``Transfer`` logs and decode their amounts exactly.

An ERC-20 transfer is the event ``Transfer(address indexed from, address indexed
to, uint256 value)``. Its log carries the event signature hash in ``TOPIC0``,
the two indexed addresses in ``TOPIC1`` and ``TOPIC2``, and the amount as a
single 32-byte ABI word in ``DATA``. ERC-721 emits an event with the *same*
signature hash, but its ``tokenId`` is indexed too: it sits in ``TOPIC3`` and
``DATA`` is empty. The log *shape* therefore tells the two standards apart.

The decoded ``NAME`` and ``PARAMS`` columns are deliberately not used: ``NAME``
is shared by unrelated events and the key names inside ``PARAMS`` vary even for
a single token contract, whereas ``DATA`` is the canonical encoding.

References
----------
https://goethereumbook.org/en/event-read-erc20/
"""

from __future__ import annotations

from typing import Any, Final, cast

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds

TRANSFER_TOPIC0: Final = (
    "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"
)
"""``keccak256("Transfer(address,address,uint256)")``, shared by ERC-20 and ERC-721."""

WORD_HEX_LENGTH: Final = 66
"""Length of one hex-encoded 32-byte ABI word: ``"0x"`` plus 64 hex digits."""

WORD_BYTES: Final = 32
LIMB_COUNT: Final = 8
"""A uint256 is split into eight 32-bit limbs, most significant first."""

LIMB_BITS: Final = 32


def erc20_transfer_filter(
    token_addresses: pa.Array[Any] | None = None,
) -> ds.Expression:
    """
    Build the filter expression that selects ERC-20 ``Transfer`` logs.

    A log matches when ``TOPIC0`` is the ``Transfer`` signature, ``TOPIC3`` is
    null and ``DATA`` is exactly one 32-byte word. The event has three
    arguments, so one data word implies that ``TOPIC1`` and ``TOPIC2`` hold the
    two indexed addresses; checking them would only cost reading two wide
    columns.

    Parameters
    ----------
    token_addresses : pa.Array or None, optional
        Lower-case token contract addresses. When given, only logs emitted by
        these contracts (``ADDRESS``) match.

    Returns
    -------
    ds.Expression
        Expression usable as ``pyarrow.dataset.Dataset.to_batches(filter=...)``
        (pushed down to the parquet scan) or ``pyarrow.Table.filter``.
    """
    expression = (
        (ds.field("TOPIC0") == TRANSFER_TOPIC0)
        & ds.field("TOPIC3").is_null()
        & (pc.utf8_length(ds.field("DATA")) == WORD_HEX_LENGTH)
    )
    if token_addresses is not None:
        expression &= ds.field("ADDRESS").isin(token_addresses)
    return expression


def decode_uint256_words(
    data: pa.Array[Any] | pa.ChunkedArray[Any],
) -> npt.NDArray[np.uint32]:
    """
    Decode hex-encoded 32-byte words into 32-bit limbs, without a Python loop.

    Parameters
    ----------
    data : pa.Array or pa.ChunkedArray
        String values of the form ``"0x"`` followed by exactly 64 hex digits,
        as found in the ``DATA`` column of ERC-20 ``Transfer`` logs.

    Returns
    -------
    npt.NDArray[np.uint32]
        Array of shape ``(len(data), 8)``; row ``i`` holds the limbs of value
        ``i``, most significant first, so that
        ``value == sum(limb << 32 * (7 - j) for j, limb in enumerate(row))``.

    Raises
    ------
    ValueError
        If a value is null, does not start with ``"0x"``, is not exactly 66
        characters long, or contains a non-hex character.
    """
    array = data.combine_chunks() if isinstance(data, pa.ChunkedArray) else data
    if pa.types.is_string(array.type):
        array = array.cast(pa.large_string())
    elif not pa.types.is_large_string(array.type):
        raise ValueError(f"expected a string array, got {array.type}")
    n_rows = len(array)
    if n_rows == 0:
        return np.zeros((0, LIMB_COUNT), dtype=np.uint32)
    if array.null_count:
        raise ValueError(f"{array.null_count} null value(s) in a uint256 word column")
    # A non-empty large_string array always has an int64 offsets buffer; the
    # values buffer exists once every value is known to be 66 bytes long.
    buffers = array.buffers()
    offsets = np.frombuffer(
        cast("pa.Buffer", buffers[1]),
        dtype=np.int64,
        count=n_rows + 1,
        offset=array.offset * 8,
    )
    if not (np.diff(offsets) == WORD_HEX_LENGTH).all():
        raise ValueError(
            f"every uint256 word must be {WORD_HEX_LENGTH} characters ('0x' + 64 hex)"
        )
    start = int(offsets[0])
    chars = np.frombuffer(
        cast("pa.Buffer", buffers[2]),
        dtype=np.uint8,
        count=n_rows * WORD_HEX_LENGTH,
        offset=start,
    ).reshape(n_rows, WORD_HEX_LENGTH)
    if not ((chars[:, 0] == ord("0")) & (chars[:, 1] == ord("x"))).all():
        raise ValueError("every uint256 word must start with '0x'")
    try:
        raw = bytes.fromhex(chars[:, 2:].tobytes().decode("ascii"))
    except ValueError as error:  # includes UnicodeDecodeError
        raise ValueError(f"non-hex character in a uint256 word: {error}") from error
    if len(raw) != n_rows * WORD_BYTES:  # bytes.fromhex skips ASCII whitespace
        raise ValueError("non-hex character in a uint256 word")
    return np.frombuffer(raw, dtype=">u4").reshape(n_rows, LIMB_COUNT).astype(np.uint32)


def limbs_to_ints(limb_sums: npt.NDArray[np.uint64]) -> list[int]:
    """
    Recombine per-limb sums into exact Python integers.

    Parameters
    ----------
    limb_sums : npt.NDArray[np.uint64]
        Array of shape ``(n, 8)``: for each of ``n`` values (or groups), the
        sum of each 32-bit limb, most significant limb first, as returned by
        `decode_uint256_words` or by summing its rows.

    Returns
    -------
    list[int]
        ``sum(limb_sums[i, j] << 32 * (7 - j) for j in range(8))`` for each row
        ``i``, computed with arbitrary precision.

    Raises
    ------
    ValueError
        If `limb_sums` does not have shape ``(n, 8)``.
    """
    if limb_sums.ndim != 2 or limb_sums.shape[1] != LIMB_COUNT:  # noqa: PLR2004
        raise ValueError(
            f"expected limb sums of shape (n, {LIMB_COUNT}), got {limb_sums.shape}"
        )
    shifts = [LIMB_BITS * (LIMB_COUNT - 1 - j) for j in range(LIMB_COUNT)]
    return [
        sum(int(limb) << shift for limb, shift in zip(row, shifts, strict=True))
        for row in limb_sums.tolist()
    ]
