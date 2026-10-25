"""
Aggregate ERC-20 transfers per day and token, exactly and in bounded memory.

Transfers arrive in record batches (columns ``BLOCK_DATE``, ``ADDRESS``,
``DATA``). Each batch is reduced to one row per ``(day, token)`` holding the
transfer count and the sum of each 32-bit limb of the amounts
(`partial_aggregate`); the partial results are then summed and the limbs
recombined into exact Python integers (`combine_partials`). Limb sums are
exact in ``uint64`` while a group has fewer than ``2**32`` transfers, so no
amount is ever rounded or truncated, even when a daily sum exceeds
``2**256``.
"""

from collections.abc import Iterable
from typing import Final

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

from ethereum_transfers.transfers import (
    LIMB_COUNT,
    decode_uint256_words,
    limbs_to_ints,
)

INPUT_COLUMNS: Final = ("BLOCK_DATE", "ADDRESS", "DATA")
"""Columns `partial_aggregate` needs from each batch of transfer logs."""

AGGREGATE_COLUMNS: Final = (
    "day",
    "token_address",
    "transfer_count",
    "transfer_amount",
)
"""Columns of the frame returned by `combine_partials` and `aggregate_transfers`."""

MAX_GROUP_ROWS: Final = 2**32 - 1
"""Largest group for which ``uint64`` limb sums are guaranteed exact."""

_KEYS: Final = ["BLOCK_DATE", "ADDRESS"]
_COUNT: Final = "n"
_LIMBS: Final = [f"limb{i}" for i in range(LIMB_COUNT)]


def partial_aggregate(batch: pa.RecordBatch | pa.Table) -> pa.Table:
    """
    Reduce one batch of transfer logs to per-``(day, token)`` partial sums.

    Parameters
    ----------
    batch : pa.RecordBatch or pa.Table
        ERC-20 transfer logs with at least the columns ``BLOCK_DATE``,
        ``ADDRESS`` and ``DATA`` (see
        `ethereum_transfers.transfers.erc20_transfer_filter`).

    Returns
    -------
    pa.Table
        One row per distinct ``(BLOCK_DATE, ADDRESS)`` with the columns
        ``BLOCK_DATE``, ``ADDRESS``, ``n`` (``uint64`` transfer count) and
        ``limb0`` ... ``limb7`` (``uint64`` limb sums).

    Raises
    ------
    ValueError
        If a required column is missing or a ``DATA`` value is not one
        hex-encoded 32-byte word.
    """
    missing = [name for name in INPUT_COLUMNS if name not in batch.schema.names]
    if missing:
        raise ValueError(f"transfer batch is missing column(s): {', '.join(missing)}")
    limbs = decode_uint256_words(batch.column("DATA")).astype(np.uint64)
    table = batch if isinstance(batch, pa.Table) else pa.Table.from_batches([batch])
    table = table.select(_KEYS).append_column(
        _COUNT, pa.array(np.ones(len(limbs), dtype=np.uint64))
    )
    for i, name in enumerate(_LIMBS):
        table = table.append_column(name, pa.array(np.ascontiguousarray(limbs[:, i])))
    return _sum_by_key(table)


def combine_partials(
    partials: Iterable[pa.Table], *, max_group_rows: int = MAX_GROUP_ROWS
) -> pd.DataFrame:
    """
    Sum partial aggregates and recombine the limbs into exact amounts.

    Parameters
    ----------
    partials : Iterable[pa.Table]
        Results of `partial_aggregate`, in any order; the same
        ``(day, token)`` may appear in several of them.
    max_group_rows : int, default 2**32 - 1
        Largest number of transfers allowed in one ``(day, token)`` group.
        The default is the bound below which ``uint64`` limb sums are exact.

    Returns
    -------
    pd.DataFrame
        Columns ``day`` (ISO ``YYYY-MM-DD`` string), ``token_address`` (string),
        ``transfer_count`` (``int64``) and ``transfer_amount`` (``object``
        holding exact Python ``int`` sums in raw base units), sorted by
        ``day`` then ``token_address``.

    Raises
    ------
    OverflowError
        If a group has more than `max_group_rows` transfers.
    """
    tables = [table for table in partials if table.num_rows]
    if not tables:
        return _empty_aggregate()
    totals = _sum_by_key(pa.concat_tables(tables))
    counts = totals[_COUNT].to_numpy()
    if int(counts.max()) > max_group_rows:
        raise OverflowError(
            f"a (day, token) group has {int(counts.max())} transfers; limb sums "
            f"are only exact up to {max_group_rows}"
        )
    limb_sums = np.column_stack([totals[name].to_numpy() for name in _LIMBS])
    frame = pd.DataFrame(
        {
            "day": pc.cast(totals["BLOCK_DATE"], pa.string()).to_pylist(),
            "token_address": totals["ADDRESS"].to_pylist(),
            "transfer_count": counts.astype(np.int64),
            "transfer_amount": pd.Series(
                limbs_to_ints(limb_sums.astype(np.uint64)), dtype=object
            ),
        }
    )
    return frame.sort_values(["day", "token_address"], ignore_index=True)


def aggregate_transfers(
    batches: Iterable[pa.RecordBatch | pa.Table],
    *,
    max_group_rows: int = MAX_GROUP_ROWS,
) -> pd.DataFrame:
    """
    Aggregate a stream of transfer-log batches per day and token.

    Only one batch and the (small) partial results are held in memory at a
    time, so memory does not grow with the number of input rows.

    Parameters
    ----------
    batches : Iterable[pa.RecordBatch or pa.Table]
        ERC-20 transfer logs, for example from a filtered dataset scan.
    max_group_rows : int, default 2**32 - 1
        Passed to `combine_partials`.

    Returns
    -------
    pd.DataFrame
        See `combine_partials`.
    """
    return combine_partials(
        (partial_aggregate(batch) for batch in batches if batch.num_rows),
        max_group_rows=max_group_rows,
    )


def _sum_by_key(table: pa.Table) -> pa.Table:
    """Sum the count and limb columns per ``(BLOCK_DATE, ADDRESS)``."""
    summed = table.group_by(_KEYS, use_threads=False).aggregate(
        [(name, "sum") for name in (_COUNT, *_LIMBS)]
    )
    # pyarrow names aggregates "<column>_sum"; keep the input names so that
    # partial results can be summed again.
    return summed.rename_columns(
        [name.removesuffix("_sum") for name in summed.column_names]
    )


def _empty_aggregate() -> pd.DataFrame:
    """Return an aggregate frame with no rows and the documented dtypes."""
    return pd.DataFrame(
        {
            "day": pd.Series([], dtype="str"),
            "token_address": pd.Series([], dtype="str"),
            "transfer_count": pd.Series([], dtype=np.int64),
            "transfer_amount": pd.Series([], dtype=object),
        }
    )
