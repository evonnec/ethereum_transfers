"""
Rank tokens per day by a transfer metric, deterministically and exactly.

The default answers "the most transferred token per day": the top token of
each day, ordered by the metric descending, at most 100 rows. Ties are broken
by the lower token address, then the earlier day, so the output never depends
on input row order. Metric values are compared as exact integers, including
Python ``int`` values above ``2**64``.
"""

from typing import Final

import pandas as pd

DEFAULT_PER_DAY: Final = 1
DEFAULT_LIMIT: Final = 100

_RANK: Final = "_rank"


def rank_top(
    aggregate: pd.DataFrame,
    column: str,
    *,
    per_day: int | None = DEFAULT_PER_DAY,
    limit: int | None = DEFAULT_LIMIT,
) -> pd.DataFrame:
    """
    Select the top tokens per day by one metric column.

    Parameters
    ----------
    aggregate : pd.DataFrame
        One row per ``(day, token_address)`` with the metric in `column`, as
        returned by `ethereum_transfers.aggregate.aggregate_transfers`. The
        metric must hold non-null integers (``int64`` or Python ``int``).
    column : str
        Name of the metric column to rank by.
    per_day : int or None, default 1
        Number of tokens to keep per day; ``None`` keeps every token, which
        turns the result into a global ranking of ``(day, token)`` pairs.
    limit : int or None, default 100
        Maximum number of rows returned; ``None`` returns all of them.

    Returns
    -------
    pd.DataFrame
        Columns ``day``, ``token_address`` and `column`, ordered by `column`
        descending, then ``day`` and ``token_address`` ascending, with a fresh
        ``RangeIndex``. Within a day, tokens with equal values are ranked by
        ``token_address`` ascending.

    Raises
    ------
    ValueError
        If `per_day` is less than 1, `limit` is negative, a required column is
        missing, or `column` holds missing values.
    """
    if per_day is not None and per_day < 1:
        raise ValueError(f"per_day must be at least 1 or None, got {per_day}")
    if limit is not None and limit < 0:
        raise ValueError(f"limit must be non-negative or None, got {limit}")
    missing = [
        name for name in ("day", "token_address", column) if name not in aggregate
    ]
    if missing:
        raise ValueError(f"aggregate is missing column(s): {', '.join(missing)}")
    ranked = aggregate.loc[:, ["day", "token_address", column]]
    if ranked[column].isna().any():
        raise ValueError(f"column {column!r} has missing values")
    # Dense rank of the exact values, 0 = largest. Ranking through Python
    # objects keeps integers beyond int64 exact (no float conversion).
    distinct = sorted(set(ranked[column].tolist()), reverse=True)
    position = {value: rank for rank, value in enumerate(distinct)}
    ranked[_RANK] = [position[value] for value in ranked[column].tolist()]
    ranked = ranked.sort_values(["day", _RANK, "token_address"], kind="stable")
    if per_day is not None:
        ranked = ranked.groupby("day", sort=False).head(per_day)
    ranked = ranked.sort_values([_RANK, "day", "token_address"], kind="stable")
    if limit is not None:
        ranked = ranked.head(limit)
    return ranked.drop(columns=_RANK).reset_index(drop=True)
