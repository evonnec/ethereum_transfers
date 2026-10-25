"""
Transfer metrics and how each one ranks tokens per day.

``count_of_transfers``
    Number of ERC-20 ``Transfer`` logs per token per day.
``quantity_transferred``
    Exact sum of the transferred amounts per token per day, in the token's raw
    base units (no ``decimals`` scaling).

`compute_metric` is pure: it takes the per-``(day, token)`` aggregate from
`ethereum_transfers.aggregate.aggregate_transfers` and returns the ranked
result with the columns ``day``, ``token_address`` and the metric name.
"""

from enum import StrEnum
from typing import Final, assert_never

import pandas as pd

from ethereum_transfers.rank import DEFAULT_LIMIT, DEFAULT_PER_DAY, rank_top

TOP_N: Final = DEFAULT_LIMIT
"""Default maximum number of rows in a metric result."""


class Metric(StrEnum):
    """Transfer metric used to rank tokens per day; names the output file."""

    COUNT_OF_TRANSFERS = "count_of_transfers"
    QUANTITY_TRANSFERRED = "quantity_transferred"


def aggregate_column(metric: Metric) -> str:
    """
    Return the aggregate column a metric ranks by.

    Parameters
    ----------
    metric : Metric
        The metric.

    Returns
    -------
    str
        ``"transfer_count"`` or ``"transfer_amount"``.
    """
    match metric:
        case Metric.COUNT_OF_TRANSFERS:
            return "transfer_count"
        case Metric.QUANTITY_TRANSFERRED:
            return "transfer_amount"
        case _:
            assert_never(metric)


def compute_metric(
    aggregate: pd.DataFrame,
    metric: Metric,
    *,
    per_day: int | None = DEFAULT_PER_DAY,
    limit: int | None = TOP_N,
) -> pd.DataFrame:
    """
    Rank tokens per day by one metric.

    Parameters
    ----------
    aggregate : pd.DataFrame
        Per-``(day, token)`` aggregate as returned by
        `ethereum_transfers.aggregate.aggregate_transfers`.
    metric : Metric
        Metric to rank by.
    per_day : int or None, default 1
        Tokens kept per day; ``None`` ranks all ``(day, token)`` pairs together.
    limit : int or None, default `TOP_N`
        Maximum number of rows; ``None`` for no limit.

    Returns
    -------
    pd.DataFrame
        Columns ``day``, ``token_address`` and a column named after `metric`,
        ordered by the metric descending, then day and token address ascending.
        See `ethereum_transfers.rank.rank_top`.
    """
    column = aggregate_column(metric)
    ranked = rank_top(aggregate, column, per_day=per_day, limit=limit)
    return ranked.rename(columns={column: str(metric)})
