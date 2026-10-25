"""Rank token addresses per day by a transfer metric.

The functions here are pure: they take and return DataFrames and do no IO.

``count_of_transfers``
    Number of log rows per token address per day.
``quantity_transferred``
    Sum of the transferred ``value`` per token address per day.

Each metric keeps the top token address of each day, ordered by the metric
descending and limited to `TOP_N` rows.
"""

import json
from enum import StrEnum
from typing import Final, assert_never

import pandas as pd

TOP_N: Final = 100


class Metric(StrEnum):
    """Transfer metric used to rank token addresses per day."""

    COUNT_OF_TRANSFERS = "count_of_transfers"
    QUANTITY_TRANSFERRED = "quantity_transferred"


def join_tokens(logs: pd.DataFrame, tokens: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only the log rows emitted by a named token from the token list.

    Parameters
    ----------
    logs : pd.DataFrame
        Decoded logs with at least ``ADDRESS``, ``PARAMS`` and ``BLOCK_DATE``.
    tokens : pd.DataFrame
        Token list as read from ``tokens.csv`` (``address``, ``name``,
        ``symbol``, ...).

    Returns
    -------
    pd.DataFrame
        `logs` joined to `tokens` on the token address, with a fresh
        ``RangeIndex``.

    Notes
    -----
    Assumes every token in the list that has both a ``name`` and a ``symbol``
    is an ERC-20 token. Rows missing ``ADDRESS``, ``PARAMS``, ``BLOCK_DATE`` or
    a token ``name`` are dropped.
    """
    named_tokens = tokens.dropna(subset=["name", "symbol"], ignore_index=True).rename(
        columns={"address": "ADDRESS"}
    )
    return logs.merge(named_tokens, how="left", on="ADDRESS").dropna(
        subset=["ADDRESS", "PARAMS", "BLOCK_DATE", "name"], ignore_index=True
    )


def count_of_transfers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Find the token address with the most transfers on each day.

    Parameters
    ----------
    df : pd.DataFrame
        Token logs as returned by `join_tokens`.

    Returns
    -------
    pd.DataFrame
        Columns ``ADDRESS``, ``BLOCK_DATE`` and ``count_of_transfers``, one row
        per day, ordered by ``count_of_transfers`` descending, at most `TOP_N`
        rows.
    """
    count_df = (
        df.groupby(["ADDRESS", "BLOCK_DATE"]).size().to_frame("count_of_transfers")
    )
    # number the addresses within each day by count of transfers, descending
    count_df["row_num"] = (
        count_df.sort_values(by=["count_of_transfers", "BLOCK_DATE"], ascending=False)
        .groupby("BLOCK_DATE")
        .cumcount()
        + 1
    )
    top_transferred_address_per_day = count_df[count_df["row_num"] == 1].drop(
        columns="row_num"
    )
    result = top_transferred_address_per_day.sort_values(
        by="count_of_transfers", ascending=False
    ).head(TOP_N)
    return result.reset_index()


def quantity_transferred(df: pd.DataFrame) -> pd.DataFrame:
    """
    Find the token address with the largest quantity transferred on each day.

    Parameters
    ----------
    df : pd.DataFrame
        Token logs as returned by `join_tokens`.

    Returns
    -------
    pd.DataFrame
        Columns ``ADDRESS``, ``BLOCK_DATE`` and ``quantity_transferred``, one
        row per day, ordered by ``quantity_transferred`` descending, at most
        `TOP_N` rows.
    """
    transfers = df[["ADDRESS", "BLOCK_DATE", "PARAMS"]]
    # PARAMS is the JSON string of the decoded event arguments; for a transfer
    # its keys are ``from``, ``to`` and ``value`` (the amount transferred).
    # https://medium.com/@eiki1212/ethereum-transaction-structure-explained-aa5a94182ad6
    transfers["PARAMS"] = transfers["PARAMS"][1]
    transfers["PARAMS"] = transfers["PARAMS"].apply(json.loads)
    transfers["value_transferred"] = transfers["PARAMS"].apply(
        lambda params: params["value"]
    )
    # sum the value transferred per address per day and broadcast it to the rows
    # as a categorical (per-row codes into the per-group sums). Under pandas 3 a
    # ``str`` column -- what ``transform("sum")`` returns, and what sorting turns
    # an object column into -- stores its own copy of the group's sum in every
    # row. The categories are sorted, so the ordering below is unchanged.
    quantity = (
        transfers.groupby(["ADDRESS", "BLOCK_DATE"])["value_transferred"]
        .sum()
        .astype("category")
        .rename("quantity_transferred")
    )
    transfers = transfers.join(quantity, on=["ADDRESS", "BLOCK_DATE"])
    # number the rows within each day by quantity transferred, descending
    transfers["row_num"] = (
        transfers.sort_values(
            by=["BLOCK_DATE", "quantity_transferred"], ascending=False
        )
        .groupby("BLOCK_DATE")
        .cumcount()
        + 1
    )
    # back to plain values for the final ordering of the (one per day) top rows
    top_transferred_address_per_day = (
        transfers[transfers["row_num"] == 1]
        .drop(columns=["row_num", "PARAMS", "value_transferred"])
        .astype({"quantity_transferred": object})
    )
    result = top_transferred_address_per_day.sort_values(
        by="quantity_transferred", ascending=False
    ).head(TOP_N)
    return result.reset_index(drop=True)


def compute_metric(df: pd.DataFrame, metric: Metric) -> pd.DataFrame:
    """
    Compute the top token address per day for the given metric.

    Parameters
    ----------
    df : pd.DataFrame
        Token logs as returned by `join_tokens`.
    metric : Metric
        Metric to rank token addresses by.

    Returns
    -------
    pd.DataFrame
        Columns ``ADDRESS``, ``BLOCK_DATE`` and a column named after `metric`.
    """
    match metric:
        case Metric.COUNT_OF_TRANSFERS:
            return count_of_transfers(df)
        case Metric.QUANTITY_TRANSFERRED:
            return quantity_transferred(df)
        case _:
            assert_never(metric)
