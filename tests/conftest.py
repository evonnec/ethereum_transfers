"""Shared fixtures: tiny synthetic decoded logs and token lists.

Nothing here reads the real data. Token addresses are ``"0x"`` followed by one
letter repeated 40 times; the `tokens` fixture lists ``a``, ``b`` and ``c`` as
named tokens, ``d`` without a name and ``e`` without a symbol, and ``f`` is not
listed at all.
"""

import json
from collections.abc import Callable, Iterable
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

# keccak256 of the ERC-20 event signatures, as found in TOPIC0
TRANSFER_TOPIC = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"
APPROVAL_TOPIC = "0x8c5be1e5ebec7d5bd14f71427d1e84f3dd0314c0f7b2291e5b200ac8c7c3b925"
SENDER = "0x" + "1" * 40
RECIPIENT = "0x" + "2" * 40

A, B, C, UNNAMED, NO_SYMBOL, UNLISTED = ("0x" + c * 40 for c in "abcdef")
DAY_1, DAY_2, DAY_3 = date(2023, 4, 1), date(2023, 4, 2), date(2023, 4, 3)

# (token address, block date, amount) for a Transfer, or with a 4th item, the
# decoded event name ("Transfer" or "Approval")
type Event = tuple[str, date, int] | tuple[str, date, int, str]
type MakeLogs = Callable[..., pd.DataFrame]
type MakeTokens = Callable[..., pd.DataFrame]
type WriteInputs = Callable[[pd.DataFrame, pd.DataFrame], tuple[Path, Path]]


def _topic(address: str) -> str:
    """Left-pad an address to a 32-byte indexed topic."""
    return "0x" + address.removeprefix("0x").rjust(64, "0")


def _log_row(index: int, event: Event) -> dict[str, object]:
    """Build one decoded log row shaped like the real parquet schema."""
    address, day, amount, *rest = event
    name = rest[0] if rest else "Transfer"
    if name == "Transfer":
        topic0 = TRANSFER_TOPIC
        params = {"from": SENDER, "to": RECIPIENT, "value": str(amount)}
    elif name == "Approval":
        topic0 = APPROVAL_TOPIC
        params = {"owner": SENDER, "spender": RECIPIENT, "value": str(amount)}
    else:
        raise ValueError(f"unsupported event {name!r}")
    return {
        "LOG_INDEX": index,
        "TRANSACTION_HASH": f"0x{index:064x}",
        "TRANSACTION_INDEX": 0,
        "ADDRESS": address,
        "DATA": f"0x{amount:064x}",
        "TOPIC0": topic0,
        "TOPIC1": _topic(SENDER),
        "TOPIC2": _topic(RECIPIENT),
        "TOPIC3": None,
        "BLOCK_TIMESTAMP": pd.Timestamp(day).as_unit("ns"),
        "BLOCK_NUMBER": 17_000_000 + index,
        "NAME": name,
        "PARAMS": json.dumps(params, indent=2),
        "BLOCK_DATE": day,
    }


@pytest.fixture
def make_logs() -> MakeLogs:
    """Return a factory: ``make_logs(*events)`` -> decoded logs DataFrame."""

    def make(*events: Event) -> pd.DataFrame:
        return pd.DataFrame([_log_row(i, event) for i, event in enumerate(events)])

    return make


@pytest.fixture
def make_tokens() -> MakeTokens:
    """Return a factory: ``make_tokens(*(address, symbol, name))`` -> token list.

    The columns match ``tokens.csv``; ``decimals`` is 0 so raw and scaled
    amounts are the same.
    """

    def make(*tokens: tuple[str, str | None, str | None]) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "address": address,
                    "symbol": symbol,
                    "name": name,
                    "decimals": 0,
                    "total_supply": 10**30,
                    "block_timestamp": "2020-01-01 00:00:00 UTC",
                    "block_number": 10_000_000,
                    "block_hash": "0x" + "9" * 64,
                }
                for address, symbol, name in tokens
            ]
        )

    return make


@pytest.fixture
def write_inputs(tmp_path: Path) -> WriteInputs:
    """Return a writer: ``write_inputs(logs, tokens)`` -> (parquet, CSV) paths."""

    def write(logs: pd.DataFrame, tokens: pd.DataFrame) -> tuple[Path, Path]:
        logs_path = tmp_path / "logs.parquet"
        tokens_path = tmp_path / "tokens.csv"
        logs.to_parquet(logs_path)
        tokens.to_csv(tokens_path, index=False)
        return logs_path, tokens_path

    return write


@pytest.fixture
def tokens(make_tokens: MakeTokens) -> pd.DataFrame:
    """Token list with three named tokens and two incomplete entries."""
    return make_tokens(
        (A, "AAA", "Token A"),
        (B, "BBB", "Token B"),
        (C, "CCC", "Token C"),
        (UNNAMED, "DDD", None),
        (NO_SYMBOL, None, "Token E"),
    )


@pytest.fixture
def logs(make_logs: MakeLogs) -> pd.DataFrame:
    """Three days of logs from named, incomplete and unlisted tokens."""
    events: Iterable[Event] = [
        (A, DAY_1, 5),
        (A, DAY_1, 5),
        (B, DAY_1, 5),
        (UNLISTED, DAY_1, 5),
        (B, DAY_2, 5),
        (B, DAY_2, 5),
        (C, DAY_2, 5, "Approval"),
        (UNNAMED, DAY_2, 5),
        (C, DAY_3, 5),
        (NO_SYMBOL, DAY_3, 5),
    ]
    return make_logs(*events)


@pytest.fixture
def inputs(
    logs: pd.DataFrame, tokens: pd.DataFrame, write_inputs: WriteInputs
) -> tuple[Path, Path]:
    """The `logs` and `tokens` fixtures written to a parquet and a CSV file."""
    return write_inputs(logs, tokens)
