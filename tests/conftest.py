"""Shared fixtures: tiny synthetic decoded logs, token lists and an oracle.

Nothing here reads the real data. Token addresses are ``"0x"`` followed by one
letter repeated 40 times; the `tokens` fixture lists ``a``, ``b`` and ``c`` as
named tokens, ``d`` without a name and ``e`` without a symbol, and ``f`` is not
listed at all.

Two factories build logs at the two levels the code works on:

* `make_logs` returns a pandas DataFrame shaped like the real parquet files,
  for end-to-end tests that write inputs to disk.
* `factory` (a `LogFactory`) returns Arrow tables of every log shape found in
  the real data (ERC-20, ERC-721, Approval, ...), for unit tests. Test modules
  cannot import ``conftest`` under ``--import-mode=importlib``, so the shared
  constants and helpers are exposed on that fixture object.

`oracle` computes expected results with plain Python integers and sorting.
"""

import json
import random
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Final

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

# keccak256 of the event signatures, as found in TOPIC0
TRANSFER_TOPIC: Final = (
    "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"
)
APPROVAL_TOPIC: Final = (
    "0x8c5be1e5ebec7d5bd14f71427d1e84f3dd0314c0f7b2291e5b200ac8c7c3b925"
)
WETH_DEPOSIT_TOPIC: Final = (
    "0xe1fffcc4923d04b559f4d29a8bfc6cda04eb5b0d3c460751c2402c5c5cc9109c"
)
ENS_TRANSFER_TOPIC: Final = (
    "0xd4735d920b0f87494915f556dd9b54c8f309026070caea5c737245152564d266"
)
SENDER: Final = "0x" + "1" * 40
RECIPIENT: Final = "0x" + "2" * 40

A, B, C, UNNAMED, NO_SYMBOL, UNLISTED = ("0x" + c * 40 for c in "abcdef")
DAY_1, DAY_2, DAY_3 = date(2023, 4, 1), date(2023, 4, 2), date(2023, 4, 3)

# (token address, block date, amount) for a Transfer, or with a 4th item, the
# decoded event name ("Transfer" or "Approval")
type Event = tuple[str, date, int] | tuple[str, date, int, str]
type MakeLogs = Callable[..., pd.DataFrame]
type MakeTokens = Callable[..., pd.DataFrame]
type WriteInputs = Callable[[pd.DataFrame, pd.DataFrame], tuple[Path, Path]]
type AggregateKey = tuple[str, str]
"""``(ISO day, token address)``."""


def word(value: int) -> str:
    """Hex-encode `value` as one 32-byte ABI word, as in the ``DATA`` column."""
    return "0x" + format(value, "064x")


def topic(address: str) -> str:
    """Left-pad a 20-byte address to a 32-byte indexed topic."""
    return "0x" + address.removeprefix("0x").rjust(64, "0")


# --- pandas DataFrames shaped like the real parquet files ---------------------


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
        "DATA": word(amount),
        "TOPIC0": topic0,
        "TOPIC1": topic(SENDER),
        "TOPIC2": topic(RECIPIENT),
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


# --- Arrow tables of every log shape, for unit tests -------------------------

_LOG_COLUMNS: Final[list[tuple[str, pa.DataType]]] = [
    ("BLOCK_DATE", pa.date32()),
    *(
        (name, pa.string())
        for name in ("ADDRESS", "DATA", "TOPIC0", "TOPIC1", "TOPIC2", "TOPIC3")
    ),
    ("NAME", pa.string()),
    ("PARAMS", pa.string()),
]
LOG_SCHEMA: Final = pa.schema(_LOG_COLUMNS)


@dataclass(frozen=True)
class Log:
    """One decoded log row with the columns the pipeline can see."""

    BLOCK_DATE: date
    ADDRESS: str
    DATA: str
    TOPIC0: str | None
    TOPIC1: str | None = None
    TOPIC2: str | None = None
    TOPIC3: str | None = None
    NAME: str | None = None
    PARAMS: str | None = None


class LogFactory:
    """Build synthetic logs of every shape the real dataset contains."""

    TRANSFER: Final = TRANSFER_TOPIC
    TOKEN_A: Final = A
    TOKEN_B: Final = B
    UNLISTED: Final = UNLISTED
    SENDER: Final = SENDER
    RECIPIENT: Final = RECIPIENT
    DAY_1: Final = DAY_1
    DAY_2: Final = DAY_2
    word = staticmethod(word)
    topic = staticmethod(topic)

    @staticmethod
    def erc20(
        token: str,
        day: date,
        amount: int,
        *,
        sender: str = SENDER,
        recipient: str = RECIPIENT,
    ) -> Log:
        """Return an ERC-20 ``Transfer``: 3 topics, amount in one data word."""
        params = json.dumps({"from": sender, "to": recipient, "value": str(amount)})
        return Log(
            BLOCK_DATE=day,
            ADDRESS=token,
            DATA=word(amount),
            TOPIC0=TRANSFER_TOPIC,
            TOPIC1=topic(sender),
            TOPIC2=topic(recipient),
            NAME="Transfer",
            PARAMS=params,
        )

    @staticmethod
    def erc721(token: str, day: date, token_id: int) -> Log:
        """Return an ERC-721 ``Transfer``: same topic0, tokenId in TOPIC3, no data."""
        params = {"from": SENDER, "to": RECIPIENT, "tokenId": str(token_id)}
        return Log(
            BLOCK_DATE=day,
            ADDRESS=token,
            DATA="0x",
            TOPIC0=TRANSFER_TOPIC,
            TOPIC1=topic(SENDER),
            TOPIC2=topic(RECIPIENT),
            TOPIC3=word(token_id),
            NAME="Transfer",
            PARAMS=json.dumps(params),
        )

    @staticmethod
    def non_indexed_transfer(token: str, day: date, value: int) -> Log:
        """Return a pre-standard ``Transfer`` with no indexed args (CryptoKitties)."""
        data = "0x" + "".join(word(v)[2:] for v in (int(SENDER, 16), 0, value))
        return Log(
            BLOCK_DATE=day,
            ADDRESS=token,
            DATA=data,
            TOPIC0=TRANSFER_TOPIC,
            NAME="Transfer",
        )

    @staticmethod
    def approval(token: str, day: date, amount: int) -> Log:
        """Return an ERC-20 ``Approval`` (same shape as a transfer, other topic0)."""
        return Log(
            BLOCK_DATE=day,
            ADDRESS=token,
            DATA=word(amount),
            TOPIC0=APPROVAL_TOPIC,
            TOPIC1=topic(SENDER),
            TOPIC2=topic(RECIPIENT),
            NAME="Approval",
        )

    @staticmethod
    def weth_deposit(token: str, day: date, amount: int) -> Log:
        """Return a WETH ``Deposit(address indexed dst, uint wad)``."""
        return Log(
            BLOCK_DATE=day,
            ADDRESS=token,
            DATA=word(amount),
            TOPIC0=WETH_DEPOSIT_TOPIC,
            TOPIC1=topic(RECIPIENT),
            NAME="Deposit",
        )

    @staticmethod
    def ens_transfer(token: str, day: date) -> Log:
        """Return ENS ``Transfer(bytes32 indexed node, address owner)``."""
        return Log(
            BLOCK_DATE=day,
            ADDRESS=token,
            DATA=word(int(RECIPIENT, 16)),
            TOPIC0=ENS_TRANSFER_TOPIC,
            TOPIC1=word(12345),
            NAME="Transfer",
        )

    @staticmethod
    def table(logs: Iterable[Log]) -> pa.Table:
        """Return `logs` as a table with the dataset's column names and types."""
        return pa.Table.from_pylist([asdict(log) for log in logs], schema=LOG_SCHEMA)

    def write_dataset(
        self,
        directory: Path,
        logs: Iterable[Log],
        *,
        parts: int = 2,
        empty_part: bool = True,
        success_marker: bool = True,
    ) -> Path:
        """Write `logs` as a Spark-style parquet directory and return its path."""
        table = self.table(logs)
        directory.mkdir(parents=True, exist_ok=True)
        step = -(-table.num_rows // parts) if table.num_rows else 0
        for part in range(parts):
            chunk = table.slice(part * step, step) if step else table.slice(0, 0)
            pq.write_table(chunk, directory / f"part-{part + 1:05d}.snappy.parquet")
        if empty_part:
            pq.write_table(table.slice(0, 0), directory / "part-00000.snappy.parquet")
        if success_marker:
            (directory / "_SUCCESS").touch()
            (directory / "_committed_1").write_text("{}")
        return directory


class Oracle:
    """Reference results computed with plain Python integers and sorting."""

    @staticmethod
    def aggregate(logs: Iterable[Log]) -> dict[AggregateKey, tuple[int, int]]:
        """Return ``{(day, token): (transfer count, exact amount sum)}``."""
        totals: defaultdict[AggregateKey, list[int]] = defaultdict(lambda: [0, 0])
        for log in logs:
            entry = totals[log.BLOCK_DATE.isoformat(), log.ADDRESS]
            entry[0] += 1
            entry[1] += int(log.DATA, 16)
        return {key: (count, amount) for key, (count, amount) in totals.items()}

    @staticmethod
    def rank(
        values: Mapping[AggregateKey, int],
        *,
        per_day: int | None,
        limit: int | None,
    ) -> list[tuple[str, str, int]]:
        """Return the top ``(day, token, value)`` rows by brute force."""
        by_day: defaultdict[str, list[tuple[str, int]]] = defaultdict(list)
        for (day, token), value in values.items():
            by_day[day].append((token, value))
        kept: list[tuple[str, str, int]] = []
        for day, tokens in by_day.items():
            best = sorted(tokens, key=lambda item: (-item[1], item[0]))
            kept.extend((day, token, value) for token, value in best[:per_day])
        kept.sort(key=lambda row: (-row[2], row[0], row[1]))
        return kept if limit is None else kept[:limit]


@pytest.fixture
def factory() -> LogFactory:
    """Factory for synthetic logs of every shape (Arrow tables)."""
    return LogFactory()


@pytest.fixture
def oracle() -> Oracle:
    """Plain-Python reference implementation."""
    return Oracle()


@pytest.fixture
def rng() -> random.Random:
    """Seeded random generator so property tests are reproducible."""
    return random.Random(20230423)  # noqa: S311 - test data, not cryptography
