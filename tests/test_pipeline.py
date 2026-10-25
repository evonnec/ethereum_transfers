"""End-to-end metric results: parquet + token CSV in, metric CSVs out.

These go through `generate_output` and read the CSVs back, so they hold however
the computation is split between functions. The tests that used to be strict
``xfail`` (numeric sums, each row's own amount, non-Transfer events excluded,
the day's total) pin the bugs fixed by the transfer/aggregate/rank rewrite.
"""

from collections.abc import Callable
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import pytest

from ethereum_transfers.metrics import Metric
from ethereum_transfers.pipeline import aggregate_inputs, generate_output

if TYPE_CHECKING:
    from conftest import LogFactory, MakeLogs, MakeTokens, WriteInputs

type RunMetrics = Callable[[pd.DataFrame], dict[Metric, pd.DataFrame]]

# the addresses listed by the `tokens` fixture (see conftest.py)
A, B, C, UNNAMED, NO_SYMBOL, UNLISTED = ("0x" + c * 40 for c in "abcdef")
DAY_1, DAY_2, DAY_3 = date(2023, 4, 1), date(2023, 4, 2), date(2023, 4, 3)
HEADER = "day,token_address,{metric}"


def read_results(written: list[Path]) -> dict[Metric, pd.DataFrame]:
    """Read the metric CSVs back as strings, keyed by metric."""
    return {
        Metric(path.stem): pd.read_csv(path, dtype=str, keep_default_na=False)
        for path in written
    }


@pytest.fixture
def run_metrics(
    tmp_path: Path, tokens: pd.DataFrame, write_inputs: "WriteInputs"
) -> RunMetrics:
    """Return ``run(logs)``: write the inputs, run every metric, read the CSVs."""

    def run(logs: pd.DataFrame) -> dict[Metric, pd.DataFrame]:
        logs_path, tokens_path = write_inputs(logs, tokens)
        written = generate_output(
            logs_path=logs_path,
            tokens_path=tokens_path,
            output_dir=tmp_path / "out",
            metrics=list(Metric),
        )
        return read_results(written)

    return run


def winners(result: pd.DataFrame, metric: Metric) -> dict[str, tuple[str, Decimal]]:
    """Map each ISO day in a metric CSV to its (token address, metric value)."""
    columns = result[["day", "token_address", metric]]
    return {
        day: (address, Decimal(value))
        for day, address, value in columns.itertuples(index=False, name=None)
    }


def test_count_of_transfers_end_to_end(
    make_logs: "MakeLogs", run_metrics: RunMetrics
) -> None:
    logs = make_logs(
        (A, DAY_1, 1),
        (A, DAY_1, 1),
        (B, DAY_1, 1),
        (B, DAY_2, 1),
        (C, DAY_3, 1),
        (C, DAY_3, 1),
        (C, DAY_3, 1),
    )

    result = run_metrics(logs)[Metric.COUNT_OF_TRANSFERS]

    assert result.to_dict("list") == {
        "day": ["2023-04-03", "2023-04-01", "2023-04-02"],
        "token_address": [C, A, B],
        "count_of_transfers": ["3", "2", "1"],
    }


@pytest.mark.parametrize(
    ("amounts", "total"), [((2, 10), 12), ((5, 5), 10)], ids=["2+10", "5+5"]
)
def test_quantity_sums_amounts_numerically(
    make_logs: "MakeLogs",
    run_metrics: RunMetrics,
    amounts: tuple[int, int],
    total: int,
) -> None:
    logs = make_logs(*[(A, DAY_1, amount) for amount in amounts])

    result = run_metrics(logs)[Metric.QUANTITY_TRANSFERRED]

    assert winners(result, Metric.QUANTITY_TRANSFERRED) == {
        "2023-04-01": (A, Decimal(total))
    }


def test_quantity_uses_each_rows_own_amount(
    make_logs: "MakeLogs", run_metrics: RunMetrics
) -> None:
    logs = make_logs((A, DAY_1, 100), (B, DAY_2, 7), (C, DAY_3, 3))

    result = run_metrics(logs)[Metric.QUANTITY_TRANSFERRED]

    assert winners(result, Metric.QUANTITY_TRANSFERRED) == {
        "2023-04-01": (A, Decimal(100)),
        "2023-04-02": (B, Decimal(7)),
        "2023-04-03": (C, Decimal(3)),
    }


@pytest.mark.parametrize(
    ("metric", "value"),
    [(Metric.COUNT_OF_TRANSFERS, 2), (Metric.QUANTITY_TRANSFERRED, 10)],
)
def test_non_transfer_events_are_excluded(
    make_logs: "MakeLogs", run_metrics: RunMetrics, metric: Metric, value: int
) -> None:
    # A has one Transfer and two (much larger) Approvals; B has two Transfers
    logs = make_logs(
        (A, DAY_1, 1),
        (A, DAY_1, 1000, "Approval"),
        (A, DAY_1, 1000, "Approval"),
        (B, DAY_1, 5),
        (B, DAY_1, 5),
    )

    result = run_metrics(logs)[metric]

    assert winners(result, metric) == {"2023-04-01": (B, Decimal(value))}


def test_quantity_reports_one_row_per_day_with_the_days_total(
    make_logs: "MakeLogs", run_metrics: RunMetrics
) -> None:
    # the winner has the largest total, not the largest single transfer
    logs = make_logs(
        (A, DAY_1, 3),
        (A, DAY_1, 4),
        (B, DAY_1, 5),
        (B, DAY_2, 1),
        (B, DAY_2, 1),
        (B, DAY_2, 1),
        (A, DAY_2, 2),
    )

    result = run_metrics(logs)[Metric.QUANTITY_TRANSFERRED]

    assert len(result) == 2
    assert winners(result, Metric.QUANTITY_TRANSFERRED) == {
        "2023-04-01": (A, Decimal(7)),
        "2023-04-02": (B, Decimal(3)),
    }


def test_amounts_beyond_uint256_are_written_exactly(
    make_logs: "MakeLogs", run_metrics: RunMetrics
) -> None:
    big = 2**256 - 1
    logs = make_logs((A, DAY_1, big), (A, DAY_1, big), (B, DAY_1, 1))

    result = run_metrics(logs)[Metric.QUANTITY_TRANSFERRED]

    assert result["quantity_transferred"].tolist() == [str(2 * big)]


def test_tokens_without_name_or_symbol_are_counted(
    make_logs: "MakeLogs", run_metrics: RunMetrics
) -> None:
    # tokens.csv lists them; missing metadata (e.g. MKR's bytes32 name) is fine
    logs = make_logs(
        (UNNAMED, DAY_1, 1),
        (UNNAMED, DAY_1, 1),
        (A, DAY_1, 1),
        (NO_SYMBOL, DAY_2, 1),
    )

    result = run_metrics(logs)[Metric.COUNT_OF_TRANSFERS]

    assert winners(result, Metric.COUNT_OF_TRANSFERS) == {
        "2023-04-01": (UNNAMED, Decimal(2)),
        "2023-04-02": (NO_SYMBOL, Decimal(1)),
    }


def test_unlisted_tokens_are_ignored(
    make_logs: "MakeLogs", run_metrics: RunMetrics
) -> None:
    logs = make_logs((UNLISTED, DAY_1, 9), (UNLISTED, DAY_1, 9), (A, DAY_1, 1))

    results = run_metrics(logs)

    for metric in Metric:
        assert winners(results[metric], metric) == {"2023-04-01": (A, Decimal(1))}


def test_ties_go_to_the_lower_token_address(
    make_logs: "MakeLogs", run_metrics: RunMetrics
) -> None:
    logs = make_logs((C, DAY_1, 4), (B, DAY_1, 4), (C, DAY_2, 1), (A, DAY_2, 1))

    results = run_metrics(logs)

    for metric in Metric:
        assert [
            address for address, _ in winners(results[metric], metric).values()
        ] == [
            B,
            A,
        ]


def test_erc721_and_non_standard_transfers_are_excluded(
    factory: "LogFactory",
    make_tokens: "MakeTokens",
    tmp_path: Path,
) -> None:
    # A only has ERC-721 / Approval / Deposit / ENS-style / non-indexed logs
    logs_dir = factory.write_dataset(
        tmp_path / "logs",
        [
            factory.erc721(A, DAY_1, token_id=10**30),
            factory.approval(A, DAY_1, 10**30),
            factory.weth_deposit(A, DAY_1, 10**30),
            factory.ens_transfer(A, DAY_1),
            factory.non_indexed_transfer(A, DAY_1, 10**30),
            factory.erc20(B, DAY_1, 1),
        ],
    )
    tokens_path = tmp_path / "tokens.csv"
    make_tokens((A, "AAA", "Token A"), (B, "BBB", "Token B")).to_csv(
        tokens_path, index=False
    )

    results = read_results(
        generate_output(
            logs_path=logs_dir,
            tokens_path=tokens_path,
            output_dir=tmp_path / "out",
            metrics=list(Metric),
        )
    )

    for metric in Metric:
        assert winners(results[metric], metric) == {"2023-04-01": (B, Decimal(1))}


@pytest.mark.parametrize(
    "listed",
    [
        pytest.param([A, A, A], id="duplicate-rows"),
        pytest.param([A.upper().replace("0X", "0x")], id="upper-case"),
    ],
)
def test_token_list_rows_are_normalized(
    make_logs: "MakeLogs",
    make_tokens: "MakeTokens",
    write_inputs: "WriteInputs",
    listed: list[str],
) -> None:
    logs_path, tokens_path = write_inputs(
        make_logs((A, DAY_1, 1), (A, DAY_1, 2)),
        make_tokens(*[(address, "AAA", "Token A") for address in listed]),
    )

    aggregate = aggregate_inputs(logs_path, tokens_path)

    assert aggregate[["token_address", "transfer_count"]].to_numpy().tolist() == [
        [A, 2]
    ]


@pytest.mark.parametrize(
    ("per_day", "limit", "expected"),
    [
        (1, None, [(C, "2023-04-02", "3"), (A, "2023-04-01", "2")]),
        (
            None,
            3,
            [(C, "2023-04-02", "3"), (A, "2023-04-01", "2"), (B, "2023-04-01", "1")],
        ),
        (2, 1, [(C, "2023-04-02", "3")]),
    ],
    ids=["per-day-1", "global-top-3", "limit-1"],
)
def test_per_day_and_limit(  # noqa: PLR0913, PLR0917 - fixtures plus parameters
    make_logs: "MakeLogs",
    tokens: pd.DataFrame,
    write_inputs: "WriteInputs",
    tmp_path: Path,
    per_day: int | None,
    limit: int | None,
    expected: list[tuple[str, str, str]],
) -> None:
    logs_path, tokens_path = write_inputs(
        make_logs((A, DAY_1, 1), (A, DAY_1, 1), (B, DAY_1, 1), *[(C, DAY_2, 1)] * 3),
        tokens,
    )

    [path] = generate_output(
        logs_path=logs_path,
        tokens_path=tokens_path,
        output_dir=tmp_path / "out",
        metrics=[Metric.COUNT_OF_TRANSFERS],
        per_day=per_day,
        limit=limit,
    )

    result = pd.read_csv(path, dtype=str)
    assert [
        (row.token_address, row.day, row.count_of_transfers)
        for row in result.itertuples()
    ] == expected


def test_no_transfers_writes_header_only_csvs(
    make_logs: "MakeLogs", run_metrics: RunMetrics
) -> None:
    results = run_metrics(make_logs((UNLISTED, DAY_1, 1), (A, DAY_1, 1, "Approval")))

    for metric, result in results.items():
        assert list(result.columns) == ["day", "token_address", metric]
        assert result.empty


class TestGenerateOutput:
    def test_writes_one_csv_per_metric_into_a_new_directory(
        self, inputs: tuple[Path, Path], tmp_path: Path
    ) -> None:
        logs_path, tokens_path = inputs
        output_dir = tmp_path / "nested" / "out"

        written = generate_output(
            logs_path=logs_path,
            tokens_path=tokens_path,
            output_dir=output_dir,
            metrics=list(Metric),
        )

        assert written == [
            output_dir / "count_of_transfers.csv",
            output_dir / "quantity_transferred.csv",
        ]
        assert sorted(output_dir.iterdir()) == written

    @pytest.mark.parametrize("metric", list(Metric))
    def test_csv_has_a_header_and_one_row_per_day(
        self,
        inputs: tuple[Path, Path],
        logs: pd.DataFrame,
        tmp_path: Path,
        metric: Metric,
    ) -> None:
        logs_path, tokens_path = inputs

        [path] = generate_output(
            logs_path=logs_path,
            tokens_path=tokens_path,
            output_dir=tmp_path,
            metrics=[metric],
        )

        header, *body = path.read_text().splitlines()
        assert header == HEADER.format(metric=metric)
        # every day in the `logs` fixture has a transfer from a listed token
        assert len(body) == logs["BLOCK_DATE"].nunique()

    @pytest.mark.parametrize(
        "metrics",
        [
            [Metric.COUNT_OF_TRANSFERS],
            [Metric.QUANTITY_TRANSFERRED],
            [Metric.QUANTITY_TRANSFERRED, Metric.COUNT_OF_TRANSFERS],
            [],
        ],
        ids=["count", "quantity", "quantity+count", "none"],
    )
    def test_writes_only_the_requested_metrics_in_order(
        self, inputs: tuple[Path, Path], tmp_path: Path, metrics: list[Metric]
    ) -> None:
        logs_path, tokens_path = inputs
        output_dir = tmp_path / "out"

        written = generate_output(
            logs_path=logs_path,
            tokens_path=tokens_path,
            output_dir=output_dir,
            metrics=metrics,
        )

        assert [path.name for path in written] == [f"{m}.csv" for m in metrics]
        assert sorted(output_dir.iterdir()) == sorted(written)

    def test_csv_format(
        self,
        make_logs: "MakeLogs",
        tokens: pd.DataFrame,
        write_inputs: "WriteInputs",
        tmp_path: Path,
    ) -> None:
        logs_path, tokens_path = write_inputs(
            make_logs((A, DAY_1, 7), (B, DAY_2, 1), (B, DAY_2, 2)), tokens
        )

        count_path, quantity_path = generate_output(
            logs_path=logs_path,
            tokens_path=tokens_path,
            output_dir=tmp_path / "out",
            metrics=list(Metric),
        )

        assert count_path.read_text() == (
            "day,token_address,count_of_transfers\n"
            f"2023-04-02,{B},2\n"
            f"2023-04-01,{A},1\n"
        )
        assert quantity_path.read_text() == (
            "day,token_address,quantity_transferred\n"
            f"2023-04-01,{A},7\n"
            f"2023-04-02,{B},3\n"
        )

    def test_logs_the_funnel(
        self,
        inputs: tuple[Path, Path],
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        logs_path, tokens_path = inputs
        caplog.set_level("INFO", logger="ethereum_transfers")

        generate_output(
            logs_path=logs_path,
            tokens_path=tokens_path,
            output_dir=tmp_path,
            metrics=[Metric.COUNT_OF_TRANSFERS],
        )

        # 5 listed tokens; 8 transfers from them (the Approval and the unlisted
        # token's transfer are dropped) over 3 days, in 6 (day, token) groups
        assert "5 distinct token addresses" in caplog.text
        assert "8 ERC-20 transfers of 5 tokens over 3 days (6 day/token groups)" in (
            caplog.text
        )
        assert "wrote 3 rows to" in caplog.text
