from collections.abc import Callable
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from ethereum_transfers.files import generate_output, write_new_output
from ethereum_transfers.metrics import Metric

type MakeLogs = Callable[..., pd.DataFrame]
type WriteInputs = Callable[[pd.DataFrame, pd.DataFrame], tuple[Path, Path]]

# the addresses listed by the `tokens` fixture (see conftest.py)
A, B = ("0x" + c * 40 for c in "ab")
DAY_1, DAY_2 = date(2023, 4, 1), date(2023, 4, 2)


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
        assert header == f"ADDRESS,BLOCK_DATE,{metric}"
        # every day in the `logs` fixture has a log from a named token
        assert len(body) == logs["BLOCK_DATE"].nunique()

    @pytest.mark.parametrize(
        "metrics",
        [
            [Metric.COUNT_OF_TRANSFERS],
            [Metric.QUANTITY_TRANSFERRED],
            [Metric.QUANTITY_TRANSFERRED, Metric.COUNT_OF_TRANSFERS],
        ],
        ids=["count", "quantity", "quantity+count"],
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

    def test_count_csv_format(
        self,
        make_logs: MakeLogs,
        tokens: pd.DataFrame,
        write_inputs: WriteInputs,
        tmp_path: Path,
    ) -> None:
        logs_path, tokens_path = write_inputs(
            make_logs((A, DAY_1, 1), (B, DAY_2, 1), (B, DAY_2, 1)), tokens
        )

        [path] = generate_output(
            logs_path=logs_path,
            tokens_path=tokens_path,
            output_dir=tmp_path / "out",
            metrics=[Metric.COUNT_OF_TRANSFERS],
        )

        assert path.read_text() == (
            "ADDRESS,BLOCK_DATE,count_of_transfers\n"
            f"{B},2023-04-02,2\n"
            f"{A},2023-04-01,1\n"
        )


class TestWriteNewOutput:
    @pytest.mark.parametrize("n_rows", [1, 4, 1000])
    def test_keeps_the_first_rows(
        self, inputs: tuple[Path, Path], tmp_path: Path, n_rows: int
    ) -> None:
        logs_path, _ = inputs
        output = tmp_path / "sample.gzip"

        write_new_output(logs_path, output, n_rows=n_rows)

        expected = pd.read_parquet(logs_path).head(n_rows)
        pd.testing.assert_frame_equal(pd.read_parquet(output), expected)

    def test_reads_a_directory_of_parquet_parts(
        self, logs: pd.DataFrame, tmp_path: Path
    ) -> None:
        source = tmp_path / "dataset"
        source.mkdir()
        logs.iloc[:6].to_parquet(source / "part-00000.parquet", index=False)
        logs.iloc[6:].to_parquet(source / "part-00001.parquet", index=False)
        output = tmp_path / "sample.gzip"

        write_new_output(source, output)

        pd.testing.assert_frame_equal(pd.read_parquet(output), logs)
