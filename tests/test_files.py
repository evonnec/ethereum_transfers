from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ethereum_transfers.files import (
    read_token_addresses,
    scan_transfers,
    write_head_sample,
    write_metric_csv,
)

if TYPE_CHECKING:
    from conftest import LogFactory, MakeTokens

A, B, C = ("0x" + c * 40 for c in "abc")


def write_csv(path: Path, text: str) -> Path:
    path.write_text(text)
    return path


def amounts(table: pa.Table) -> list[int]:
    """Decode the DATA column of `table` with plain Python ints."""
    return [int(str(data), 16) for data in table["DATA"].to_pylist()]


# --- read_token_addresses (review section 7.4, K1-K6) --------------------------


class TestReadTokenAddresses:
    def test_deduplicates_and_lower_cases(self, tmp_path: Path) -> None:  # K1, K2
        path = write_csv(
            tmp_path / "tokens.csv",
            f"address,symbol\n{A},AAA\n{A},AAA\n{B.upper().replace('0X', '0x')},BBB\n"
            f"{A},AAA\n",
        )

        assert read_token_addresses(path).to_pylist() == [A, B]

    def test_keeps_tokens_without_metadata(
        self, make_tokens: "MakeTokens", tmp_path: Path
    ) -> None:  # K3
        path = tmp_path / "tokens.csv"
        make_tokens((A, None, None), (B, "", ""), (C, "CCC", "Token C")).to_csv(
            path, index=False
        )

        assert read_token_addresses(path).to_pylist() == [A, B, C]

    def test_ignores_na_like_names_and_symbols(self, tmp_path: Path) -> None:  # K4
        path = write_csv(
            tmp_path / "tokens.csv",
            f"address,symbol,name\n{A},NA,NULL\n{B},nan,None\n",
        )

        assert read_token_addresses(path).to_pylist() == [A, B]

    @pytest.mark.parametrize(
        "bad",
        ["0x123", "", "0x" + "g" * 40, "aa" * 21],
        ids=["short", "empty", "hex", "no-0x"],
    )
    def test_rejects_invalid_addresses(self, tmp_path: Path, bad: str) -> None:  # K5
        path = write_csv(tmp_path / "tokens.csv", f"address,symbol\n{A},A\n{bad},X\n")

        with pytest.raises(ValueError, match="record 2 has an invalid token address"):
            read_token_addresses(path)

    def test_parses_quoted_multi_line_fields(self, tmp_path: Path) -> None:  # K6
        path = write_csv(
            tmp_path / "tokens.csv",
            f'address,name\n{A},"two\nlines"\n{B},"say ""hi"""\n',
        )

        assert read_token_addresses(path).to_pylist() == [A, B]

    def test_requires_an_address_column(self, tmp_path: Path) -> None:
        path = write_csv(tmp_path / "tokens.csv", "token,symbol\nx,y\n")

        with pytest.raises(ValueError, match="no 'address' column"):
            read_token_addresses(path)


# --- scan_transfers -----------------------------------------------------------


class TestScanTransfers:
    def test_streams_only_listed_erc20_transfers_and_three_columns(
        self, factory: "LogFactory", tmp_path: Path
    ) -> None:
        day = factory.DAY_1
        path = factory.write_dataset(
            tmp_path / "logs",
            [
                factory.erc20(A, day, 1),
                factory.erc721(A, day, 2),
                factory.approval(A, day, 3),
                factory.erc20(B, day, 4),
                factory.erc20(factory.UNLISTED, day, 5),
                factory.erc20(A, day, 6),
            ],
            parts=3,
        )

        batches = list(scan_transfers(path, pa.array([A, B]), batch_rows=1))

        assert all(batch.num_rows <= 1 for batch in batches)
        table = pa.Table.from_batches(batches)
        assert table.column_names == ["BLOCK_DATE", "ADDRESS", "DATA"]
        assert sorted(amounts(table)) == [1, 4, 6]

    def test_reads_a_single_parquet_file(
        self, factory: "LogFactory", tmp_path: Path
    ) -> None:
        path = tmp_path / "logs.parquet"
        pq.write_table(factory.table([factory.erc20(A, factory.DAY_1, 9)]), path)

        table = pa.Table.from_batches(list(scan_transfers(path, pa.array([A]))))

        assert table["ADDRESS"].to_pylist() == [A]

    def test_no_matching_rows(self, factory: "LogFactory", tmp_path: Path) -> None:
        path = factory.write_dataset(
            tmp_path / "logs", [factory.erc20(A, factory.DAY_1, 1)]
        )

        batches = scan_transfers(path, pa.array([B]))

        assert sum(batch.num_rows for batch in batches) == 0


# --- write_metric_csv (W1-W3) ------------------------------------------------


class TestWriteMetricCsv:
    def test_writes_exact_integers_without_an_index(self, tmp_path: Path) -> None:
        result = pd.DataFrame(
            {
                "day": ["2023-04-01", "2023-04-02"],
                "token_address": [A, B],
                "quantity_transferred": pd.Series([2**256 - 1, 0], dtype=object),
            }
        )
        path = tmp_path / "nested" / "dir" / "quantity_transferred.csv"

        write_metric_csv(result, path)

        assert path.read_text() == (
            "day,token_address,quantity_transferred\n"
            f"2023-04-01,{A},{2**256 - 1}\n"
            f"2023-04-02,{B},0\n"
        )

    def test_empty_result_writes_the_header(self, tmp_path: Path) -> None:
        path = tmp_path / "count_of_transfers.csv"

        write_metric_csv(
            pd.DataFrame(columns=["day", "token_address", "count_of_transfers"]), path
        )

        assert path.read_text() == "day,token_address,count_of_transfers\n"


# --- write_head_sample -------------------------------------------------------


class TestWriteHeadSample:
    @pytest.mark.parametrize("n_rows", [0, 1, 4, 7, 1000])
    def test_keeps_the_first_rows_in_file_order(
        self, factory: "LogFactory", tmp_path: Path, n_rows: int
    ) -> None:
        logs = [factory.erc20(A, factory.DAY_1, amount) for amount in range(7)]
        source = factory.write_dataset(tmp_path / "logs", logs, parts=3)
        output = tmp_path / "sample.gzip"

        written = write_head_sample(source, output, n_rows=n_rows, batch_rows=2)

        sample = pq.read_table(output)
        assert written == sample.num_rows == min(n_rows, len(logs))
        assert amounts(sample) == list(range(written))
        assert sample.schema.equals(factory.table([]).schema)

    def test_matches_pandas_head_of_a_single_file(
        self, inputs: tuple[Path, Path], tmp_path: Path
    ) -> None:
        logs_path, _ = inputs
        output = tmp_path / "sample.gzip"

        write_head_sample(logs_path, output, n_rows=4)

        expected = pd.read_parquet(logs_path).head(4)
        pd.testing.assert_frame_equal(pd.read_parquet(output), expected)

    def test_writes_gzip_parquet(self, factory: "LogFactory", tmp_path: Path) -> None:
        source = factory.write_dataset(
            tmp_path / "logs", [factory.erc20(A, factory.DAY_1, 1)]
        )
        output = tmp_path / "sample.gzip"

        write_head_sample(source, output)

        metadata = pq.ParquetFile(output).metadata
        assert metadata.row_group(0).column(0).compression == "GZIP"
