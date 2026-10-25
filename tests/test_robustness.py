"""User errors give a clean message and a non-zero exit; input case variants work."""

import os
import stat
import sys
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import pytest

from ethereum_transfers.cli import main

if TYPE_CHECKING:
    from conftest import MakeLogs, WriteInputs

A, B = "0x" + "a" * 40, "0x" + "b" * 40
DAY = date(2023, 4, 1)
TRANSFER_TOPIC0 = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"


def run(logs: Path, tokens: Path, out: Path, *extra: str) -> int:
    """Run the CLI's ``run`` command and return its exit status."""
    return main(
        ["run", "--logs", str(logs), "--tokens", str(tokens), "-o", str(out), *extra]
    )


def assert_clean_failure(
    code: int, capsys: pytest.CaptureFixture[str], *needles: str
) -> None:
    """The run failed with one readable line and no traceback."""
    err = capsys.readouterr().err
    assert code == 1
    assert "Traceback" not in err
    assert err.startswith("ethereum-transfers: error: ")
    assert all(needle in err for needle in needles), err


class TestUserErrorsAreReported:
    def test_missing_logs_path(
        self,
        inputs: tuple[Path, Path],
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        missing = tmp_path / "nope"
        assert_clean_failure(run(missing, inputs[1], tmp_path / "o"), capsys, "nope")

    def test_missing_tokens_path(
        self,
        inputs: tuple[Path, Path],
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        missing = tmp_path / "nope.csv"
        assert_clean_failure(
            run(inputs[0], missing, tmp_path / "o"), capsys, "nope.csv"
        )

    @pytest.mark.parametrize(
        "markers", [[], ["_SUCCESS"]], ids=["empty", "success-only"]
    )
    def test_logs_directory_without_parquet_files(
        self,
        inputs: tuple[Path, Path],
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        markers: list[str],
    ) -> None:
        logs = tmp_path / "logs_dir"
        logs.mkdir()
        for name in markers:
            (logs / name).write_text("")
        assert_clean_failure(
            run(logs, inputs[1], tmp_path / "o"), capsys, "no parquet files"
        )

    def test_logs_without_the_needed_columns(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        logs = tmp_path / "logs.parquet"
        pd.DataFrame({"ADDRESS": ["x"]}).to_parquet(logs)
        tokens = tmp_path / "t.csv"
        tokens.write_text(f"address\n{A}\n")
        assert_clean_failure(run(logs, tokens, tmp_path / "o"), capsys, "TOPIC0")

    def test_tokens_without_an_address_column(
        self,
        inputs: tuple[Path, Path],
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        tokens = tmp_path / "t.csv"
        tokens.write_text("symbol,name\nX,Y\n")
        assert_clean_failure(run(inputs[0], tokens, tmp_path / "o"), capsys, "address")

    def test_empty_tokens_file(
        self,
        inputs: tuple[Path, Path],
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        tokens = tmp_path / "t.csv"
        tokens.write_text("")
        assert_clean_failure(run(inputs[0], tokens, tmp_path / "o"), capsys, "t.csv")

    def test_malformed_data_value(
        self,
        make_logs: "MakeLogs",
        write_inputs: "WriteInputs",
        tokens: pd.DataFrame,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        logs = make_logs((A, DAY, 1), (A, DAY, 2))
        logs.loc[1, "DATA"] = "0x" + "zz" * 32
        logs_path, tokens_path = write_inputs(logs, tokens)
        assert_clean_failure(
            run(logs_path, tokens_path, tmp_path / "o"), capsys, "non-hex"
        )

    def test_output_dir_is_a_file(
        self,
        inputs: tuple[Path, Path],
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        blocker = tmp_path / "file"
        blocker.write_text("")
        assert_clean_failure(run(*inputs, blocker / "out"), capsys, "file")

    @pytest.mark.skipif(
        sys.platform == "win32" or os.geteuid() == 0, reason="needs POSIX non-root"
    )
    def test_unwritable_output_dir(
        self,
        inputs: tuple[Path, Path],
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        read_only = tmp_path / "ro"
        read_only.mkdir()
        read_only.chmod(stat.S_IRUSR | stat.S_IXUSR)
        try:
            assert_clean_failure(
                run(*inputs, read_only / "out"), capsys, "Permission denied"
            )
        finally:
            read_only.chmod(stat.S_IRWXU)


def test_header_only_token_list_gives_header_only_outputs(
    inputs: tuple[Path, Path], tmp_path: Path
) -> None:
    tokens = tmp_path / "t.csv"
    tokens.write_text("address,symbol\n")
    assert run(inputs[0], tokens, tmp_path / "o") == 0
    count = (tmp_path / "o" / "count_of_transfers.csv").read_text()
    assert count == "day,token_address,count_of_transfers\n"


def test_limit_zero_writes_header_only_files(
    inputs: tuple[Path, Path], tmp_path: Path
) -> None:
    assert run(*inputs, tmp_path / "o", "--limit", "0") == 0
    text = (tmp_path / "o" / "quantity_transferred.csv").read_text()
    assert text == "day,token_address,quantity_transferred\n"


class TestCaseVariantsInTheLogs:
    """Hex is case-insensitive; the same contract must not be missed or split."""

    def test_upper_case_log_address_is_counted_as_the_lower_case_token(
        self,
        make_logs: "MakeLogs",
        write_inputs: "WriteInputs",
        tokens: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        logs = make_logs((A, DAY, 1), (A, DAY, 2), (B, DAY, 3))
        logs.loc[1, "ADDRESS"] = "0x" + "A" * 40
        logs_path, tokens_path = write_inputs(logs, tokens)
        assert run(logs_path, tokens_path, tmp_path / "o") == 0
        result = pd.read_csv(tmp_path / "o" / "count_of_transfers.csv", dtype=str)
        assert result.to_dict("records") == [
            {"day": "2023-04-01", "token_address": A, "count_of_transfers": "2"}
        ]

    def test_upper_case_data_and_topic0_are_transfers(
        self,
        make_logs: "MakeLogs",
        write_inputs: "WriteInputs",
        tokens: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        logs = make_logs((A, DAY, 255), (A, DAY, 255))
        logs.loc[0, "DATA"] = "0x" + format(255, "064X")
        logs.loc[1, "TOPIC0"] = "0x" + TRANSFER_TOPIC0[2:].upper()
        logs_path, tokens_path = write_inputs(logs, tokens)
        assert run(logs_path, tokens_path, tmp_path / "o") == 0
        result = pd.read_csv(tmp_path / "o" / "quantity_transferred.csv", dtype=str)
        assert result["quantity_transferred"].tolist() == ["510"]
