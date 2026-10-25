import runpy
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

import pandas as pd
import pytest

from ethereum_transfers import __version__
from ethereum_transfers.cli import (
    DEFAULT_LOGS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TOKENS,
    main,
)

CSV_NAMES = ["count_of_transfers.csv", "quantity_transferred.csv"]


def run_args(inputs: tuple[Path, Path], output_dir: Path) -> list[str]:
    """Return ``run`` arguments that read `inputs` and write to `output_dir`."""
    logs_path, tokens_path = inputs
    return [
        "run",
        "--logs",
        str(logs_path),
        "--tokens",
        str(tokens_path),
        "--output-dir",
        str(output_dir),
    ]


def csv_names(directory: Path) -> list[str]:
    """Return the names of the metric CSVs found in `directory`."""
    return [name for name in CSV_NAMES if (directory / name).is_file()]


def test_run_writes_a_csv_per_metric(inputs: tuple[Path, Path], tmp_path: Path) -> None:
    output_dir = tmp_path / "out"

    assert main(run_args(inputs, output_dir)) == 0

    assert csv_names(output_dir) == CSV_NAMES


@pytest.mark.parametrize(
    ("metric_args", "expected"),
    [
        (["-m", "count_of_transfers"], ["count_of_transfers.csv"]),
        (["--metric", "quantity_transferred"], ["quantity_transferred.csv"]),
        (["-m", "quantity_transferred", "count_of_transfers"], CSV_NAMES),
    ],
    ids=["count", "quantity", "both"],
)
def test_run_metric_option_selects_the_metrics(
    inputs: tuple[Path, Path],
    tmp_path: Path,
    metric_args: list[str],
    expected: list[str],
) -> None:
    output_dir = tmp_path / "out"

    assert main([*run_args(inputs, output_dir), *metric_args]) == 0

    assert csv_names(output_dir) == expected


def test_no_command_runs_with_the_default_paths(
    inputs: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # the default --logs is the full dataset directory of parquet parts
    logs_path, tokens_path = inputs
    workdir = tmp_path / "workdir"
    (workdir / DEFAULT_LOGS).mkdir(parents=True)
    shutil.copy(logs_path, workdir / DEFAULT_LOGS / "part-00000.snappy.parquet")
    (workdir / DEFAULT_LOGS / "_SUCCESS").touch()
    shutil.copy(tokens_path, workdir / DEFAULT_TOKENS)
    monkeypatch.chdir(workdir)

    assert main([]) == 0

    assert csv_names(workdir / DEFAULT_OUTPUT_DIR) == CSV_NAMES


@pytest.mark.parametrize(
    ("options", "expected_rows"),
    [
        ([], 3),
        (["--limit", "2"], 2),
        (["--per-day", "all"], 6),
        (["--per-day", "ALL", "--limit", "all"], 6),
        (["--per-day", "2", "--limit", "4"], 4),
        (["--limit", "0"], 0),
    ],
    ids=[
        "default",
        "limit-2",
        "per-day-all",
        "all-all",
        "per-day-2-limit-4",
        "limit-0",
    ],
)
def test_run_per_day_and_limit_options(
    inputs: tuple[Path, Path], tmp_path: Path, options: list[str], expected_rows: int
) -> None:
    # the `logs` fixture has transfers from listed tokens in 6 (day, token)
    # pairs over 3 days
    output_dir = tmp_path / "out"

    argv = [*run_args(inputs, output_dir), "-m", "count_of_transfers", *options]
    assert main(argv) == 0

    result = pd.read_csv(output_dir / "count_of_transfers.csv", dtype=str)
    assert list(result.columns) == ["day", "token_address", "count_of_transfers"]
    assert len(result) == expected_rows


@pytest.mark.parametrize(
    "argv", [["--help"], ["run", "--help"], ["sample", "--help"]], ids=" ".join
)
def test_help_exits_0(argv: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(argv)

    assert exit_info.value.code == 0
    assert capsys.readouterr().out.startswith("usage: ethereum-transfers")


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])

    assert exit_info.value.code == 0
    assert capsys.readouterr().out == f"ethereum-transfers {__version__}\n"


@pytest.mark.parametrize(
    "argv",
    [
        ["run", "--metric", "bogus"],
        ["bogus"],
        ["sample", "--rows", "many"],
        ["sample", "--rows", "-1"],
        ["run", "--per-day", "0"],
        ["run", "--per-day", "some"],
        ["run", "--limit", "-1"],
        ["run", "--limit", "1.5"],
    ],
    ids=" ".join,
)
def test_invalid_arguments_exit_2(
    argv: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(argv)

    assert exit_info.value.code == 2
    assert "error:" in capsys.readouterr().err


def test_sample_writes_the_first_rows(
    inputs: tuple[Path, Path], tmp_path: Path
) -> None:
    logs_path, _ = inputs
    output = tmp_path / "sample.gzip"

    argv = ["sample", "--input", str(logs_path), "--output", str(output), "-n", "3"]
    assert main(argv) == 0

    expected = pd.read_parquet(logs_path).head(3)
    pd.testing.assert_frame_equal(pd.read_parquet(output), expected)


def test_python_m_runs_main(
    inputs: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # in-process, so __main__.py counts towards coverage
    output_dir = tmp_path / "out"
    monkeypatch.setattr(
        sys, "argv", ["ethereum_transfers", *run_args(inputs, output_dir)]
    )

    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("ethereum_transfers", run_name="__main__")

    assert exit_info.value.code == 0
    assert csv_names(output_dir) == CSV_NAMES


def test_python_m_help_in_a_subprocess() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "ethereum_transfers", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.startswith("usage: ethereum-transfers")


@pytest.fixture(scope="module")
def console_script() -> str:
    """Path of the installed ``ethereum-transfers`` console script."""
    script = shutil.which("ethereum-transfers", path=sysconfig.get_path("scripts"))
    assert script is not None, "the project is not installed in this environment"
    return script


def test_console_script_runs(
    console_script: str, inputs: tuple[Path, Path], tmp_path: Path
) -> None:
    output_dir = tmp_path / "out"

    completed = subprocess.run(
        [console_script, *run_args(inputs, output_dir)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert csv_names(output_dir) == CSV_NAMES


def test_console_script_exits_2_on_an_invalid_metric(console_script: str) -> None:
    completed = subprocess.run(
        [console_script, "run", "--metric", "bogus"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "invalid choice: 'bogus'" in completed.stderr
