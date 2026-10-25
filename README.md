# Ethereum Transfers
Measure Ethereum `ERC-20`` transfer volume with two metrics: (1) count of transfers and (2) sum of transfer amounts  

## set up the environment
Install [uv](https://docs.astral.sh/uv/getting-started/installation/) (e.g. `brew install uv`), then:

```sh
uv sync
```

This creates `.venv/` from `uv.lock` with the package installed (editable) and the dev
tools. If the Python pinned in `.python-version` (3.15) isn't installed, uv downloads it.
Python 3.13+ is supported.

## run
The package installs an `ethereum-transfers` command (also runnable as
`python -m ethereum_transfers`):

```sh
uv run ethereum-transfers --help
uv run ethereum-transfers        # same as `run` with the default paths below
uv run ethereum-transfers run --logs eth_logs_decoded_2023_sample_100.gzip \
    --tokens tokens.csv --output-dir out
uv run ethereum-transfers run --metric count_of_transfers
uv run python -m ethereum_transfers --help
```

`run` writes `count_of_transfers.csv` and `quantity_transferred.csv` into the output
directory (default: the current directory). `--logs` takes a parquet file or a directory
of parquet parts.

To make the smaller input used by default, take the first rows of the full dataset
(this loads the whole dataset into memory):

```sh
uv run ethereum-transfers sample --input eth_logs_decoded_2023_sample \
    --output eth_logs_decoded_2023_sample_100.gzip --rows 1000000
```

## project layout
```text
src/ethereum_transfers/
    __init__.py    package docstring, __version__, public API re-exports
    __main__.py    entry point for `python -m ethereum_transfers`
    cli.py         argparse CLI: `run` (default) and `sample` subcommands
    files.py       reads the inputs, writes the metric CSVs, sampling helper
    metrics.py     pure DataFrame functions: join_tokens and the two metrics
tests/             pytest suite on small synthetic data (fixtures in conftest.py)
pyproject.toml     metadata, dependencies, and ruff / mypy / pytest / coverage config
uv.lock            locked dependency versions
```

## development checks
CI runs exactly these:

```sh
uv sync --locked            # install the locked environment
uv run ruff check           # lint
uv run ruff format --check  # formatting
uv run mypy                 # strict type check of src/ and tests/
uv run pytest               # tests, with branch coverage (fails under 95%)
uv build                    # sdist and wheel into dist/
```

Tests marked `xfail` describe known bugs in the current metrics. They are strict, so
each one starts failing once its bug is fixed, until the marker is removed. The
`integration` test runs the CLI on real data and is skipped unless you point it at
some (use a sample, not the full dataset):

```sh
ETHEREUM_TRANSFERS_LOGS=eth_logs_decoded_2023_sample_100.gzip \
ETHEREUM_TRANSFERS_TOKENS=tokens.csv uv run pytest -m integration --no-cov
```

## CI

GitHub Actions (`.github/workflows/ci.yml`) runs on every pull request, every push to `main`, every `v*` tag, and on demand. Lint (`ruff check`), format (`ruff format --check`), typecheck (`mypy`) and tests (`pytest` on Python 3.13, 3.14 and 3.15) run in parallel from the locked uv environment. Coverage appears in each test job's summary and as an artifact.

Once all checks pass, `uv build` creates the sdist and wheel. The wheel is smoke-tested in a clean environment (`ethereum-transfers --help`) and both files are uploaded as the `dist` artifact.

**Cutting a release:** set the version (`uv version <x.y.z>`), commit, then `git tag v<x.y.z> && git push origin v<x.y.z>`. CI checks that the tag matches the project version and creates a GitHub Release with the wheel and sdist attached.

**Publishing to a package index** is off by default. To turn it on, register a trusted publisher on the index (workflow `ci.yml`, environment `pypi`) and set the repository variable `PUBLISH_TO_PYPI=true`. Set `PYPI_PUBLISH_URL` to target a different index, e.g. TestPyPI's `https://test.pypi.org/legacy/`.

## problem statement
Use the Ethereum Logs data below to find the most transferred ERC-20 token per day.  
  
A list of ERC-20 tokens is also provided in `tokens.csv`.  
  
Specifically, measure transfer volume with two metrics:  
(1) count of transfers  
(2) sum of transfer amounts  
  
The resulting dataset should be two CSVs, one for each metric, each with 3 columns  
  
(day, token address, transfer metric), ordered by transfer metric descending, limit 100.  
  
Data Contents  
  
- Ethereum Decoded Logs (Parquet)
- ERC-20 Token List (csv)


### Examine the data

BLOCK_DATE is transaction date  
`sample_output1 = ["ADDRESS", "BLOCK_DATE", "count_of_transfers"]`  
`sample_output2 = ["ADDRESS", "BLOCK_DATE", "quantity_of_transfers"]`
  
If token addresss exists in tokens.csv, this is included in the metric, else next transaction
Fields `symbol` and `name` are applicable in determining if the token's address from the logs, exist





