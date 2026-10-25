# Ethereum Transfers
Find the most transferred ERC-20 token per day in decoded Ethereum logs, measured two
ways: (1) the count of transfers and (2) the sum of the transferred amounts.

## set up the environment
Install [uv](https://docs.astral.sh/uv/getting-started/installation/) (e.g. `brew install uv`), then:

```sh
uv sync
```

This creates `.venv/` from `uv.lock` with the package installed (editable) and the dev
tools. If the Python pinned in `.python-version` (3.15) isn't installed, uv downloads it.
Python 3.13+ is supported.

## input data
The inputs are not in git (they are large; see `.gitignore`). Put them in the repository
root, or pass their paths on the command line:

| path | contents |
|---|---|
| `eth_logs_decoded_2023_sample/` | decoded logs: a Spark output directory of snappy parquet parts (18,200,863 rows, 2023-04-23 to 2023-04-30) |
| `tokens.csv` | the ERC-20 token list (`address,symbol,name,decimals,...`) |

## run
The package installs an `ethereum-transfers` command (also runnable as
`python -m ethereum_transfers`):

```sh
uv run ethereum-transfers --help
uv run ethereum-transfers        # same as `run` with the default paths
uv run ethereum-transfers run --logs eth_logs_decoded_2023_sample \
    --tokens tokens.csv --output-dir output
uv run ethereum-transfers run --metric count_of_transfers
uv run ethereum-transfers run --per-day all --limit 100   # top (day, token) pairs overall
uv run python -m ethereum_transfers --help
```

`run` writes `count_of_transfers.csv` and `quantity_transferred.csv` into the output
directory (default: `output/`). `--logs` takes a parquet file or a directory of parquet
parts (Spark marker files such as `_SUCCESS` are ignored). The full dataset takes about
2 seconds and 1.2 GB of memory, because the input is streamed (see
[performance](#performance)).
A missing or empty input, a token list without an `address` column or a malformed `DATA`
value prints `ethereum-transfers: error: ...` to stderr and exits with status 1 (invalid
arguments exit with 2). Hex in `ADDRESS` and `TOPIC0` is matched case-insensitively.

To make a smaller input for experiments, take the first rows of the dataset (also
streamed):

```sh
uv run ethereum-transfers sample --input eth_logs_decoded_2023_sample \
    --output eth_logs_decoded_2023_sample_100.gzip --rows 1000000
```

The first rows in file order are not a random sample: the parquet parts are split by
date, so the first million rows cover only 3 of the 8 days, and only part of each.

## interpretation and assumptions
The problem statement (below) leaves some definitions open. This is how they are settled:

| question | answer |
|---|---|
| What is a transfer? | An ERC-20 `Transfer(address indexed from, address indexed to, uint256 value)` log: `TOPIC0` is `0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef` (keccak256 of the signature), `TOPIC3` is null and `DATA` is a single 32-byte word. ERC-721 transfers share that `TOPIC0` but carry the token id in `TOPIC3` with empty `DATA`, so they are excluded. Other events from token contracts (Approval, Deposit, Withdrawal, ...) are not transfers. |
| What is an ERC-20 token? | A contract whose `ADDRESS` is listed in `tokens.csv`: addresses are lower-cased and de-duplicated, and `name`/`symbol`/`decimals` are not required (MKR, for example, has `bytes32` metadata that the list leaves empty). |
| Where does the amount come from? | `DATA`, decoded as an unsigned 256-bit integer, as in [Reading ERC-20 token event logs](https://goethereumbook.org/en/event-read-erc20/). The decoded `NAME`/`PARAMS` columns are not used: `NAME` is shared by unrelated events, and the key names inside `PARAMS` vary even within one token contract. |
| "Sum of transfer amounts": which units? | Raw base units, summed exactly (Python integers; a daily sum here reaches 257 bits). No `decimals` scaling: without prices, amounts of different tokens are not comparable in any unit, and `decimals` is missing or implausible for many tokens. |
| "Most transferred token per day" + "limit 100" | The top token of each day, ordered by the metric descending, at most 100 rows. With 8 days of data that gives 8 rows; the limit matters for more than 100 days. `--per-day N\|all` and `--limit N\|all` give other readings, e.g. `--per-day all` ranks all (day, token) pairs together. |
| Day | `BLOCK_DATE`, which is the UTC date of `BLOCK_TIMESTAMP`. |
| Ties | Within a day, the lower token address wins; rows with equal metrics are ordered by day, then address. |

Self-transfers (`from == to`) and zero-value transfers are transfers and are counted.

## output
| file | columns | types |
|---|---|---|
| `count_of_transfers.csv` | `day,token_address,count_of_transfers` | ISO date, lower-case `0x` + 40 hex address, integer |
| `quantity_transferred.csv` | `day,token_address,quantity_transferred` | ISO date, lower-case `0x` + 40 hex address, non-negative integer of any size (raw base units) |

Both files have a header and no index column. Amounts can exceed 64 bits, so read the
`quantity_transferred` column as text or `decimal.Decimal` (e.g.
`pd.read_csv(path, dtype={"quantity_transferred": str})`), not as a float.

## results
On the full dataset (`eth_logs_decoded_2023_sample`), 3,359,694 transfers of 1,306
listed tokens qualify. The outputs (also in `tests/golden/`):

`count_of_transfers.csv`: WETH every day.

| day | token_address | count_of_transfers |
|---|---|---|
| 2023-04-30 | 0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2 (WETH) | 358263 |
| 2023-04-23 | 0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2 (WETH) | 301054 |
| 2023-04-24 | 0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2 (WETH) | 295272 |
| 2023-04-28 | 0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2 (WETH) | 275657 |
| 2023-04-25 | 0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2 (WETH) | 267800 |
| 2023-04-29 | 0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2 (WETH) | 266547 |
| 2023-04-26 | 0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2 (WETH) | 264764 |
| 2023-04-27 | 0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2 (WETH) | 237736 |

`quantity_transferred.csv` (raw base units; every winner has 18 decimals):

| day | token_address | quantity_transferred |
|---|---|---|
| 2023-04-25 | 0x0f5d2fb29fb7d3cfee444a200298f468908cc942 (MANA) | 150127166855054711754185208022270434443779478604121699783651713431226147362738 |
| 2023-04-26 | 0x6fbd60b001f4c393edb471713cb7e33145475601 | 5209524875670748271971032714974944 |
| 2023-04-24 | 0xcc19e8e4d099130e2dd89efb50c882cbd42bece1 | 4744821631634086110399878710965797 |
| 2023-04-28 | 0x27fbdd15fb0820b50b9c3683ea628bb6ab9b2d70 | 4290562883818456320064949980209312 |
| 2023-04-27 | 0x469ec2389e6e1538d54d795cd9da8e2cbda26285 | 2438008066583788618229282414339631 |
| 2023-04-30 | 0x3ce432ae1e60d28ca3e322bca8b77204c1488ab1 | 766983666305301877095043178711588 |
| 2023-04-23 | 0xa21af1050f7b26e0cff45ee51548254c41ed6b5c | 531889843546170167807255205951591 |
| 2023-04-29 | 0x2f9678085acf1897f63a102e9f7cb835609158d1 | 178674942841140016817766738151629 |

### caveats
- The answer is limited to `tokens.csv`: USDC and DAI, for example, are not listed.
- Raw-unit sums favour tokens with huge supplies and 18 decimals. WETH, the count winner,
  ranks 139th to 167th by raw amount.
- On 2023-04-25, MANA wins because of two self-transfers of about 1.2e77 and 3.5e76 base
  units. Self-transfers are 11% of all qualifying transfers; excluding them (not done
  here) would make `0xfd957f21bd95e723645c07c48a2d8acb8ffb3794` that day's winner.
- `tokens.csv` also lists NFT collections; their ERC-721 transfers are excluded by the
  log shape, not by the list.

Possible follow-ups, not implemented: `--normalize` (divide by `decimals`, written as
exact decimal strings), `--exclude-self-transfers`, and `--start`/`--end` date filters.

## performance
`run` reads only the `BLOCK_DATE`, `ADDRESS`, `DATA`, `TOPIC0` and `TOPIC3` columns, pushes
the transfer filter down into the parquet scan, and reads one file at a time. Each batch
is reduced to per-(day, token) counts and 32-bit-limb sums of the amounts, which stay
exact in `uint64`, so memory does not grow with the input. On the full dataset (Apple
silicon, 8 cores) a run takes about 2 seconds with a peak of about 1.2 GB. `sample
--rows 1000000` takes about 16 seconds (gzip) with a peak of about 1.6 GB.

## project layout
```text
src/ethereum_transfers/
    __init__.py    package docstring, __version__, public API re-exports
    __main__.py    entry point for `python -m ethereum_transfers`
    cli.py         argparse CLI: `run` (default) and `sample` subcommands
    files.py       reads the token list, streams the logs, writes CSVs and samples
    pipeline.py    one run: aggregate the inputs once, rank and write each metric
    transfers.py   ERC-20 Transfer filter; exact uint256 decoding of DATA
    aggregate.py   streamed per-(day, token) counts and exact amount sums
    rank.py        deterministic top-N-per-day ranking
    metrics.py     the Metric enum and how each metric is ranked
tests/             pytest suite on small synthetic data (fixtures in conftest.py)
tests/golden/      expected outputs for the full dataset (integration tests)
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

The `integration` tests run the CLI on the real data and are skipped unless you point
them at it. With the full dataset they also compare the outputs with `tests/golden/` and
check the totals. Running only them (`-m integration`) covers a fraction of the code, so
turn the coverage gate off for that run:

```sh
ETHEREUM_TRANSFERS_LOGS=eth_logs_decoded_2023_sample \
ETHEREUM_TRANSFERS_TOKENS=tokens.csv uv run pytest -m integration --no-cov
```

With the two variables set, plain `uv run pytest` runs the integration tests together
with everything else.

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
