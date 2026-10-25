"""Property tests: arbitrary transfers, batch splits and amounts match the oracle."""

from datetime import date, timedelta
from typing import TYPE_CHECKING

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from ethereum_transfers.aggregate import aggregate_transfers
from ethereum_transfers.metrics import Metric, compute_metric
from ethereum_transfers.transfers import erc20_transfer_filter

if TYPE_CHECKING:
    from conftest import LogFactory, Oracle

TOKENS = ["0x" + c * 40 for c in "abc"]
MAX_UINT256 = 2**256 - 1
# full-range values, plus the boundaries where limb carries and ties happen
amounts = st.one_of(
    st.integers(0, MAX_UINT256),
    st.sampled_from([0, 1, 2**32 - 1, 2**32, 2**64, 2**255, MAX_UINT256]),
    st.integers(0, 3),
)
events = st.lists(
    st.tuples(
        st.sampled_from(TOKENS), st.integers(0, 2), amounts, st.booleans()
    ),  # token, day offset, amount, is an ERC-721 transfer (must be ignored)
    max_size=40,
)


@settings(
    max_examples=150,
    derandomize=True,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    events=events,
    n_batches=st.integers(1, 4),
    per_day=st.sampled_from([1, 2, None]),
    limit=st.sampled_from([0, 1, 2, None]),
)
def test_results_match_the_oracle(  # noqa: PLR0913, PLR0917 - fixtures plus drawn values
    factory: "LogFactory",
    oracle: "Oracle",
    events: list[tuple[str, int, int, bool]],
    n_batches: int,
    per_day: int | None,
    limit: int | None,
) -> None:
    base = date(2023, 4, 1)
    erc20 = [
        factory.erc20(token, base + timedelta(days=day), amount)
        for token, day, amount, nft in events
        if not nft
    ]
    noise = [
        factory.erc721(token, base + timedelta(days=day), amount % 1000)
        for token, day, amount, nft in events
        if nft
    ]
    batches = [factory.table((erc20 + noise)[i::n_batches]) for i in range(n_batches)]
    selected = [batch.filter(erc20_transfer_filter()) for batch in batches]
    aggregate = aggregate_transfers(selected)

    expected = oracle.aggregate(erc20)
    for metric, position in (
        (Metric.COUNT_OF_TRANSFERS, 0),
        (Metric.QUANTITY_TRANSFERRED, 1),
    ):
        got = compute_metric(aggregate, metric, per_day=per_day, limit=limit)
        want = oracle.rank(
            {key: value[position] for key, value in expected.items()},
            per_day=per_day,
            limit=limit,
        )
        assert list(got.itertuples(index=False, name=None)) == want
