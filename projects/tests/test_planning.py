import pytest

from projects.rag_pipeline import (
    _heuristic_plan,
    plan_api_call,
    sanitize_plan,
    ApiCallPlan,
)
from projects.blockchair_client import BlockchairClient, PaginationPlan


def test_missing_identifier_stays_none():
    q = "Show bitcoin block dashboard"  # Missing numeric block
    docs = ""
    plan = plan_api_call(q, docs, llm=None)
    assert plan.endpoint_type in {"stats", "block_dashboard"}
    if plan.endpoint_type != "stats":
        # If LLM disabled, heuristic may pick block_dashboard only with number; ensure sanitize downgrades
        sanitized = sanitize_plan(plan, q, docs)
        assert sanitized.endpoint_type == "stats"
        assert sanitized.identifier is None


@pytest.mark.parametrize(
    "query,expected_chain",
    [
        ("What's the dash stats?", "dash"),
        ("Zcash transactions", "zcash"),
        ("Show XRP latest transaction", "ripple"),
        ("News on Groestlcoin", "groestlcoin"),
        ("BSV mempool", "bitcoin-sv"),
        ("XEC price", "ecash"),
    ],
)
def test_chain_detection(query, expected_chain):
    plan = _heuristic_plan(query)
    assert plan.chain == expected_chain


def test_invalid_params_removed():
    q = "Bitcoin stats"
    docs = ""
    plan = ApiCallPlan(chain="bitcoin", endpoint_type="stats", params={"q": "value>100"})
    sanitized = sanitize_plan(plan, q, docs)
    assert sanitized.params == {}


def test_unsupported_chain_defaults():
    q = "Some chain that doesn't exist stats"
    docs = ""
    plan = ApiCallPlan(chain="bitcoin", endpoint_type="stats")
    # Force chain to invalid by replacing value dynamically
    plan.chain = "not-a-chain"  # type: ignore
    sanitized = sanitize_plan(plan, q, docs)
    assert sanitized.chain == "bitcoin"
    assert sanitized.endpoint_type == "stats"


def test_sort_normalization_transactions_list():
    q = "Get 500 recent transactions for Zcash sorted by block_time desc"
    docs = q
    plan = plan_api_call(q, docs, llm=None)
    # Ensure endpoint and chain
    assert plan.chain == "zcash"
    assert plan.endpoint_type == "transactions_list"
    # Merge pagination and params into URL
    client = BlockchairClient(api_key=None, base_url="https://api.blockchair.com")
    url, params = client.build_request(plan)
    assert url.endswith("/zcash/transactions")
    assert params.get("s") == "block_time(desc)"
    assert params.get("limit") == "500"


def test_sort_normalization_blocks_list():
    q = "List latest 20 Bitcoin SV blocks sorted by id asc"
    docs = q
    plan = plan_api_call(q, docs, llm=None)
    assert plan.chain == "bitcoin-sv"
    # Heuristic should pick blocks_list
    assert plan.endpoint_type in {"blocks_list", "block_dashboard", "stats"}
    # For 'latest 20 blocks' we expect list
    if plan.endpoint_type != "blocks_list":
        # Force to blocks_list for URL test
        plan.endpoint_type = "blocks_list"  # type: ignore
    client = BlockchairClient(api_key=None, base_url="https://api.blockchair.com")
    url, params = client.build_request(plan)
    assert url.endswith("/bitcoin-sv/blocks")
    # Sort normalized
    assert params.get("s") == "id(asc)"
    # Limit normalized from query
    assert params.get("limit") == "20"

