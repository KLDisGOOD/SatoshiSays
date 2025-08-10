from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from rag_pipeline import plan_api_call
from blockchair_client import BlockchairClient


TEST_CASES_JSON = r"""
[
  {
    "query": "Get 500 recent transactions for Zcash sorted by block_time desc",
    "expected_chain": "zcash",
    "expected_endpoint": "transactions_list",
    "expected_s": "block_time(desc)",
    "expected_limit": 500
  },
  {
    "query": "List recent payments on stellar",
    "expected_chain": "stellar",
    "expected_endpoint": "stats",
    "expected_s": null,
    "expected_limit": null
  },
  {
    "query": "List latest 20 Bitcoin SV blocks sorted by id asc",
    "expected_chain": "bitcoin-sv",
    "expected_endpoint": "blocks_list",
    "expected_s": "id(asc)",
    "expected_limit": 20
  },
  {
    "query": "Stats for Zcash address",
    "expected_chain": "zcash",
    "expected_endpoint": "stats",
    "expected_identifier": null
  },
  {
    "query": "Ethereum stats for the latest Bitcoin block",
    "expected_chain": "bitcoin",
    "expected_endpoint": "stats"
  },
  {
    "query": "Litecoin wallet starting with LZ... but also check Dash recent transactions",
    "expected_chain": "litecoin",
    "expected_endpoint": "address_dashboard",
    "expected_identifier": "LZ..."
  }
]
"""


def run_cases(cases: List[Dict[str, Any]]) -> None:
    client = BlockchairClient(api_key=None)
    failures: List[str] = []
    for idx, case in enumerate(cases, start=1):
        q = case["query"]
        plan = plan_api_call(q, q, llm=None)
        url, params = client.build_request(plan)
        ok = True
        mismatches: List[str] = []

        def check(key: str, actual, expected) -> None:
            nonlocal ok
            if expected is None:
                if actual not in (None, ""):
                    ok = False
                    mismatches.append(f"{key}: expected None, got {actual}")
            else:
                if str(actual) != str(expected):
                    ok = False
                    mismatches.append(f"{key}: expected {expected}, got {actual}")

        check("chain", plan.chain, case.get("expected_chain"))
        check("endpoint", plan.endpoint_type, case.get("expected_endpoint"))
        check("s", params.get("s"), case.get("expected_s"))
        check("limit", params.get("limit"), case.get("expected_limit"))
        if "expected_identifier" in case:
            check("identifier", plan.identifier, case.get("expected_identifier"))

        if not ok:
            failures.append(
                f"Case {idx} failed. Query: {q}\nURL: {url}\nParams: {json.dumps(params, ensure_ascii=False)}\nMismatches: {mismatches}"
            )

    if failures:
        print("Some cases FAILED:\n" + "\n\n".join(failures))
        raise SystemExit(1)
    print("All cases passed.")


if __name__ == "__main__":
    cases = json.loads(TEST_CASES_JSON)
    run_cases(cases)

