from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

import requests

from projects.config import settings


AllowedChain = Literal[
    # Bitcoin family
    "bitcoin",
    "bitcoin-cash",
    "bitcoin-sv",
    # Other UTXO chains
    "litecoin",
    "dogecoin",
    "dash",
    "groestlcoin",
    "zcash",
    "ecash",
    # Account-based and others
    "ethereum",
    "ripple",
    "stellar",
    "monero",
    "cardano",
    "mixin",
]


@dataclass
class PaginationPlan:
    limit: Optional[int] = None
    offset: Optional[int] = None
    max_pages: int = 1


@dataclass
class ApiCallPlan:
    chain: AllowedChain
    endpoint_type: Literal[
        "stats",
        "address_dashboard",
        "transaction_dashboard",
        "block_dashboard",
        "raw_address",
        "raw_transaction",
        "transactions_list",
        "table_query",
    ]
    identifier: Optional[str] = None
    params: Dict[str, str] = field(default_factory=dict)
    pagination: PaginationPlan = field(default_factory=PaginationPlan)


class BlockchairClient:
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.api_key = api_key or settings.BLOCKCHAIR_API_KEY
        self.base_url = (base_url or settings.BLOCKCHAIR_BASE_URL).rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "BlockchairClient/1.0"})

    @staticmethod
    def _normalize_sort_value(sort_value: str) -> str:
        parts = [p.strip() for p in (sort_value or "").split(",") if p.strip()]
        normalized_parts: List[str] = []
        for p in parts:
            if "(" in p and ")" in p:
                field, rest = p.split("(", 1)
                direction = rest.split(")", 1)[0].strip().lower()
                normalized_parts.append(f"{field.strip()}({direction})")
            else:
                tokens = p.split()
                if len(tokens) >= 2 and tokens[-1].lower() in ("asc", "desc"):
                    direction = tokens[-1].lower()
                    field = " ".join(tokens[:-1]).strip().replace(" ", "_")
                    normalized_parts.append(f"{field}({direction})")
                else:
                    normalized_parts.append(p)
        return ",".join(normalized_parts)

    def _build_path(self, plan: ApiCallPlan) -> Tuple[str, Dict[str, str]]:
        chain = plan.chain
        etype = plan.endpoint_type
        ident = plan.identifier
        params = dict(plan.params or {})

        if etype == "stats":
            path = f"/{chain}/stats"
        elif etype == "address_dashboard":
            if not ident:
                raise ValueError("identifier (address) is required for address_dashboard")
            path = f"/{chain}/dashboards/address/{ident}"
        elif etype == "transaction_dashboard":
            if not ident:
                raise ValueError("identifier (tx hash) is required for transaction_dashboard")
            path = f"/{chain}/dashboards/transaction/{ident}"
        elif etype == "block_dashboard":
            if not ident:
                raise ValueError("identifier (block hash/height) is required for block_dashboard")
            path = f"/{chain}/dashboards/block/{ident}"
        elif etype == "raw_address":
            if not ident:
                raise ValueError("identifier (address) is required for raw_address")
            path = f"/{chain}/raw/address/{ident}"
        elif etype == "raw_transaction":
            if not ident:
                raise ValueError("identifier (tx hash) is required for raw_transaction")
            path = f"/{chain}/raw/transaction/{ident}"
        elif etype == "transactions_list":
            path = f"/{chain}/transactions"
            # Normalize sort param to Blockchair's 's'
            if "sort" in params and "s" not in params:
                params["s"] = params.pop("sort")
        elif etype == "blocks_list":
            path = f"/{chain}/blocks"
        elif etype == "table_query":
            path = f"/{chain}/tables"
            if "q" not in params and "query" in params:
                params["q"] = params.pop("query")
        else:
            raise ValueError(f"Unsupported endpoint_type: {etype}")

        # Normalize sort syntax if present
        if "s" in params and params["s"]:
            params["s"] = self._normalize_sort_value(params["s"])

        if self.api_key and "key" not in params:
            params["key"] = self.api_key
        return path, params

    def _request(self, path: str, params: Dict[str, str]) -> requests.Response:
        url = f"{self.base_url}{path}"
        resp = self.session.get(url, params=params, timeout=60)
        if resp.status_code == 429:
            for delay in (1, 2, 4, 8):
                time.sleep(delay)
                resp = self.session.get(url, params=params, timeout=60)
                if resp.ok:
                    break
        resp.raise_for_status()
        return resp

    def execute(self, plan: ApiCallPlan) -> Dict:
        path, params = self._build_path(plan)

        combined: Dict[str, List] = {}
        data: Dict = {}
        current_offset = plan.pagination.offset if plan.pagination.offset is not None else None
        limit = plan.pagination.limit
        max_pages = max(1, plan.pagination.max_pages)

        for page_idx in range(max_pages):
            page_params = dict(params)
            if limit is not None:
                page_params["limit"] = str(limit)
            if current_offset is not None:
                page_params["offset"] = str(current_offset)

            resp = self._request(path, page_params)
            data = resp.json()

            if not isinstance(data, dict) or any(k in data for k in ["data", "context", "rows"]) is False:
                return data

            merged_once = False
            for key in ("data", "rows"):
                if key in data and isinstance(data[key], list):
                    combined.setdefault(key, []).extend(data[key])
                    merged_once = True
            if not merged_once:
                if page_idx == 0:
                    return data
                else:
                    break

            if limit is None:
                break
            if current_offset is None:
                current_offset = 0
            current_offset += limit

        last_context = data.get("context") if isinstance(data, dict) else None
        result: Dict = {**combined}
        if last_context is not None:
            result["context"] = last_context
        return result

    def build_request(self, plan: ApiCallPlan) -> Tuple[str, Dict[str, str]]:
        """Helper to construct the request URL and params without executing it.
        Useful for validation/tests.
        """
        path, params = self._build_path(plan)
        page_params = dict(params)
        if plan.pagination.limit is not None:
            page_params["limit"] = str(plan.pagination.limit)
        if plan.pagination.offset is not None:
            page_params["offset"] = str(plan.pagination.offset)
        return f"{self.base_url}{path}", page_params