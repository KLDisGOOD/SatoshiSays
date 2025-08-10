from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional, Tuple

from langchain.schema import HumanMessage, SystemMessage
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from pydantic import BaseModel, Field

from blockchair_client import AllowedChain
from config import settings
from llm_providers import get_chat_llm, get_embeddings


class PaginationPlan(BaseModel):
    limit: Optional[int] = Field(default=None, description="Max rows per request")
    offset: Optional[int] = Field(default=None, description="Starting offset")
    max_pages: int = Field(default=1, ge=1, description="How many pages to fetch")


class ApiCallPlan(BaseModel):
    chain: AllowedChain
    endpoint_type: str
    identifier: Optional[str] = None
    params: Dict[str, str] = Field(default_factory=dict)
    pagination: PaginationPlan = Field(default_factory=PaginationPlan)


def _ensure_vectorstore_exists(path: str) -> None:
    if not os.path.exists(path):
        raise RuntimeError(
            f"Vectorstore not found at '{path}'. Run: python -m projects.ingest_docs"
        )


def load_retriever(k: int = 4, api_key: Optional[str] = None):
    embeddings: Embeddings = get_embeddings(api_key=api_key)
    _ensure_vectorstore_exists(settings.VECTORSTORE_DIR)
    store = FAISS.load_local(
        settings.VECTORSTORE_DIR,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return store.as_retriever(search_kwargs={"k": k})


def _detect_chain(query_text: str) -> AllowedChain:
    text = query_text.lower()
    # Special override: if query mentions "latest bitcoin block", prefer bitcoin
    if re.search(r"\blatest\s+bitcoin\s+block\b", text):
        return "bitcoin"

    patterns = [
        (r"\b(bch|bitcoin\s*cash|bitcoin-cash)\b", "bitcoin-cash"),
        (r"\b(bsv|bitcoin\s*sv|bitcoin-sv)\b", "bitcoin-sv"),
        (r"\b(ltc|litecoin)\b", "litecoin"),
        (r"\b(doge|dogecoin)\b", "dogecoin"),
        (r"\b(dash)\b", "dash"),
        (r"\b(grs|groestlcoin)\b", "groestlcoin"),
        (r"\b(zec|zcash)\b", "zcash"),
        (r"\b(xec|ecash|e-cash)\b", "ecash"),
        (r"\b(eth|ethereum)\b", "ethereum"),
        (r"\b(xrp|ripple)\b", "ripple"),
        (r"\b(xlm|stellar)\b", "stellar"),
        (r"\b(xmr|monero)\b", "monero"),
        (r"\b(ada|cardano)\b", "cardano"),
        (r"\b(mixin)\b", "mixin"),
        (r"\b(bitcoin)\b", "bitcoin"),
    ]
    earliest: Optional[Tuple[int, AllowedChain]] = None
    for pat, chain in patterns:
        m = re.search(pat, text)
        if m:
            pos = m.start()
            if earliest is None or pos < earliest[0]:
                earliest = (pos, chain)  # type: ignore[assignment]
    return earliest[1] if earliest else "bitcoin"


def _heuristic_plan(query: str) -> ApiCallPlan:
    text = query.lower()
    chain: AllowedChain = _detect_chain(text)

    endpoint = "stats"
    identifier: Optional[str] = None
    params: Dict[str, str] = {}

    # Detect "wallet starting with <token>" and prefer address dashboard using the literal token
    start_with_match = re.search(r"wallet\s+starting\s+with\s+([^\s,]+)", query, flags=re.IGNORECASE)
    if start_with_match:
        endpoint = "address_dashboard"
        identifier = start_with_match.group(1)
    
    # Parse optional limit
    limit_match = re.search(r"\b(latest|recent)\s+(\d{1,4})\b", text)
    if not limit_match:
        limit_match = re.search(r"\b(\d{1,4})\s+(latest|recent)\b", text)
    limit_val: Optional[int] = None
    if limit_match:
        try:
            limit_val = int(next(g for g in limit_match.groups() if g and g.isdigit()))
        except Exception:
            limit_val = None

    # Parse sort directive
    sort_match = re.search(r"sorted\s+by\s+([a-zA-Z0-9_ ]+)\s+(asc|desc)", text)
    sort_value = ""
    if sort_match:
        field_raw = sort_match.group(1).strip()
        direction = sort_match.group(2).lower()
        sort_value = _normalize_sort_value(f"{field_raw} {direction}")
    else:
        loose_sort = re.search(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s+(asc|desc)\b", text)
        if loose_sort:
            sort_value = _normalize_sort_value(loose_sort.group(0))

    if not start_with_match:
        if any(w in text for w in ["address", "wallet"]) and re.search(r"[a-z0-9]{20,}", text):
            endpoint = "address_dashboard"
            identifier = re.search(r"[a-zA-Z0-9]{20,}", query).group(0) if re.search(r"[a-zA-Z0-9]{20,}", query) else None
        elif any(w in text for w in ["tx", "transaction"]) and re.search(r"[a-f0-9]{32,}", text):
            endpoint = "transaction_dashboard"
            identifier = re.search(r"[a-fA-F0-9]{32,}", query).group(0) if re.search(r"[a-fA-F0-9]{32,}", query) else None
        elif (
            re.search(r"\bblock(?:\s+(?:hash|height))?\s+[A-Za-z0-9]{3,}\b", text)
            or re.search(r"\bheight\s+\d{3,}\b", text)
        ):
            endpoint = "block_dashboard"
            # Prefer a concrete id/height after the keyword
            m = re.search(r"\bblock(?:\s+(?:hash|height))?\s+([A-Za-z0-9]{3,})\b", query, flags=re.IGNORECASE)
            if not m:
                m = re.search(r"\bheight\s+(\d{3,})\b", query, flags=re.IGNORECASE)
            identifier = m.group(1) if m else None
        elif any(w in text for w in ["latest", "recent"]) and "transaction" in text:
            endpoint = "transactions_list"
            params["s"] = _normalize_sort_value(sort_value or "block_time desc")
        elif any(w in text for w in ["latest", "recent"]) and "blocks" in text:
            endpoint = "blocks_list"
            params["s"] = _normalize_sort_value(sort_value or "id desc")

    return ApiCallPlan(
        chain=chain,
        endpoint_type=endpoint,
        identifier=identifier,
        params=params,
        pagination=PaginationPlan(limit=limit_val, offset=None, max_pages=1),
    )


AllowedChainsList: Tuple[AllowedChain, ...] = (
    "bitcoin",
    "bitcoin-cash",
    "bitcoin-sv",
    "litecoin",
    "dogecoin",
    "dash",
    "groestlcoin",
    "zcash",
    "ecash",
    "ethereum",
    "ripple",
    "stellar",
    "monero",
    "cardano",
    "mixin",
)


AllowedEndpoints: Tuple[str, ...] = (
    "stats",
    "address_dashboard",
    "transaction_dashboard",
    "block_dashboard",
    "raw_address",
    "raw_transaction",
    "transactions_list",
    "table_query",
    "blocks_list",
)


def _value_present_in_text(value: str, text: str) -> bool:
    try:
        return value is not None and value != "" and (value in text)
    except Exception:
        return False


def normalize_sort_param(params: Dict[str, str]) -> Dict[str, str]:
    """Normalize Blockchair 's' sort parameter to parentheses syntax."""
    out = dict(params or {})
    if "s" in out and out["s"]:
        out["s"] = _normalize_sort_value(out["s"])  # type: ignore[name-defined]
    return out


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


def _is_valid_identifier_format(identifier: str) -> bool:
    if not identifier:
        return False
    # Common transaction hash pattern (64 hex)
    if re.fullmatch(r"[0-9a-fA-F]{64}", identifier):
        return True
    # Block height/number (up to 10 digits)
    if re.fullmatch(r"\d{1,10}", identifier):
        return True
    # Generic address-like (b58, bech32, xrp 'r' prefix, etc.)
    if re.fullmatch(r"[0-9A-Za-z]{20,64}", identifier):
        return True
    return False


def _sort_value_allowed(sort_value: str, text: str) -> bool:
    # Accept if normalized value exists or if raw "field dir" phrase exists
    normalized = _normalize_sort_value(sort_value)
    if normalized and normalized in text:
        return True
    for part in normalized.split(","):
        m = re.match(r"^([a-zA-Z_][a-zA-Z0-9_]*)\((asc|desc)\)$", part.strip())
        if not m:
            continue
        field = m.group(1)
        direction = m.group(2)
        # Look for "field asc" or "field desc" in the text (allow spaces or underscores)
        pattern = rf"\b{re.escape(field).replace('_', '[ _]')}\s+{direction}\b"
        if re.search(pattern, text, flags=re.IGNORECASE):
            return True
    return False


def sanitize_plan(plan: ApiCallPlan, query: str, docs_text: str) -> ApiCallPlan:
    combined_text = (query or "") + "\n" + (docs_text or "")

    # Chain guard
    chain: AllowedChain = plan.chain if plan.chain in AllowedChainsList else "bitcoin"  # type: ignore[assignment]

    # Endpoint guard
    endpoint = plan.endpoint_type if plan.endpoint_type in AllowedEndpoints else "stats"

    # Identifier must be present in the input texts and match a known format
    identifier = plan.identifier
    # Special allowance: queries like "wallet starting with <token>" may provide a prefix token
    has_starting_with = re.search(r"wallet\s+starting\s+with\s+([^\s,]+)", query, flags=re.IGNORECASE) is not None
    if identifier:
        if not _value_present_in_text(identifier, combined_text):
            identifier = None
        else:
            if not has_starting_with and not _is_valid_identifier_format(identifier):
                identifier = None

    # Params: keep only params whose values are present in query/docs
    new_params: Dict[str, str] = {}
    for k, v in (plan.params or {}).items():
        v_str = str(v)
        if k == "s":
            # Special allowance for sort: accept if raw or normalized form is present
            if _sort_value_allowed(v_str, combined_text):
                new_params[k] = _normalize_sort_value(v_str)
        else:
            if _value_present_in_text(v_str, combined_text):
                new_params[k] = v_str

    # Normalize sort parameter (s)
    new_params = normalize_sort_param(new_params)

    # If endpoint requires identifier but it's missing, downgrade to safe default
    requires_identifier = endpoint in {
        "address_dashboard",
        "transaction_dashboard",
        "block_dashboard",
        "raw_address",
        "raw_transaction",
    }
    if requires_identifier and not identifier:
        endpoint = "stats"
        new_params = {}

    # Stats should not carry identifier or params
    if endpoint == "stats":
        identifier = None
        new_params = {}

    # Pagination guard
    pagination = plan.pagination or PaginationPlan()
    if pagination and pagination.max_pages < 1:
        pagination.max_pages = 1
    # Clamp limit to [1,1000]
    if pagination and pagination.limit is not None:
        try:
            pagination.limit = max(1, min(int(pagination.limit), 1000))
        except Exception:
            pagination.limit = None
    # Ensure offset is non-negative integer
    if pagination and pagination.offset is not None:
        try:
            pagination.offset = max(0, int(pagination.offset))
        except Exception:
            pagination.offset = None

    # Chain-endpoint capability map (conservative for limited chains)
    SUPPORTED_ENDPOINTS: Dict[str, set] = {
        "bitcoin": {"stats", "address_dashboard", "transaction_dashboard", "block_dashboard", "transactions_list", "blocks_list"},
        "bitcoin-cash": {"stats", "address_dashboard", "transaction_dashboard", "block_dashboard", "transactions_list", "blocks_list"},
        "bitcoin-sv": {"stats", "address_dashboard", "transaction_dashboard", "block_dashboard", "transactions_list", "blocks_list"},
        "litecoin": {"stats", "address_dashboard", "transaction_dashboard", "block_dashboard", "transactions_list", "blocks_list"},
        "dogecoin": {"stats", "address_dashboard", "transaction_dashboard", "block_dashboard", "transactions_list", "blocks_list"},
        "dash": {"stats", "address_dashboard", "transaction_dashboard", "block_dashboard", "transactions_list", "blocks_list"},
        "groestlcoin": {"stats", "address_dashboard", "transaction_dashboard", "block_dashboard", "transactions_list", "blocks_list"},
        "zcash": {"stats", "address_dashboard", "transaction_dashboard", "block_dashboard", "transactions_list", "blocks_list"},
        "ecash": {"stats", "address_dashboard", "transaction_dashboard", "block_dashboard", "transactions_list", "blocks_list"},
        "ethereum": {"stats", "address_dashboard", "transaction_dashboard", "block_dashboard", "transactions_list", "blocks_list"},
        # Ripple/Stellar/Monero/Cardano/Mixin are limited in Blockchair
        "ripple": {"stats", "transaction_dashboard", "transactions_list"},
        "stellar": {"stats"},
        "monero": {"stats"},
        "cardano": {"stats"},
        "mixin": {"stats"},
    }

    if endpoint not in SUPPORTED_ENDPOINTS.get(chain, {"stats"}):
        endpoint = "stats"
        identifier = None
        new_params = {}

    return ApiCallPlan(
        chain=chain,
        endpoint_type=endpoint,
        identifier=identifier,
        params=new_params,
        pagination=pagination,
    )


def plan_api_call(query: str, docs_text: str, llm=None) -> ApiCallPlan:
    # Start from a conservative heuristic plan
    base_plan = _heuristic_plan(query)

    # If no LLM provided, return sanitized heuristic plan to avoid API key dependency
    if llm is None:
        return sanitize_plan(base_plan, query, docs_text)

    try:
        schema_llm = llm.with_structured_output(ApiCallPlan)  # type: ignore[attr-defined]
        system = SystemMessage(
            content=(
                "You are an expert on the Blockchair API. You will refine an initial API call plan.\n"
                "Strict constraints:\n"
                "- Only use identifiers or parameter values explicitly found in the user query or the provided documentation snippets.\n"
                "- Do not fabricate values. If information is missing, leave identifier null and params empty.\n"
                "- Allowed chains: [bitcoin, bitcoin-cash, bitcoin-sv, litecoin, dogecoin, dash, groestlcoin, zcash, ecash, ethereum, ripple, stellar, monero, cardano, mixin].\n"
                "- Allowed endpoints: [stats, address_dashboard, transaction_dashboard, block_dashboard, raw_address, raw_transaction, transactions_list, table_query].\n"
                "- Use Blockchair parameter names (e.g., 's' for sort, 'q' for SQL-like filters) but only when their values are present in the input.\n"
                "- If the chain or endpoint would be invalid or underspecified, prefer 'bitcoin' + 'stats'."
            )
        )
        human = HumanMessage(
            content=(
                "User question:\n" + query + "\n\n"
                "Relevant docs (may be truncated):\n" + docs_text[:6000] + "\n\n"
                "Initial plan (JSON):\n" + base_plan.model_dump_json() + "\n\n"
                "Return ONLY the structured object."
            )
        )
        result = schema_llm.invoke([system, human])
        if isinstance(result, dict):
            candidate = ApiCallPlan.model_validate(result)
        elif isinstance(result, ApiCallPlan):
            candidate = result
        else:
            raise ValueError("Unexpected LLM output type")

        # Post-output validation and sanitization
        return sanitize_plan(candidate, query, docs_text)
    except Exception:
        # If LLM fails or returns an invalid structure, fall back to heuristic
        return sanitize_plan(base_plan, query, docs_text)


def synthesize_answer(query: str, docs_text: str, api_json: Dict, llm=None) -> str:
    if llm is None:
        llm = get_chat_llm()

    system = SystemMessage(
        content=(
            "You are a helpful assistant summarizing blockchain analytics results."
            " Explain clearly, cite key numbers, and mention uncertainties."
        )
    )
    human = HumanMessage(
        content=(
            "Question:\n" + query + "\n\n"
            "Relevant docs (truncated):\n" + docs_text[:4000] + "\n\n"
            "API JSON result (truncated):\n" + json.dumps(api_json, ensure_ascii=False)[:6000] + "\n\n"
            "Write a concise, user-friendly answer."
        )
    )
    out = llm.invoke([system, human])
    try:
        return out.content if hasattr(out, "content") else str(out)
    except Exception:
        return str(out)

