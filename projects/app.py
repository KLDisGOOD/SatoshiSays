from __future__ import annotations

import json
import os
import re
import sys
from typing import List

# Ensure the package root is importable when running via Streamlit
CURRENT_DIR = os.path.dirname(__file__)
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_js_eval import streamlit_js_eval

from projects.blockchair_client import ApiCallPlan as ApiCallPlanRuntime
from projects.blockchair_client import BlockchairClient, PaginationPlan as RuntimePagination
from projects.config import settings
from projects.llm_providers import get_oss_llm
from projects.rag_pipeline import ApiCallPlan, load_retriever, plan_api_call, synthesize_answer

st.set_page_config(page_title=settings.APP_TITLE, layout="wide")
st.title(settings.APP_TITLE)

with st.sidebar:
    st.subheader("About this project")
    st.markdown(
        """
        This app is an open-source blockchain analytics assistant.  
        - **Ask questions** about Bitcoin, Ethereum, XRP (Ripple), and other major blockchains.
        - The app uses a local LLM (no API key required) to plan API calls to [Blockchair](https://blockchair.com/).
        - Documentation is retrieved using semantic search (Jina embeddings).
        - Results are summarized by the LLM and visualized when possible.
        - Built with LangChain, Streamlit, Jina, and Blockchair API.
        """
    )
    st.markdown("---")
    quota_placeholder = st.empty()

# Identify client once and cache in session
if "client_id" not in st.session_state:
    try:
        raw_host = streamlit_js_eval(js_expressions='window.location.hostname', key="client_ip_widget")
    except Exception:
        raw_host = None
    st.session_state["client_id"] = raw_host or "unknown"

# Show quota always in sidebar
_client_id = st.session_state.get("client_id", "unknown")
_quota_key = f"quota_{_client_id}"
_used = st.session_state.get(_quota_key, 0)
_remaining = max(0, 10 - _used)
with st.sidebar:
    quota_placeholder.info(f"Requests left: {_remaining} / 10")

query = st.text_input(
    "Ask about on-chain data (e.g., 'What's the current Bitcoin block height?')",
    placeholder="Try: What's the current Bitcoin block height?",
)

run = st.button("Run", type="primary")


@st.cache_resource(show_spinner=False)
def get_retriever_cached(jina_api_key: str):
    return load_retriever(k=4, api_key=jina_api_key)


@st.cache_resource(show_spinner=False)
def get_blockchair_client_cached(api_key: str):
    return BlockchairClient(api_key=api_key)


def _docs_to_text(docs: List):
    parts = []
    for d in docs:
        src = d.metadata.get("source") if hasattr(d, "metadata") else None
        header = f"Source: {src}" if src else ""
        parts.append(f"{header}\n{d.page_content}")
    return "\n\n---\n\n".join(parts)


xrp_pattern = re.compile(r"\b(xrp|ripple)\b", re.IGNORECASE)
amount_pattern = re.compile(r"(\d+[\d,]*\.?\d*)\s*\$|\$\s*(\d+[\d,]*\.?\d*)")


if run and query.strip():
    client_id = st.session_state.get("client_id", "unknown")

    # --- Simple VPN/proxy blocklist (hostname heuristics) ---
    blocked = False
    host_lc = str(client_id).lower()
    for bad in ("proxy", "vpn", "tor", "cloudflare", "fastly", "akamai"):
        if bad in host_lc:
            blocked = True
            break
    if blocked:
        st.error("Access blocked: VPNs or proxies are not allowed. Disable your VPN/proxy and refresh.")
        st.stop()

    # --- Per-IP quota: max 10 requests ---
    quota_key = f"quota_{client_id}"
    used = st.session_state.get(quota_key, 0)
    remaining = max(0, 10 - used)
    with st.sidebar:
        quota_placeholder.info(f"Requests left for this IP: {remaining} / 10. Email khalid.mt@outlook.com for more access.")
    if remaining <= 0:
        st.error("Quota exceeded. Email khalid.mt@outlook.com for more access.")
        st.stop()
    openrouter_key = settings.OPENROUTER_API_KEY
    openrouter_base = settings.OPENROUTER_BASE_URL
    blockchair_key = settings.BLOCKCHAIR_API_KEY
    jina_key = settings.JINA_API_KEY

    # Always use OSS model; no OpenRouter key required.
    if not jina_key:
        st.error("JINA_API_KEY is required for embeddings.")
        st.stop()

    try:
        retriever = get_retriever_cached(jina_key)
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    with st.spinner("Retrieving docs and planning API call..."):
        # Use newer retriever.invoke if available; fallback otherwise
        try:
            docs = retriever.invoke(query)  # type: ignore[attr-defined]
        except Exception:
            docs = retriever.get_relevant_documents(query)
        docs_text = _docs_to_text(docs)
        llm = get_oss_llm()
        plan: ApiCallPlan = plan_api_call(query, docs_text, llm=llm)

    # Heuristic corrections: ensure XRP query routes to ripple
    if xrp_pattern.search(query) and plan.chain != "ripple":
        plan.chain = "ripple"  # type: ignore[assignment]

    # If the query mentions latest transaction over $X on ripple, use transactions_list with s & q
    if plan.chain == "ripple" and "transaction" in query.lower() and ("latest" in query.lower() or "most recent" in query.lower()):
        usd_match = amount_pattern.search(query)
        usd_value = None
        if usd_match:
            amount_str = usd_match.group(1) or usd_match.group(2)
            if amount_str:
                usd_value = re.sub(r",", "", amount_str)
        params = {"s": "block_time(desc)"}
        if usd_value:
            params["q"] = f"value_usd(gt.{usd_value})"
        plan.endpoint_type = "transactions_list"  # type: ignore[assignment]
        plan.identifier = None
        plan.params = {**plan.params, **params}
        plan.pagination.limit = 1

    # --- Answer on top ---
    with st.spinner("Calling Blockchair API..."):
        client = get_blockchair_client_cached(blockchair_key or "")
        runtime_plan = ApiCallPlanRuntime(
            chain=plan.chain,
            endpoint_type=plan.endpoint_type,  # type: ignore[arg-type]
            identifier=plan.identifier,
            params=plan.params,
            pagination=RuntimePagination(
                limit=plan.pagination.limit,
                offset=plan.pagination.offset,
                max_pages=plan.pagination.max_pages,
            ),
        )
        try:
            api_json = client.execute(runtime_plan)
        except Exception as exc:
            st.error(f"API call failed: {exc}")
            st.stop()

    with st.spinner("Synthesizing answer..."):
        answer = synthesize_answer(query, docs_text, api_json, llm=llm)

    st.subheader("Answer")
    st.write(answer)

    # Increase quota usage on successful response
    st.session_state[quota_key] = used + 1
    remaining_after = max(0, 10 - st.session_state[quota_key])
    with st.sidebar:
        quota_placeholder.info(f"Requests left for this IP: {remaining_after} / 10")

    # --- LLM planned API call (collapsed by default) ---
    with st.expander("LLM planned (and normalized) API call:", expanded=False):
        st.json(json.loads(plan.model_dump_json()))

    # --- Raw JSON result (collapsed by default) ---
    with st.expander("Raw JSON result", expanded=False):
        st.json(api_json)

    with st.expander("Documentation context", expanded=False):
        for d in docs:
            st.markdown(f"**Source**: {d.metadata.get('source', 'unknown')}")
            st.text(d.page_content[:3000])
            st.markdown("---")

    def try_show_chart(obj):
        try:
            rows = None
            if isinstance(obj, dict):
                if isinstance(obj.get("data"), list):
                    rows = obj["data"]
                elif isinstance(obj.get("rows"), list):
                    rows = obj["rows"]
            if not rows or not isinstance(rows, list):
                return
            if len(rows) == 0 or not isinstance(rows[0], dict):
                return
            keys = rows[0].keys()
            time_key = next((k for k in keys if k in ("time", "date", "timestamp", "block_time")), None)
            value_key = next((k for k in keys if k in ("value", "value_usd", "count", "transactions")), None)
            if not time_key or not value_key:
                return
            df = pd.DataFrame(rows)
            fig = px.line(df, x=time_key, y=value_key, title="Trend")
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            return

    try_show_chart(api_json)

st.caption("SatoshiSays • Built by Khalid with ❤️")