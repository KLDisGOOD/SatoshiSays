import os
from typing import Optional

# Load variables from a local .env file, if present
try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except Exception:
    pass


def _get_streamlit_secret(key: str) -> Optional[str]:
    try:
        import streamlit as st  # type: ignore

        return st.secrets.get(key)  # type: ignore[attr-defined]
    except Exception:
        return None


def get_env(key: str, default: Optional[str] = None) -> Optional[str]:
    return os.getenv(key) or _get_streamlit_secret(key) or default


class Settings:
    """Centralized settings loader.

    Values are read from environment variables or Streamlit secrets.
    """

    # OpenRouter
    OPENROUTER_API_KEY: Optional[str] = get_env("OPENROUTER_API_KEY")
    OPENROUTER_BASE_URL: str = get_env("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1") or "https://openrouter.ai/api/v1"
    OPENROUTER_MODEL: str = get_env("OPENROUTER_MODEL", "openai/gpt-oss-20b:free") or "openai/gpt-oss-20b:free"
    HTTP_REFERER: str = get_env("HTTP_REFERER", "http://localhost") or "http://localhost"
    X_TITLE: str = get_env("X_TITLE", "SatoshiSays") or "SatoshiSays"

    # Jina Embeddings
    JINA_API_KEY: Optional[str] = get_env("JINA_API_KEY")
    JINA_EMBED_MODEL: str = get_env("JINA_EMBED_MODEL", "jina-embeddings-v3") or "jina-embeddings-v3"
    JINA_EMBED_TASK: str = get_env("JINA_EMBED_TASK", "retrieval.passage") or "retrieval.passage"
    JINA_EMBED_BASE_URL: str = get_env("JINA_EMBED_BASE_URL", "https://api.jina.ai/v1/embeddings") or "https://api.jina.ai/v1/embeddings"

    # Blockchair
    BLOCKCHAIR_API_KEY: Optional[str] = get_env("BLOCKCHAIR_API_KEY")
    BLOCKCHAIR_BASE_URL: str = get_env("BLOCKCHAIR_BASE_URL", "https://api.blockchair.com") or "https://api.blockchair.com"

    # App
    VECTORSTORE_DIR: str = get_env("VECTORSTORE_DIR", "vectorstore/blockchair_faiss") or "vectorstore/blockchair_faiss"
    APP_TITLE: str = get_env("APP_TITLE", "SatoshiSays") or "SatoshiSays"


settings = Settings()