from typing import Dict, List, Optional

import httpx
from langchain_openai import ChatOpenAI
from langchain_core.embeddings import Embeddings

from projects.config import settings
try:
    from gradio_client import Client as GradioClient  # type: ignore
except Exception:
    GradioClient = None  # type: ignore


class GradioChatWrapper:
    """Minimal wrapper to present a Chat-like interface for planning/synthesis.

    It exposes an .invoke(messages) method returning an object with .content,
    similar to LangChain ChatModels, using amd/gpt-oss-120b-chatbot via gradio_client.
    """

    def __init__(self, space: str = "amd/gpt-oss-120b-chatbot", temperature: float = 0.7) -> None:
        if GradioClient is None:
            raise RuntimeError("gradio_client not installed. Please install it to use OSS LLM.")
        self._client = GradioClient(space)
        self.temperature = temperature

    def with_structured_output(self, schema):  # passthrough; schema validation handled upstream
        return self

    def invoke(self, messages):
        # Extract system + last human
        system_prompt = None
        last_user = None
        for m in messages:
            try:
                role = getattr(m, "type") or getattr(m, "role", None)
            except Exception:
                role = None
            content = getattr(m, "content", str(m))
            if role == "system" and not system_prompt:
                system_prompt = content
            if role == "human" or role == "user":
                last_user = content
        resp = self._client.predict(
            message=last_user or "",
            system_prompt=system_prompt or "You are a helpful assistant.",
            temperature=self.temperature,
            api_name="/chat",
        )
        class R:
            content = str(resp)
        return R()



class JinaEmbeddings(Embeddings):
    def __init__(
        self,
        api_key: str,
        model: str = "jina-embeddings-v3",
        task: str = "retrieval.passage",
        base_url: str = "https://api.jina.ai/v1/embeddings",
        timeout: int = 60,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.task = task
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = httpx.Client(timeout=self.timeout)

    def _embed(self, inputs: List[str]) -> List[List[float]]:
        if not inputs:
            return []
        payload = {
            "model": self.model,
            "task": self.task,
            "input": inputs,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        resp = self._client.post(self.base_url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        vectors: List[List[float]] = [item.get("embedding", []) for item in data.get("data", [])]
        return vectors

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        out = self._embed([text])
        return out[0] if out else []


def _default_headers() -> Dict[str, str]:
    return {
        "HTTP-Referer": settings.HTTP_REFERER,
        "X-Title": settings.X_TITLE,
    }


def get_chat_llm(
    model: Optional[str] = None,
    temperature: float = 0.1,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> ChatOpenAI:
    api_key = api_key or settings.OPENROUTER_API_KEY
    if not api_key:
        raise RuntimeError("Missing OPENROUTER_API_KEY. Set it in env or Streamlit secrets.")
    return ChatOpenAI(
        model=model or settings.OPENROUTER_MODEL,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url or settings.OPENROUTER_BASE_URL,
        default_headers=_default_headers(),
    )


def get_oss_llm(temperature: float = 0.3):
    """Return a zero-key OSS LLM via gradio_client if available."""
    return GradioChatWrapper(temperature=temperature)


def get_embeddings(
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    task: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Embeddings:
    final_key = api_key or settings.JINA_API_KEY
    if not final_key:
        raise RuntimeError("Missing JINA_API_KEY. Set it in env or Streamlit secrets.")
    return JinaEmbeddings(
        api_key=final_key,
        model=model or settings.JINA_EMBED_MODEL,
        task=task or settings.JINA_EMBED_TASK,
        base_url=base_url or settings.JINA_EMBED_BASE_URL,
    )