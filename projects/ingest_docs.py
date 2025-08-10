import os
import re
from typing import List, Tuple

import requests
from bs4 import BeautifulSoup
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from config import settings
from llm_providers import get_embeddings


DOCS_START_URL = "https://blockchair.com/api/docs"
ROOT_DATA_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data.txt")


def load_docs_from_file(path: str = ROOT_DATA_FILE) -> List[Tuple[str, str]]:
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        text = re.sub(r"\n{2,}", "\n\n", text)
        return [(f"file://{os.path.basename(path)}", text)]
    except Exception:
        return []


def fetch_blockchair_docs(start_url: str = DOCS_START_URL) -> List[Tuple[str, str]]:
    urls_to_fetch = {start_url}
    fetched: List[Tuple[str, str]] = []

    session = requests.Session()
    session.headers.update({"User-Agent": "DocsIngest/1.0"})

    def fetch_one(url: str) -> str:
        resp = session.get(url, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        main = soup.find("main") or soup
        text = main.get_text("\n", strip=True)
        text = re.sub(r"\n{2,}", "\n\n", text)
        return text

    try:
        text = fetch_one(start_url)
        fetched.append((start_url, text))
    except Exception as exc:
        print(f"Failed to fetch {start_url}: {exc}")

    try:
        resp = requests.get(start_url, timeout=30)
        soup = BeautifulSoup(resp.text, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if href.startswith("#"):
                continue
            if href.startswith("/"):
                href = f"https://blockchair.com{href}"
            if not href.startswith("https://blockchair.com/api"):
                continue
            urls_to_fetch.add(href)
            if len(urls_to_fetch) >= 5:
                break
    except Exception:
        pass

    for url in list(urls_to_fetch)[1:]:
        try:
            text = fetch_one(url)
            fetched.append((url, text))
        except Exception:
            continue

    return fetched


essential_separators = ["\n\n", "\n", ". ", ".", " "]


def build_vectorstore(pages: List[Tuple[str, str]], persist_dir: str) -> str:
    docs: List[Document] = []
    for url, text in pages:
        if not text or len(text) < 200:
            continue
        docs.append(Document(page_content=text, metadata={"source": url}))

    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150, separators=essential_separators)
    chunks = splitter.split_documents(docs)

    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    os.makedirs(persist_dir, exist_ok=True)
    vectorstore.save_local(persist_dir)
    return persist_dir


def main():
    # Prefer local data.txt if available
    pages = load_docs_from_file()
    if pages:
        print(f"Loaded docs from data file with {len(pages)} chunk(s)")
    else:
        print("Fetching Blockchair documentation from the web...")
        pages = fetch_blockchair_docs()
        print(f"Fetched {len(pages)} pages.")

    print("Building vectorstore...")
    path = build_vectorstore(pages, settings.VECTORSTORE_DIR)
    print(f"Vectorstore saved to: {path}")


if __name__ == "__main__":
    main()