from __future__ import annotations

import os

# Same OpenMP workaround as main.py if this module is imported first.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from backend.config import require_google_api_key

_embeddings: GoogleGenerativeAIEmbeddings | None = None


def _get_embeddings() -> GoogleGenerativeAIEmbeddings:
    global _embeddings
    if _embeddings is None:
        # embedding-001 was removed from the v1beta API; use current Gemini embedding model.
        _embeddings = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-2-preview",
            google_api_key=require_google_api_key(),
        )
    return _embeddings


def build_vectordb(file_path: str, session_id: str) -> FAISS:
    """
    Ingest a PDF into a session-scoped FAISS index on disk (no native C++ build needed on Windows).
    """
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    if not documents:
        raise ValueError("No text could be extracted from the PDF.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)

    embedder = _get_embeddings()
    vectordb = FAISS.from_documents(chunks, embedder)

    persist_dir = Path("vector_store") / session_id
    persist_dir.mkdir(parents=True, exist_ok=True)
    vectordb.save_local(str(persist_dir))

    return vectordb


def load_vectordb(session_id: str) -> FAISS:
    """Load a persisted FAISS index for an existing session."""
    persist_dir = Path("vector_store") / session_id
    if not persist_dir.exists():
        raise FileNotFoundError(f"No vector index for session {session_id}.")

    return FAISS.load_local(
        str(persist_dir),
        _get_embeddings(),
        allow_dangerous_deserialization=True,
    )
