"""
MCP stdio server exposing document_search over a session-scoped FAISS index.

Run standalone: python -m backend.mcp.document_server
Requires SESSION_ID in the environment (set by the FastAPI upload handler).
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from mcp.server.fastmcp import FastMCP

from backend.rag_pipeline import load_vectordb

mcp = FastMCP(
    "document-assistant",
    instructions=(
        "Document retrieval MCP server. Use document_search to find relevant "
        "passages from the user's uploaded PDF."
    ),
)

_vectordb = None


def _get_vectordb():
    global _vectordb
    if _vectordb is None:
        session_id = os.environ.get("SESSION_ID", "").strip()
        if not session_id:
            raise RuntimeError("SESSION_ID environment variable is required.")
        _vectordb = load_vectordb(session_id)
    return _vectordb


@mcp.tool()
def document_search(query: str) -> str:
    """Search the user-uploaded PDF for passages relevant to the question."""
    docs = _get_vectordb().similarity_search(query, k=4)
    if not docs:
        return "No matching passages found in the document."
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def _warm_index() -> None:
    """Load FAISS before the first tool call (avoids multi-minute cold-start timeouts)."""
    try:
        _get_vectordb()
    except Exception as exc:
        import sys

        print(f"MCP warm-up skipped: {exc}", file=sys.stderr)


if __name__ == "__main__":
    _warm_index()
    mcp.run(transport="stdio")
