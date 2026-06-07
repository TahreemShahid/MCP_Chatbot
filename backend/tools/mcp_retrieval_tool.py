"""LangChain tool that delegates retrieval to the MCP document_search server."""

from langchain.tools import Tool

from backend.mcp.client_pool import get_mcp_handle


def create_mcp_retrieval_tool(session_id: str) -> Tool:
    """Expose MCP document_search as a LangChain tool for the ReAct agent."""

    def search_docs(query: str) -> str:
        handle = get_mcp_handle(session_id)
        return handle.call_tool("document_search", {"query": query})

    return Tool(
        name="document_search",
        func=search_docs,
        description=(
            "Search the user-uploaded PDF for passages relevant to the question. "
            "Implemented via MCP (Model Context Protocol) for standardized LLM-to-data "
            "communication. Input must be a specific search query."
        ),
    )
