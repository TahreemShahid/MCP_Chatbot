"""Persistent MCP stdio clients — one subprocess per upload session."""

from __future__ import annotations

import asyncio
import contextlib
import os
import sys
import threading
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class McpSessionHandle:
    """Background MCP client connected to the document_search server."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name=f"mcp-{session_id[:8]}")
        self._ready = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._session: ClientSession | None = None
        self._stdio_cm: Any = None
        self._session_cm: Any = None

    def start(self) -> None:
        self._thread.start()
        if not self._ready.wait(timeout=60):
            raise RuntimeError("MCP document server failed to start within 60s.")

    def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        if self._loop is None:
            raise RuntimeError("MCP client is not running.")
        future = asyncio.run_coroutine_threadsafe(
            self._call_tool_async(name, arguments),
            self._loop,
        )
        try:
            return future.result(timeout=180)
        except TimeoutError as exc:
            raise TimeoutError(
                "MCP document_search timed out (cold start can take up to 3 minutes on first call)."
            ) from exc

    async def _call_tool_async(self, name: str, arguments: dict[str, Any]) -> str:
        assert self._session is not None
        response = await self._session.call_tool(name, arguments=arguments)
        parts: list[str] = []
        for block in response.content:
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
        return "\n".join(parts) if parts else "No matching passages found in the document."

    def close(self) -> None:
        if self._loop is None:
            return
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)
        self._loop = None

    def _run_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._connect())
            self._ready.set()
            self._loop.run_forever()
        finally:
            with contextlib.suppress(Exception):
                self._loop.run_until_complete(self._shutdown())
            self._loop.close()

    async def _connect(self) -> None:
        env = dict(os.environ)
        env["SESSION_ID"] = self.session_id
        server_params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "backend.mcp.document_server"],
            env=env,
            cwd=str(PROJECT_ROOT),
        )
        self._stdio_cm = stdio_client(server_params)
        read, write = await self._stdio_cm.__aenter__()
        self._session_cm = ClientSession(read, write)
        self._session = await self._session_cm.__aenter__()
        await self._session.initialize()

    async def _shutdown(self) -> None:
        if self._session_cm is not None:
            await self._session_cm.__aexit__(None, None, None)
        if self._stdio_cm is not None:
            await self._stdio_cm.__aexit__(None, None, None)


_handles: dict[str, McpSessionHandle] = {}


def get_mcp_handle(session_id: str) -> McpSessionHandle:
    if session_id not in _handles:
        handle = McpSessionHandle(session_id)
        handle.start()
        _handles[session_id] = handle
    return _handles[session_id]


def close_mcp_handle(session_id: str) -> None:
    handle = _handles.pop(session_id, None)
    if handle is not None:
        handle.close()
