import json
import os

# Windows: FAISS + NumPy/MKL (or other deps) can load two OpenMP runtimes and abort.
# Set before importing anything that pulls in faiss/numpy.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import asyncio
import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.config import require_google_api_key, require_groq_api_key
from backend.mcp.client_pool import close_mcp_handle
from backend.rag_pipeline import build_vectordb
from backend.tools.agent import create_agent
from backend.tools.mcp_retrieval_tool import create_mcp_retrieval_tool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    try:
        require_google_api_key()
    except RuntimeError:
        logger.warning(
            "GOOGLE_API_KEY is not set. PDF upload/indexing will fail until you add it to .env."
        )
    try:
        require_groq_api_key()
    except RuntimeError:
        logger.warning(
            "GROQ_API_KEY is not set. Chat will fail until you add it to .env (https://console.groq.com)."
        )
    yield


app = FastAPI(
    title="Agentic Document Assistant",
    description=(
        "FastAPI + LangChain agent with MCP-integrated tools: RAG over uploaded PDFs "
        "via FAISS; chat powered by Groq, embeddings by Gemini."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

user_agents: dict[str, Any] = {}


def _friendly_api_error(exc: Exception) -> str:
    text = str(exc)
    if "429" in text or "quota" in text.lower() or "ResourceExhausted" in type(exc).__name__:
        return (
            "Groq API rate limit exceeded. Try GROQ_CHAT_MODEL=llama-3.1-8b-instant in .env, "
            "wait a minute, restart the API server, and start a new conversation."
        )
    return text


class ChatRequest(BaseModel):
    session_id: str = Field(..., examples=["550e8400-e29b-41d4-a716-446655440000"])
    message: str = Field(..., examples=["What are the main conclusions in the document?"])


class ChatResponse(BaseModel):
    response: str


class UploadResponse(BaseModel):
    session_id: str
    message: str


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "Agentic Document Assistant",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename.")

    safe_name = Path(file.filename).name
    if not safe_name.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    try:
        require_google_api_key()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

    session_id = str(uuid.uuid4())
    uploads_dir = Path("backend") / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    file_path = uploads_dir / f"{session_id}_{safe_name}"

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file.")

    file_path.write_bytes(data)

    try:
        build_vectordb(str(file_path), session_id)
        doc_tool = create_mcp_retrieval_tool(session_id)
        agent_executor = create_agent(tools=[doc_tool])
        user_agents[session_id] = agent_executor
        return UploadResponse(
            session_id=session_id,
            message="Document processed, MCP server linked, and agent ready.",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Upload processing failed")
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        if file_path.exists():
            file_path.unlink(missing_ok=True)


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    if request.session_id not in user_agents:
        raise HTTPException(status_code=404, detail="Session not found.")

    try:
        require_groq_api_key()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

    agent = user_agents[request.session_id]
    try:
        result = await asyncio.to_thread(agent.invoke, {"input": request.message})
        return ChatResponse(response=result["output"])
    except Exception as e:
        logger.exception("Chat failed")
        if "429" in str(e) or "quota" in str(e).lower() or "ResourceExhausted" in type(e).__name__:
            raise HTTPException(
                status_code=429,
                detail=(
                    "Groq API rate limit exceeded. Try GROQ_CHAT_MODEL=llama-3.1-8b-instant in .env "
                    "or wait for the limit to reset."
                ),
            ) from e
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """Server-Sent Events stream of agent tokens for the React UI."""
    if request.session_id not in user_agents:
        raise HTTPException(status_code=404, detail="Session not found.")

    try:
        require_groq_api_key()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

    agent = user_agents[request.session_id]

    async def event_generator() -> AsyncIterator[str]:
        queue: asyncio.Queue[str | None] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def run_stream() -> None:
            try:
                for chunk in agent.stream({"input": request.message}):
                    output = chunk.get("output")
                    if output:
                        loop.call_soon_threadsafe(
                            queue.put_nowait,
                            json.dumps({"token": output}),
                        )
            except Exception as exc:
                loop.call_soon_threadsafe(
                    queue.put_nowait,
                    json.dumps({"error": _friendly_api_error(exc)}),
                )
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        asyncio.create_task(asyncio.to_thread(run_stream))

        while True:
            item = await queue.get()
            if item is None:
                yield f"data: {json.dumps({'done': True})}\n\n"
                break
            yield f"data: {item}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.delete("/session/{session_id}")
async def delete_session(session_id: str) -> dict[str, str]:
    user_agents.pop(session_id, None)
    close_mcp_handle(session_id)
    return {"status": "deleted"}


_FRONTEND_DIST = Path(__file__).resolve().parent / "frontend" / "dist"
if _FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=str(_FRONTEND_DIST), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
