import os
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


@lru_cache
def require_google_api_key() -> str:
    """Required for PDF embeddings (indexing only)."""
    key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return key


@lru_cache
def require_groq_api_key() -> str:
    """Required for chat agent (Groq LLM)."""
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Get a key at https://console.groq.com and add it to .env."
        )
    return key


def groq_chat_model() -> str:
    """Override with GROQ_CHAT_MODEL in .env."""
    return (os.getenv("GROQ_CHAT_MODEL") or "llama-3.3-70b-versatile").strip()


def groq_chat_max_retries() -> int:
    raw = (os.getenv("GROQ_CHAT_MAX_RETRIES") or "1").strip()
    try:
        return max(0, min(10, int(raw)))
    except ValueError:
        return 1
