"""
Quantitative evaluation: upload latency, chat latency, and answer accuracy.

Usage (API must be running on port 8000):
    python evaluate_chatbot.py
    python evaluate_chatbot.py --api http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv
from fpdf import FPDF

load_dotenv()

DEFAULT_API = "http://127.0.0.1:8000"

# Ground-truth facts embedded in the generated evaluation PDF.
EVAL_FACTS = {
    "company": "NovaTech Labs",
    "product": "Orion Analytics Platform",
    "founded": "2019",
    "headquarters": "Austin, Texas",
    "annual_revenue": "$42 million",
    "ceo": "Dr. Elena Morales",
}

TEST_CASES = [
    {
        "question": "What company is described in the document?",
        "keywords": ["novatech", "labs"],
    },
    {
        "question": "What is the name of the product?",
        "keywords": ["orion", "analytics"],
    },
    {
        "question": "When was the company founded?",
        "keywords": ["2019"],
    },
    {
        "question": "Where is the headquarters located?",
        "keywords": ["austin", "texas"],
    },
    {
        "question": "Who is the CEO?",
        "keywords": ["elena", "morales"],
    },
]


def build_eval_pdf() -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.multi_cell(
        0,
        8,
        (
            f"{EVAL_FACTS['company']} was founded in {EVAL_FACTS['founded']} and is headquartered "
            f"in {EVAL_FACTS['headquarters']}. The company develops the "
            f"{EVAL_FACTS['product']}, an enterprise data analytics solution. "
            f"In the most recent fiscal year, {EVAL_FACTS['company']} reported "
            f"annual revenue of {EVAL_FACTS['annual_revenue']}. "
            f"The Chief Executive Officer is {EVAL_FACTS['ceo']}."
        ),
    )
    out = pdf.output()
    return out if isinstance(out, (bytes, bytearray)) else out.encode("latin-1")


def keyword_match(answer: str, keywords: list[str]) -> bool:
    text = answer.lower()
    return all(kw in text for kw in keywords)


def upload_document(api_base: str, pdf_bytes: bytes) -> tuple[str | None, float, str]:
    url = f"{api_base.rstrip('/')}/upload"
    files = {"file": ("eval_document.pdf", pdf_bytes, "application/pdf")}
    start = time.perf_counter()
    try:
        response = requests.post(url, files=files, timeout=600)
        latency = time.perf_counter() - start
        if response.status_code != 200:
            detail = response.json().get("detail", response.text)
            return None, latency, str(detail)
        data = response.json()
        return data.get("session_id"), latency, data.get("message", "ok")
    except requests.RequestException as exc:
        return None, time.perf_counter() - start, str(exc)


def chat(api_base: str, session_id: str, message: str) -> tuple[str | None, float, str]:
    url = f"{api_base.rstrip('/')}/chat"
    start = time.perf_counter()
    try:
        response = requests.post(
            url,
            json={"session_id": session_id, "message": message},
            timeout=300,
        )
        latency = time.perf_counter() - start
        if response.status_code != 200:
            detail = response.json().get("detail", response.text)
            return None, latency, str(detail)
        return response.json().get("response"), latency, ""
    except requests.RequestException as exc:
        return None, time.perf_counter() - start, str(exc)


def health_check(api_base: str) -> bool:
    try:
        response = requests.get(f"{api_base.rstrip('/')}/health", timeout=10)
        return response.status_code == 200
    except requests.RequestException:
        return False


def evaluate_retrieval(session_id: str) -> tuple[int, list[float]]:
    """MCP retrieval accuracy/latency without calling the chat LLM."""
    from backend.mcp.client_pool import close_mcp_handle, get_mcp_handle

    passed = 0
    latencies: list[float] = []
    print("Starting MCP client (loads FAISS index in subprocess)…")
    handle = get_mcp_handle(session_id)

    print("\n--- MCP retrieval layer (no LLM) ---\n")
    try:
        for index, case in enumerate(TEST_CASES, 1):
            question = case["question"]
            keywords = case["keywords"]
            print(f"[Retrieval {index}/{len(TEST_CASES)}] {question}")

            start = time.perf_counter()
            try:
                text = handle.call_tool("document_search", {"query": question})
                latency = time.perf_counter() - start
                latencies.append(latency)
                ok = keyword_match(text, keywords)
                if ok:
                    passed += 1
                    print(f"  Pass ({latency:.3f}s)")
                else:
                    print(f"  Fail ({latency:.3f}s) — expected keywords: {keywords}")
                preview = text.replace("\n", " ")[:120]
                print(f"  Context: {preview}{'...' if len(text) > 120 else ''}")
            except Exception as exc:
                latencies.append(time.perf_counter() - start)
                print(f"  Error: {exc}")
    finally:
        with contextlib.suppress(Exception):
            close_mcp_handle(session_id)

    return passed, latencies


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate MCP document chatbot.")
    parser.add_argument("--api", default=os.environ.get("DOCUMENT_ASSISTANT_API", DEFAULT_API))
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Evaluate MCP document_search only (no Groq chat calls).",
    )
    args = parser.parse_args()
    api_base = args.api.rstrip("/")

    if not health_check(api_base):
        print(f"API not reachable at {api_base}. Start the server: python main.py")
        return 1

    if not os.getenv("GOOGLE_API_KEY"):
        print("Warning: GOOGLE_API_KEY not set; PDF upload/indexing may fail.")
    if not args.retrieval_only and not os.getenv("GROQ_API_KEY"):
        print("Warning: GROQ_API_KEY not set; chat evaluation may fail.")

    print("\n--- MCP Document Assistant Evaluation ---\n")

    pdf_bytes = build_eval_pdf()
    session_id, upload_latency, upload_err = upload_document(api_base, pdf_bytes)
    print(f"Upload latency: {upload_latency:.3f}s")
    if not session_id:
        print(f"Upload failed: {upload_err}")
        return 1
    print(f"Session: {session_id}\n")

    if args.retrieval_only:
        passed, chat_latencies = evaluate_retrieval(session_id)
        total = len(TEST_CASES)
        accuracy = (passed / total) * 100 if total else 0.0
        avg_chat = sum(chat_latencies) / len(chat_latencies) if chat_latencies else 0.0
        summary = {
            "mode": "retrieval_only",
            "upload_latency_sec": round(upload_latency, 3),
            "retrieval_accuracy_pct": round(accuracy, 1),
            "tests_passed": passed,
            "tests_total": total,
            "avg_retrieval_latency_sec": round(avg_chat, 3),
        }
        print("\n=============================================")
        print("     RETRIEVAL-ONLY EVALUATION SUMMARY       ")
        print("=============================================")
        print(f"Upload latency              : {summary['upload_latency_sec']:.3f}s")
        print(
            f"MCP retrieval accuracy      : {summary['retrieval_accuracy_pct']:.1f}% "
            f"({summary['tests_passed']}/{summary['tests_total']} passed)"
        )
        print(f"Avg retrieval latency       : {summary['avg_retrieval_latency_sec']:.3f}s")
        print("=============================================")
        Path("eval_results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return 0 if passed == total else 2

    chat_latencies: list[float] = []
    passed = 0

    for index, case in enumerate(TEST_CASES, 1):
        if index > 1:
            time.sleep(1)  # reduce Groq burst rate limits between agent turns
        question = case["question"]
        keywords = case["keywords"]
        print(f"[Test {index}/{len(TEST_CASES)}] {question}")

        answer, latency, err = chat(api_base, session_id, question)
        chat_latencies.append(latency)

        if err or not answer:
            print(f"  Failed ({latency:.3f}s): {err or 'empty response'}")
            continue

        ok = keyword_match(answer, keywords)
        if ok:
            passed += 1
            print(f"  Pass ({latency:.3f}s)")
        else:
            print(f"  Fail ({latency:.3f}s) — expected keywords: {keywords}")
        preview = answer.replace("\n", " ")[:160]
        print(f"  Answer: {preview}{'...' if len(answer) > 160 else ''}")

    total = len(TEST_CASES)
    accuracy = (passed / total) * 100 if total else 0.0
    avg_chat = sum(chat_latencies) / len(chat_latencies) if chat_latencies else 0.0

    summary = {
        "upload_latency_sec": round(upload_latency, 3),
        "accuracy_pct": round(accuracy, 1),
        "tests_passed": passed,
        "tests_total": total,
        "avg_chat_latency_sec": round(avg_chat, 3),
        "min_chat_latency_sec": round(min(chat_latencies), 3) if chat_latencies else 0.0,
        "max_chat_latency_sec": round(max(chat_latencies), 3) if chat_latencies else 0.0,
    }

    print("\n=============================================")
    print("          FINAL EVALUATION SUMMARY           ")
    print("=============================================")
    print(f"Upload latency              : {summary['upload_latency_sec']:.3f}s")
    print(
        f"Answer accuracy             : {summary['accuracy_pct']:.1f}% "
        f"({summary['tests_passed']}/{summary['tests_total']} passed)"
    )
    print(f"Avg chat latency            : {summary['avg_chat_latency_sec']:.3f}s")
    print(f"Fastest chat response       : {summary['min_chat_latency_sec']:.3f}s")
    print(f"Slowest chat response       : {summary['max_chat_latency_sec']:.3f}s")
    print("=============================================")

    results_path = Path("eval_results.json")
    results_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nResults saved to {results_path.resolve()}")

    return 0 if passed == total else 2


if __name__ == "__main__":
    sys.exit(main())
