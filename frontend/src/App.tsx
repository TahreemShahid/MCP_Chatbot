import { FormEvent, useRef, useState } from "react";
import {
  deleteSession,
  Message,
  streamChat,
  uploadPdf,
} from "./api";
import { parseMarkdown } from "./markdown";

const GREETING: Message = {
  id: "greeting",
  role: "assistant",
  content:
    "Hello — I'm your **MCP-integrated document assistant**. Upload a PDF, then ask questions. " +
    "Answers are powered by Groq; document search runs through MCP.",
};

function uid(): string {
  return crypto.randomUUID();
}

export default function App() {
  const [messages, setMessages] = useState<Message[]>([GREETING]);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [docName, setDocName] = useState<string | null>(null);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [uploading, setUploading] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  async function handleUpload(file: File) {
    setUploading(true);
    try {
      const result = await uploadPdf(file);
      setSessionId(result.session_id);
      setDocName(file.name);
      setMessages((prev) => [
        ...prev,
        {
          id: uid(),
          role: "assistant",
          content: `**${file.name}** is indexed via MCP. What would you like to know?`,
        },
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          id: uid(),
          role: "assistant",
          content: `Upload failed: ${err instanceof Error ? err.message : String(err)}`,
        },
      ]);
    } finally {
      setUploading(false);
    }
  }

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    const text = input.trim();
    if (!text || !sessionId || busy) return;

    const userMessage: Message = { id: uid(), role: "user", content: text };
    const assistantId = uid();
    setMessages((prev) => [...prev, userMessage, { id: assistantId, role: "assistant", content: "" }]);
    setInput("");
    setBusy(true);

    try {
      await streamChat(sessionId, text, (token) => {
        setMessages((prev) =>
          prev.map((m) => (m.id === assistantId ? { ...m, content: token } : m)),
        );
      });
    } catch (err) {
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantId
            ? {
                ...m,
                content: `Error: ${err instanceof Error ? err.message : String(err)}`,
              }
            : m,
        ),
      );
    } finally {
      setBusy(false);
    }
  }

  async function resetChat() {
    if (sessionId) {
      try {
        await deleteSession(sessionId);
      } catch {
        /* ignore cleanup errors */
      }
    }
    setSessionId(null);
    setDocName(null);
    setMessages([GREETING]);
    setInput("");
    if (fileRef.current) fileRef.current.value = "";
  }

  return (
    <div className="app">
      <header className="header">
        <div>
          <h1>MCP Document Assistant</h1>
          <p>LangChain agent · FastAPI · MCP tools · FAISS retrieval</p>
        </div>
        <button type="button" className="btn-primary" onClick={resetChat}>
          New conversation
        </button>
      </header>

      <section className="status">
        {docName ? (
          <span className="status-pill active">Active: {docName}</span>
        ) : (
          <span className="status-pill">Upload a PDF to begin</span>
        )}
      </section>

      <section className="upload-panel">
        <label className="upload-label">
          <input
            ref={fileRef}
            type="file"
            accept="application/pdf"
            disabled={uploading}
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) void handleUpload(file);
            }}
          />
          <span>
            {uploading
              ? "Indexing PDF (large files may take a few minutes)…"
              : "Attach PDF (max 50 MB)"}
          </span>
        </label>
      </section>

      <main className="thread">
        {messages.map((m) => (
          <div key={m.id} className={`bubble-row ${m.role}`}>
            {m.role === "assistant" && <div className="avatar bot">✦</div>}
            <div className={`bubble ${m.role}`}>
              {m.content
                ? parseMarkdown(m.content)
                : busy && m.role === "assistant"
                  ? "…"
                  : ""}
            </div>
            {m.role === "user" && <div className="avatar user">You</div>}
          </div>
        ))}
      </main>

      <form className="composer" onSubmit={handleSubmit}>
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder={sessionId ? "Ask about your document…" : "Upload a PDF first…"}
          disabled={!sessionId || busy}
        />
        <button
          type="submit"
          className="btn-primary"
          disabled={!sessionId || busy || !input.trim()}
        >
          Send
        </button>
      </form>
    </div>
  );
}
