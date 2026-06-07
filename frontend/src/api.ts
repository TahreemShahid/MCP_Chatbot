const API_BASE = import.meta.env.VITE_API_URL ?? "";

export type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
};

export async function uploadPdf(file: File): Promise<{ session_id: string; message: string }> {
  const form = new FormData();
  form.append("file", file);
  const response = await fetch(`${API_BASE}/upload`, { method: "POST", body: form });
  if (!response.ok) {
    const body = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(body.detail ?? "Upload failed");
  }
  return response.json();
}

export async function chat(sessionId: string, message: string): Promise<string> {
  const response = await fetch(`${API_BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, message }),
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(body.detail ?? "Chat failed");
  }
  const data = await response.json();
  return data.response ?? "";
}

export async function streamChat(
  sessionId: string,
  message: string,
  onToken: (token: string) => void,
): Promise<void> {
  const response = await fetch(`${API_BASE}/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, message }),
  });

  if (!response.ok || !response.body) {
    const body = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(body.detail ?? "Stream failed");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";

    for (const line of lines) {
      if (!line.startsWith("data: ")) continue;
      const payload = JSON.parse(line.slice(6)) as {
        token?: string;
        error?: string;
        done?: boolean;
      };
      if (payload.error) {
        const msg = payload.error.includes("quota")
          ? payload.error
          : payload.error.length > 280
            ? "Request failed. If this is a Gemini 429 error, switch GEMINI_CHAT_MODEL in .env and restart the API."
            : payload.error;
        throw new Error(msg);
      }
      if (payload.token) onToken(payload.token);
    }
  }
}

export async function deleteSession(sessionId: string): Promise<void> {
  await fetch(`${API_BASE}/session/${sessionId}`, { method: "DELETE" });
}
