"use client";

import { useEffect, useState, useRef } from "react";
import Link from "next/link";
import { getChatContext, clearChatContext } from "@/lib/chatContext";
import { invokeChat, getToken, type ChatMessage } from "@/lib/api";
import type { PlantRec } from "@/app/components/PlantCard";
import { MarkdownContent } from "@/app/components/MarkdownContent";

export default function ChatPage() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputValue, setInputValue] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedPlant, setSelectedPlant] = useState<Record<string, unknown> | null>(null);
  const [recommendedPlants, setRecommendedPlants] = useState<Record<string, unknown>[]>([]);
  const [recsOpen, setRecsOpen] = useState(true);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });

  useEffect(() => {
    const { selectedPlant: sp, recommendedPlants: recs } = getChatContext();
    if (sp) setSelectedPlant(sp as Record<string, unknown>);
    if (recs?.length) setRecommendedPlants(recs as Record<string, unknown>[]);
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const isLoggedIn = !!getToken();

  const handleSend = async () => {
    const text = inputValue.trim();
    if (!text || loading) return;

    const userMsg: ChatMessage = { role: "user", content: text };
    setMessages((m) => [...m, userMsg]);
    setInputValue("");
    setLoading(true);
    setError(null);

    try {
      const allMessages: ChatMessage[] = [...messages, userMsg];
      const res = await invokeChat({
        messages: allMessages,
        selectedPlant: selectedPlant ?? undefined,
        recommendedPlants: recommendedPlants.length ? recommendedPlants : undefined,
      });

      const outMessages = res?.messages ?? [];
      const normalized: ChatMessage[] = Array.isArray(outMessages)
        ? outMessages.map((m: { role?: string; content?: string }) => ({
            role: (m?.role === "assistant" ? "assistant" : "user") as "user" | "assistant",
            content: String(m?.content ?? ""),
          }))
        : allMessages;
      setMessages(normalized.length ? normalized : [...allMessages]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Chat failed");
    } finally {
      setLoading(false);
    }
  };

  if (!isLoggedIn) {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center px-4 bg-gradient-to-b from-sage-50 to-forest-50">
        <p className="text-forest-600 mb-4">Please sign in to chat.</p>
        <Link href="/auth" className="text-forest-700 font-medium underline">
          Sign in
        </Link>
      </div>
    );
  }

  const plants = recommendedPlants as PlantRec[];

  return (
    <div className="min-h-screen flex bg-gradient-to-b from-sage-50 to-forest-50">
      {/* Left: dropdown recommendations */}
      {plants.length > 0 && (
        <aside className="w-56 shrink-0 border-r border-sage-200 bg-white/95 flex flex-col">
          <button
            type="button"
            onClick={() => setRecsOpen((o) => !o)}
            className="flex items-center justify-between px-4 py-3 border-b border-sage-200 text-left hover:bg-sage-50 transition-colors"
          >
            <span className="text-sm font-medium text-forest-800">Recommendations</span>
            <span className="text-sage-500 text-xs">{recsOpen ? "▼" : "▶"}</span>
          </button>
          {recsOpen && (
            <div className="flex-1 overflow-y-auto p-2 space-y-2">
              {plants.slice(0, 5).map((p) => {
                const isSelected = selectedPlant && (p as Record<string, unknown>).plant_id === (selectedPlant as Record<string, unknown>).plant_id;
                return (
                  <div
                    key={p.plant_id}
                    className={`w-full flex items-center gap-3 p-2 rounded-lg border bg-white ${
                      isSelected ? "border-forest-500 ring-1 ring-forest-500" : "border-sage-200"
                    }`}
                  >
                    <div className="w-12 h-12 shrink-0 rounded-lg overflow-hidden bg-sage-100 flex items-center justify-center">
                      {p.img_url ? (
                        <img
                          src={p.img_url}
                          alt={p.common_name ?? p.latin ?? ""}
                          className="w-full h-full object-cover"
                          referrerPolicy="no-referrer"
                        />
                      ) : (
                        <span className="text-xl text-sage-400">🌱</span>
                      )}
                    </div>
                    <span className="text-sm font-medium text-forest-800 truncate">
                      {p.common_name ?? p.latin ?? `Plant #${p.plant_id}`}
                    </span>
                  </div>
                );
              })}
            </div>
          )}
        </aside>
      )}

      <div className="flex-1 flex flex-col min-w-0">
        <div className="border-b border-sage-200 bg-white/95 backdrop-blur px-4 py-3">
          <div className="max-w-2xl mx-auto flex items-center justify-between">
            <Link href="/" className="text-forest-600 hover:text-forest-800 text-sm">
              ← Back
            </Link>
            <h1 className="text-lg font-semibold text-forest-800">Plant chat</h1>
            <button
              type="button"
              onClick={clearChatContext}
              className="text-sage-500 hover:text-forest-600 text-xs"
            >
              Clear context
            </button>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto px-4 py-6">
          <div className="max-w-2xl mx-auto space-y-4">
          {messages.length === 0 && (
            <div className="rounded-xl border border-sage-200 bg-white p-4 text-forest-600 text-sm">
              {selectedPlant ? (
                <p>
                  Ask about <strong>{selectedPlant.common_name ?? selectedPlant.latin ?? "this plant"}</strong> or explore others.
                </p>
              ) : recommendedPlants.length > 0 ? (
                <p>Ask about your recommendations or any plant you&apos;re curious about.</p>
              ) : (
                <p>Ask about any plant—care, light, watering, or what might suit your space.</p>
              )}
            </div>
          )}
          {messages.map((msg, i) => (
            <div
              key={i}
              className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
            >
              <div
                className={`max-w-[90%] sm:max-w-[85%] rounded-2xl px-5 py-4 ${
                  msg.role === "user"
                    ? "bg-forest-600 text-white shadow-md"
                    : "bg-white border border-sage-200 shadow-sm text-forest-800"
                }`}
              >
                <div className="text-sm leading-relaxed [&_.markdown-content_p]:my-1.5 [&_.markdown-content_ul]:my-2 [&_.markdown-content_strong]:font-semibold">
                  <MarkdownContent content={msg.content} />
                </div>
              </div>
            </div>
          ))}
          {loading && (
            <div className="flex justify-start">
              <div className="rounded-2xl px-4 py-3 bg-white border border-sage-200 shadow-sm text-forest-600 text-sm">
                …
              </div>
            </div>
          )}
          {error && (
            <p className="text-rose-600 text-sm text-center">{error}</p>
          )}
          <div ref={messagesEndRef} />
          </div>
        </div>

        <div className="border-t border-sage-200 bg-white p-4">
        <div className="max-w-2xl mx-auto flex gap-2">
          <input
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleSend()}
            placeholder="Ask about plants..."
            className="flex-1 px-4 py-3 rounded-xl border border-sage-200 text-forest-800 text-sm placeholder:text-sage-400 focus:outline-none focus:ring-2 focus:ring-sage-400 focus:border-transparent"
            disabled={loading}
          />
          <button
            type="button"
            onClick={handleSend}
            disabled={loading || !inputValue.trim()}
            className="px-4 py-3 rounded-xl bg-forest-600 text-white text-sm font-medium hover:bg-forest-700 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Send
          </button>
        </div>
      </div>
      </div>
    </div>
  );
}
