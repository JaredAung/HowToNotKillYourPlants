"use client";

import { useEffect, useState, useRef } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { addToGarden, getToken } from "@/lib/api";
import type { PlantRec } from "@/app/components/PlantCard";

import { ADD_TO_GARDEN_PLANT_KEY } from "@/lib/addToGarden";

type ChatMessage = {
  role: "bot" | "user";
  text: string;
};

export default function AddToGardenChatPage() {
  const router = useRouter();
  const [plant, setPlant] = useState<PlantRec | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputValue, setInputValue] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [step, setStep] = useState<"confirm" | "name" | "done">("confirm");
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });

  useEffect(() => {
    if (typeof window === "undefined") return;
    const stored = sessionStorage.getItem(ADD_TO_GARDEN_PLANT_KEY);
    if (stored) {
      try {
        setPlant(JSON.parse(stored) as PlantRec);
      } catch {
        router.push("/");
      }
    } else {
      router.push("/");
    }
  }, [router]);

  useEffect(() => {
    if (!plant) return;
    if (messages.length === 0) {
      const plantName = plant.common_name ?? plant.latin ?? `Plant #${plant.plant_id}`;
      setMessages([
        {
          role: "bot",
          text: `Are you sure you want to plant this? (${plantName})`,
        },
      ]);
    }
  }, [plant, messages.length]);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const isLoggedIn = !!getToken();

  const handleSend = async () => {
    const text = inputValue.trim();
    if (!text && step !== "confirm") return;

    if (step === "confirm") {
      const lower = text.toLowerCase();
      if (lower === "no" || lower === "n" || lower === "cancel") {
        sessionStorage.removeItem(ADD_TO_GARDEN_PLANT_KEY);
        router.push("/");
        return;
      }
      if (lower === "yes" || lower === "y") {
        setMessages((m) => [...m, { role: "user", text }, { role: "bot", text: "Would you like to name the plant? If you skip, we'll use the latin name." }]);
        setStep("name");
        setInputValue("");
        return;
      }
      setMessages((m) => [...m, { role: "user", text }, { role: "bot", text: "Please reply Yes or No." }]);
      setInputValue("");
      return;
    }

    if (step === "name") {
      const isSkip = !text || text.toLowerCase() === "skip" || text.toLowerCase() === "no";
      const customName = isSkip ? undefined : text;
      setMessages((m) => [...m, { role: "user", text: text || "Skip" }]);
      setInputValue("");
      setLoading(true);
      setError(null);
      try {
        await addToGarden(plant!.plant_id, customName);
        setMessages((m) => [...m, { role: "bot", text: "Your plant has been added to your garden! 🌱" }]);
        setStep("done");
        sessionStorage.removeItem(ADD_TO_GARDEN_PLANT_KEY);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to add");
        setMessages((m) => [...m, { role: "bot", text: `Something went wrong: ${err instanceof Error ? err.message : "Failed to add"}` }]);
      } finally {
        setLoading(false);
      }
    }
  };

  const handleSkipName = async () => {
    if (step !== "name" || !plant) return;
    setMessages((m) => [...m, { role: "user", text: "Skip" }]);
    setLoading(true);
    setError(null);
    try {
      await addToGarden(plant.plant_id, undefined);
      setMessages((m) => [...m, { role: "bot", text: "Your plant has been added to your garden! 🌱" }]);
      setStep("done");
      sessionStorage.removeItem(ADD_TO_GARDEN_PLANT_KEY);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to add");
      setMessages((m) => [...m, { role: "bot", text: `Something went wrong: ${err instanceof Error ? err.message : "Failed to add"}` }]);
    } finally {
      setLoading(false);
    }
  };

  if (!isLoggedIn) {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center px-4 bg-gradient-to-b from-sage-50 to-forest-50">
        <p className="text-forest-600 mb-4">Please sign in to add plants to your garden.</p>
        <Link href="/auth" className="text-forest-700 font-medium underline">
          Sign in
        </Link>
      </div>
    );
  }

  if (!plant) {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center px-4 bg-gradient-to-b from-sage-50 to-forest-50">
        <p className="text-forest-600">Loading...</p>
      </div>
    );
  }

  const defaultName = plant.latin ?? plant.common_name ?? `Plant #${plant.plant_id}`;

  return (
    <div className="min-h-screen flex flex-col bg-gradient-to-b from-sage-50 to-forest-50">
      <div className="border-b border-sage-200 bg-white/95 backdrop-blur px-4 py-3">
        <div className="max-w-2xl mx-auto flex items-center justify-between">
          <Link href="/" className="text-forest-600 hover:text-forest-800 text-sm">
            ← Back
          </Link>
          <h1 className="text-lg font-semibold text-forest-800">Add to garden</h1>
          <span className="w-12" />
        </div>
      </div>

      <div className="flex-1 overflow-y-auto px-4 py-6">
        <div className="max-w-2xl mx-auto space-y-4">
          {messages.map((msg, i) => (
            <div
              key={i}
              className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
            >
              <div
                className={`max-w-[85%] rounded-2xl px-4 py-3 ${
                  msg.role === "user"
                    ? "bg-forest-600 text-white"
                    : "bg-white border border-sage-200 shadow-sm text-forest-800"
                }`}
              >
                <p className="text-sm leading-relaxed">{msg.text}</p>
              </div>
            </div>
          ))}
          {error && (
            <p className="text-rose-600 text-sm text-center">{error}</p>
          )}
          <div ref={messagesEndRef} />
        </div>
      </div>

      {step === "done" ? (
        <div className="border-t border-sage-200 bg-white p-4">
          <div className="max-w-2xl mx-auto flex gap-3">
            <Link
              href="/"
              className="flex-1 py-3 rounded-lg border border-sage-300 text-forest-700 text-sm font-medium text-center hover:bg-sage-50"
            >
              Done
            </Link>
            <Link
              href="/garden"
              className="flex-1 py-3 rounded-lg bg-forest-600 text-white text-sm font-medium text-center hover:bg-forest-700"
            >
              View garden
            </Link>
          </div>
        </div>
      ) : (
        <div className="border-t border-sage-200 bg-white p-4">
          <div className="max-w-2xl mx-auto">
            {step === "name" && (
              <p className="text-sage-500 text-xs mb-2">Default: &quot;{defaultName}&quot;</p>
            )}
            <div className="flex gap-2">
              <input
                type="text"
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleSend()}
                placeholder={step === "confirm" ? "Yes or No" : "Enter a name or leave blank"}
                className="flex-1 px-4 py-3 rounded-xl border border-sage-200 text-forest-800 text-sm placeholder:text-sage-400 focus:outline-none focus:ring-2 focus:ring-sage-400 focus:border-transparent"
                disabled={loading}
              />
              {step === "name" && (
                <button
                  type="button"
                  onClick={handleSkipName}
                  disabled={loading}
                  className="px-4 py-3 rounded-xl border border-sage-300 text-forest-700 text-sm font-medium hover:bg-sage-50 disabled:opacity-50"
                >
                  Skip
                </button>
              )}
              <button
                type="button"
                onClick={handleSend}
                disabled={loading || (step === "confirm" && !inputValue.trim())}
                className="px-4 py-3 rounded-xl bg-forest-600 text-white text-sm font-medium hover:bg-forest-700 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {loading ? "..." : "Send"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
