"use client";

import { parseExplanation } from "@/lib/parseExplanation";

export function ExplanationDisplay({
  explanation,
  loading,
}: {
  explanation: string | null;
  loading: boolean;
}) {
  if (loading) {
    return (
      <p className="text-sm text-sage-500 animate-pulse">Generating personalized explanation...</p>
    );
  }
  if (!explanation) return null;

  const parsed = parseExplanation(explanation);
  if (parsed.items.length === 0 && !parsed.intro) {
    return (
      <p className="text-sm text-forest-700 leading-relaxed">
        Your top recommendations are based on your preferences. Hover over each card to see care details.
      </p>
    );
  }

  return (
    <div className="space-y-4">
      {parsed.intro && (
        <div className="text-forest-700 text-sm leading-relaxed pl-1 border-l-2 border-sage-300 space-y-2">
          {parsed.intro
            .split(/(?<=[.!?])\s+/)
            .filter(Boolean)
            .map((sentence, j) => (
              <p key={j}>{sentence}</p>
            ))}
        </div>
      )}
      <div className="space-y-3">
        {parsed.items.map((item, i) => (
          <div
            key={i}
            className="rounded-xl border border-sage-200 bg-white/95 backdrop-blur shadow-leaf p-4 text-left hover:border-sage-300 transition-colors"
          >
            <p className="font-semibold text-forest-800 text-sm mb-1.5 flex items-center gap-2">
              <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-sage-200/80 text-[10px] font-bold text-forest-700">
                {i + 1}
              </span>
              {item.name}
              {item.latin && (
                <span className="font-normal text-forest-600 italic ml-0.5">({item.latin})</span>
              )}
            </p>
            <div className="text-forest-600 text-sm leading-relaxed pl-7 space-y-2">
              {item.text
                .split(/(?<=[.!?])\s+/)
                .filter(Boolean)
                .map((sentence, j) => (
                  <p key={j}>{sentence}</p>
                ))}
            </div>
          </div>
        ))}
      </div>
      {parsed.closing && (
        <div className="rounded-lg border border-sage-100 bg-sage-50/60 px-4 py-3 space-y-2">
          {parsed.closing
            .split(/(?<=[.!?])\s+/)
            .filter(Boolean)
            .map((sentence, j) => (
              <p key={j} className="text-forest-600 text-sm leading-relaxed">
                {sentence}
              </p>
            ))}
        </div>
      )}
    </div>
  );
}
