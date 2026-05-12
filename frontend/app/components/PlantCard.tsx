"use client";

import { useState } from "react";
import Link from "next/link";

export type PlantRec = {
  plant_id: number;
  score: number;
  rerank_score?: number;
  img_url?: string;
  latin?: string;
  common_name?: string;
  sunlight_type?: string;
  humidity?: string;
  care_level?: string;
  /** Catalog difficulty 0–100 (higher = harder); drives tier when care_level is missing. */
  difficulty_score?: number;
  water_req?: string;
  temp_min?: number;
  temp_max?: number;
};

type DifficultyTier = "easy" | "medium" | "hard";

function tierFromPlant(p: PlantRec): DifficultyTier | null {
  const raw = p.care_level?.trim().toLowerCase();
  if (raw === "easy" || raw === "beginner") return "easy";
  if (raw === "medium" || raw === "moderate") return "medium";
  if (raw === "hard" || raw === "difficult") return "hard";
  const s = p.difficulty_score;
  if (s != null && Number.isFinite(s)) {
    if (s < 40) return "easy";
    if (s < 60) return "medium";
    return "hard";
  }
  return null;
}

const TIER_LABEL: Record<DifficultyTier, string> = {
  easy: "Easy",
  medium: "Medium",
  hard: "Hard",
};

const TIER_CLASS: Record<DifficultyTier, string> = {
  easy: "bg-emerald-100 text-emerald-900 ring-emerald-300/80",
  medium: "bg-amber-100 text-amber-950 ring-amber-300/80",
  hard: "bg-rose-100 text-rose-900 ring-rose-300/80",
};

/** Text color for Sprouts / difficulty — matches pill label colors (care tier). */
const TIER_SCORE_TEXT: Record<DifficultyTier, string> = {
  easy: "text-emerald-900",
  medium: "text-amber-950",
  hard: "text-rose-900",
};

/** Display text for difficulty: exact API score (no rounding); else inferred from care_level. */
function formatDifficultyScoreText(p: PlantRec): string | null {
  if (p.difficulty_score != null && Number.isFinite(p.difficulty_score)) {
    return String(p.difficulty_score);
  }
  const c = p.care_level?.trim().toLowerCase();
  if (c === "easy" || c === "beginner" || c === "low") return "25";
  if (c === "medium" || c === "moderate") return "50";
  if (c === "hard" || c === "difficult" || c === "high") return "75";
  return null;
}

export function PlantCard({
  p,
  onAdd,
  onTalkToAgent,
  isJustAdded,
}: {
  p: PlantRec;
  onAdd?: (plant: PlantRec) => void;
  onTalkToAgent?: (plant: PlantRec) => void;
  isJustAdded?: boolean;
}) {
  const [imgError, setImgError] = useState(false);
  const tempStr =
    p.temp_min != null && p.temp_max != null
      ? `${Math.round(p.temp_min)}–${Math.round(p.temp_max)}°F`
      : null;
  const trunc = (s: string, n: number) => (s.length > n ? s.slice(0, n) + "…" : s);
  const showImg = p.img_url && !imgError;
  const displayName = p.latin || p.common_name || `Plant #${p.plant_id}`;
  const tier = tierFromPlant(p);
  const difficultyText = formatDifficultyScoreText(p);
  const careLabel = p.care_level?.trim();
  const showCareSubtitle =
    !!careLabel &&
    (!tier || careLabel.toLowerCase() !== TIER_LABEL[tier].toLowerCase());

  const hasTopCareStrip = tier != null || showCareSubtitle;

  return (
    <Link
      href={`/plant/${p.plant_id}`}
      className="block group relative w-full min-h-[280px] [perspective:800px] cursor-pointer"
    >
      <div className="absolute top-2 left-2 right-2 z-10 flex items-center justify-between gap-2">
        <div
          className={`flex min-w-0 flex-1 flex-wrap items-center gap-x-1.5 gap-y-1 ${hasTopCareStrip ? "" : "min-h-[32px]"}`}
        >
          {tier && (
            <span
              className={`inline-flex shrink-0 items-center rounded-full px-2 py-0.5 text-[11px] font-semibold ring-1 ${TIER_CLASS[tier]}`}
            >
              {TIER_LABEL[tier]}
            </span>
          )}
          {showCareSubtitle && (
            <span className="max-w-[12rem] truncate text-xs font-medium leading-snug text-forest-700 sm:text-sm">
              {careLabel}
            </span>
          )}
        </div>
        <div className="flex shrink-0 gap-1.5">
          {onTalkToAgent && (
            <button
              type="button"
              onClick={(e) => {
                e.preventDefault();
                e.stopPropagation();
                onTalkToAgent(p);
              }}
              className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full border border-sage-200 bg-white/90 text-forest-700 shadow-leaf hover:border-forest-600 hover:bg-forest-600 hover:text-white"
              title="Discuss with agent"
              aria-label="Discuss with agent"
            >
              <span className="text-sm font-medium">?</span>
            </button>
          )}
          {onAdd && (
            <button
              type="button"
              onClick={(e) => {
                e.preventDefault();
                e.stopPropagation();
                onAdd(p);
              }}
              className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-full border shadow-leaf ${
                isJustAdded
                  ? "border-emerald-600 bg-emerald-500 text-white"
                  : "border-sage-200 bg-white/90 text-forest-700 hover:border-forest-600 hover:bg-forest-600 hover:text-white"
              }`}
              title={isJustAdded ? "Added" : "Add to garden"}
              aria-label={isJustAdded ? "Added" : "Add to garden"}
            >
              <span className="text-sm font-medium">{isJustAdded ? "✓" : "+"}</span>
            </button>
          )}
        </div>
      </div>

      <div className="relative min-h-[280px] w-full transition-transform duration-500 [transform-style:preserve-3d] group-hover:[transform:rotateY(180deg)]">
        <div className="absolute inset-0 flex flex-col items-center justify-center overflow-hidden rounded-xl border border-sage-200 bg-white/80 px-3 pb-9 pt-11 shadow-leaf backdrop-blur [backface-visibility:hidden]">
          <div className="flex w-full max-w-[200px] flex-col items-center justify-center gap-3">
            <div className="flex w-full justify-center">
              <div className="aspect-square w-[min(100%,120px)] shrink-0 overflow-hidden rounded-lg bg-sage-100 flex items-center justify-center">
                {showImg ? (
                  <img
                    src={p.img_url}
                    alt={displayName}
                    className="h-full w-full object-cover"
                    referrerPolicy="no-referrer"
                    onError={() => setImgError(true)}
                  />
                ) : (
                  <span className="text-3xl text-sage-400">🌱</span>
                )}
              </div>
            </div>
            <div className="flex w-full flex-col items-center gap-0.5 px-0.5">
              {p.latin ? (
                <p className="text-center text-sm font-bold text-forest-800">{p.latin}</p>
              ) : p.common_name ? (
                <p className="text-center text-sm font-bold text-forest-800">{p.common_name}</p>
              ) : (
                <p className="text-center text-sm font-bold text-forest-800">{displayName}</p>
              )}
              {p.latin && p.common_name && (
                <p className="text-center text-sm text-forest-600">{p.common_name}</p>
              )}
            </div>
          </div>
          {difficultyText != null && (
            <div
              className={`pointer-events-none absolute bottom-2 right-2 z-[5] flex items-baseline gap-0.5 text-sm font-bold tabular-nums ${
                tier ? TIER_SCORE_TEXT[tier] : "text-forest-800"
              }`}
              title="Sprouts (difficulty points, 0–100)"
            >
              <span className="select-none text-base leading-none" aria-hidden>
                🌱
              </span>
              <span>{difficultyText}</span>
            </div>
          )}
        </div>
        <div className="absolute inset-0 flex flex-col items-center justify-center overflow-hidden rounded-xl border border-sage-200 bg-forest-50/95 p-4 shadow-leaf backdrop-blur [backface-visibility:hidden] [transform:rotateY(180deg)]">
          <p className="mb-3 text-sm font-semibold text-forest-800">Care</p>
          <div className="w-full max-w-[90%] space-y-1.5 text-left text-sm text-forest-700">
            {p.sunlight_type && <p>Sunlight: {p.sunlight_type}</p>}
            {p.humidity && <p>Humidity: {trunc(String(p.humidity), 30)}</p>}
            {p.care_level && <p>Care level: {p.care_level}</p>}
            {difficultyText != null && (
              <p>
                Difficulty:{" "}
                <span className={`font-semibold tabular-nums ${tier ? TIER_SCORE_TEXT[tier] : "text-forest-800"}`}>
                  {difficultyText}
                </span>
              </p>
            )}
            {p.water_req && <p>Water: {trunc(String(p.water_req), 30)}</p>}
            {tempStr && <p>Temp: {tempStr}</p>}
          </div>
        </div>
      </div>
    </Link>
  );
}
