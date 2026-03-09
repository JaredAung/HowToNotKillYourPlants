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
  water_req?: string;
  temp_min?: number;
  temp_max?: number;
};

export function PlantCard({
  p,
  matchPct,
  onAdd,
  onTalkToAgent,
  isJustAdded,
  showMatchPct = true,
}: {
  p: PlantRec;
  matchPct: number;
  onAdd?: (plant: PlantRec) => void;
  onTalkToAgent?: (plant: PlantRec) => void;
  isJustAdded?: boolean;
  showMatchPct?: boolean;
}) {
  const [imgError, setImgError] = useState(false);
  const tempStr =
    p.temp_min != null && p.temp_max != null
      ? `${Math.round(p.temp_min)}–${Math.round(p.temp_max)}°F`
      : null;
  const trunc = (s: string, n: number) => (s.length > n ? s.slice(0, n) + "…" : s);
  const showImg = p.img_url && !imgError;

  return (
    <Link href={`/plant/${p.plant_id}`} className="block group aspect-square [perspective:800px] cursor-pointer relative">
      <div className="absolute top-2 right-2 z-10 flex gap-1.5">
        {onTalkToAgent && (
          <button
            type="button"
            onClick={(e) => {
              e.preventDefault();
              e.stopPropagation();
              onTalkToAgent(p);
            }}
            className="h-8 w-8 rounded-full bg-white/90 shadow-leaf border border-sage-200 flex items-center justify-center text-forest-700 hover:bg-forest-600 hover:text-white hover:border-forest-600 transition-colors"
            title="Talk to agent"
            aria-label="Talk to agent"
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
            className={`h-8 w-8 rounded-full shadow-leaf border flex items-center justify-center transition-colors ${
              isJustAdded
                ? "bg-emerald-500 border-emerald-600 text-white"
                : "bg-white/90 border-sage-200 text-forest-700 hover:bg-forest-600 hover:text-white hover:border-forest-600"
            }`}
            title={isJustAdded ? "Added" : "Add to garden"}
            aria-label={isJustAdded ? "Added" : "Add to garden"}
          >
            <span className="text-sm font-medium">{isJustAdded ? "✓" : "+"}</span>
          </button>
        )}
      </div>
      <div className="relative w-full h-full transition-transform duration-500 [transform-style:preserve-3d] group-hover:[transform:rotateY(180deg)]">
        <div className="absolute inset-0 rounded-xl border border-sage-200 bg-white/80 backdrop-blur shadow-leaf overflow-hidden flex flex-col items-center justify-center p-4 [backface-visibility:hidden]">
          <div className="w-full aspect-square max-w-[120px] rounded-lg overflow-hidden bg-sage-100 flex items-center justify-center shrink-0">
            {showImg ? (
              <img
                src={p.img_url}
                alt={p.common_name ?? p.latin ?? ""}
                className="w-full h-full object-cover"
                referrerPolicy="no-referrer"
                onError={() => setImgError(true)}
              />
            ) : (
              <span className="text-3xl text-sage-400">🌱</span>
            )}
          </div>
          {p.latin && (
            <p className="font-bold text-forest-800 text-center mt-2 text-sm">{p.latin}</p>
          )}
          {p.common_name && (
            <p className="text-forest-600 text-center text-sm mt-0.5">{p.common_name}</p>
          )}
          {showMatchPct && (
            <p className="text-xs text-sage-500 mt-2">Match: {matchPct}%</p>
          )}
        </div>
        <div className="absolute inset-0 rounded-xl border border-sage-200 bg-forest-50/95 backdrop-blur shadow-leaf overflow-hidden flex flex-col items-center justify-center p-4 [backface-visibility:hidden] [transform:rotateY(180deg)]">
          <p className="text-sm font-semibold text-forest-800 mb-3">Care</p>
          <div className="text-left text-sm text-forest-700 space-y-1.5 w-full max-w-[90%]">
            {p.sunlight_type && <p>Sunlight: {p.sunlight_type}</p>}
            {p.humidity && <p>Humidity: {trunc(String(p.humidity), 30)}</p>}
            {p.care_level && <p>Care level: {p.care_level}</p>}
            {p.water_req && <p>Water: {trunc(String(p.water_req), 30)}</p>}
            {tempStr && <p>Temp: {tempStr}</p>}
          </div>
        </div>
      </div>
    </Link>
  );
}
