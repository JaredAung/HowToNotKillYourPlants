"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { getExplanation, getToken } from "@/lib/api";
import { ExplanationDisplay } from "@/app/components/ExplanationDisplay";
import { PlantCard, type PlantRec } from "@/app/components/PlantCard";
import { AddToGardenModal } from "@/app/components/AddToGardenModal";

const SEARCH_RESULTS_KEY = "searchExtractedProfile";
const SEARCH_PLANTS_KEY = "searchExtractedPlants";

type ProfileData = {
  username: string;
  profile: { name?: string; avatar_url?: string };
  location: { city?: string; state?: string; postal_code?: string; country?: string };
  environment: {
    light_level?: string;
    humidity_level?: string;
    temperature_pref?: { min_f?: number; max_f?: number };
  };
  climate?: string;
  safety: { has_kids?: boolean };
  constraints: { preferred_size?: string; hard_no?: string[] };
  preferences: {
    care_level?: string;
    care_preferences?: { watering_freq?: string; care_freq?: string };
  };
  physical_desc?: string;
  symbolism?: string;
};

function formatLabel(s: string): string {
  return s
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

export default function SearchResultsPage() {
  const [profile, setProfile] = useState<ProfileData | null>(null);
  const [plants, setPlants] = useState<PlantRec[]>([]);
  const [explanationOn, setExplanationOn] = useState(false);
  const [explanation, setExplanation] = useState<string | null>(null);
  const [explanationLoading, setExplanationLoading] = useState(false);
  const [addToGardenPlant, setAddToGardenPlant] = useState<PlantRec | null>(null);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const storedProfile = sessionStorage.getItem(SEARCH_RESULTS_KEY);
    const storedPlants = sessionStorage.getItem(SEARCH_PLANTS_KEY);
    if (storedProfile) {
      try {
        setProfile(JSON.parse(storedProfile) as ProfileData);
      } catch {
        setProfile(null);
      }
    }
    if (storedPlants) {
      try {
        setPlants(JSON.parse(storedPlants) as PlantRec[]);
      } catch {
        setPlants([]);
      }
    }
  }, []);

  useEffect(() => {
    if (!explanationOn) {
      setExplanation(null);
      return;
    }
    if (plants.length === 0) return;
    setExplanationLoading(true);
    const top5Ids = plants.slice(0, 5).map((p) => p.plant_id);
    getExplanation(top5Ids)
      .then((res) => setExplanation(res.explanation ?? ""))
      .catch(() => setExplanation(""))
      .finally(() => setExplanationLoading(false));
  }, [explanationOn, plants]);

  const isLoggedIn = !!getToken();

  if (!isLoggedIn) {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center px-4 bg-gradient-to-b from-sage-50 to-forest-50">
        <p className="text-forest-600 mb-4">Please sign in to view search results.</p>
        <Link href="/auth" className="text-forest-700 font-medium underline">
          Sign in
        </Link>
      </div>
    );
  }

  if (!profile) {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center px-4 bg-gradient-to-b from-sage-50 to-forest-50">
        <p className="text-forest-600 mb-4">No search results. Use Search to describe your plant preferences.</p>
        <Link href="/" className="text-forest-700 font-medium underline">
          Back to home
        </Link>
      </div>
    );
  }

  const { profile: p, location, environment, climate, safety, constraints, preferences, physical_desc, symbolism } = profile;
  const tempPref = environment?.temperature_pref;
  const carePref = preferences?.care_preferences;

  const sections: { title: string; items: [string, string | undefined | null][] }[] = [
    {
      title: "Profile",
      items: [
        ["Name", p?.name],
        ["Avatar URL", p?.avatar_url],
      ],
    },
    {
      title: "Location",
      items: [
        ["City", location?.city],
        ["State", location?.state],
        ["Postal code", location?.postal_code],
        ["Country", location?.country],
      ],
    },
    {
      title: "Environment",
      items: [
        ["Light level", environment?.light_level ? formatLabel(environment.light_level) : undefined],
        ["Humidity level", environment?.humidity_level ? formatLabel(environment.humidity_level) : undefined],
        ["Temp range", tempPref?.min_f != null && tempPref?.max_f != null ? `${tempPref.min_f}–${tempPref.max_f}°F` : undefined],
      ],
    },
    {
      title: "Other",
      items: [
        ["Climate", climate],
        ["Has kids", safety?.has_kids != null ? (safety.has_kids ? "Yes" : "No") : undefined],
        ["Preferred size", constraints?.preferred_size ? formatLabel(constraints.preferred_size) : undefined],
        ["Care level", preferences?.care_level ? formatLabel(preferences.care_level) : undefined],
        ["Watering freq", carePref?.watering_freq ? formatLabel(carePref.watering_freq) : undefined],
        ["Care freq", carePref?.care_freq ? formatLabel(carePref.care_freq) : undefined],
      ],
    },
    {
      title: "Plant description (from your text)",
      items: [
        ["Physical description", physical_desc],
        ["Symbolism / meaning", symbolism],
      ],
    },
  ];

  return (
    <div className="min-h-screen px-4 py-8 bg-gradient-to-b from-sage-50 to-forest-50">
      <div className="max-w-4xl mx-auto">
        <div className="mb-6">
          <h1 className="text-2xl font-semibold text-forest-800">Extracted profile</h1>
          <p className="text-sm text-forest-600 mt-1">
            Merged from your text and existing profile. Not saved to your account.
          </p>
        </div>

        <div className="rounded-xl border border-sage-200 bg-white shadow-leaf p-6 space-y-6">
          <div className="flex items-center gap-4 pb-4 border-b border-sage-200">
            {p?.avatar_url ? (
              <img
                src={p.avatar_url}
                alt={p?.name || profile.username}
                className="h-16 w-16 rounded-full object-cover"
              />
            ) : (
              <div className="h-16 w-16 rounded-full bg-sage-200 flex items-center justify-center text-2xl">
                🌱
              </div>
            )}
            <div>
              <p className="font-semibold text-forest-800">{p?.name || "—"}</p>
              <p className="text-sm text-forest-600">@{profile.username}</p>
            </div>
          </div>

          {sections.map(({ title, items }) => {
            const filtered = items.filter(([, v]) => v != null && v !== "");
            if (filtered.length === 0) return null;
            return (
              <div key={title}>
                <h2 className="text-sm font-semibold text-forest-700 mb-2">{title}</h2>
                <dl className="space-y-1.5">
                  {filtered.map(([label, value]) => (
                    <div key={label} className="flex justify-between gap-4">
                      <dt className="text-forest-600 text-sm">{formatLabel(label)}</dt>
                      <dd className="text-forest-800 text-sm text-right">{String(value)}</dd>
                    </div>
                  ))}
                </dl>
              </div>
            );
          })}
        </div>

        {plants.length > 0 && (
          <div className="mt-8">
            <div className="flex items-center justify-between gap-4 mb-4">
              <h2 className="text-lg font-semibold text-forest-800">Recommended plants</h2>
              <label className="flex items-center gap-2 cursor-pointer">
                <span className="text-sm text-forest-600">Explanation</span>
                <button
                  type="button"
                  role="switch"
                  aria-checked={explanationOn}
                  onClick={() => setExplanationOn((o) => !o)}
                  className={`relative inline-flex h-6 w-11 shrink-0 rounded-full border-2 border-transparent transition-colors focus:outline-none focus:ring-2 focus:ring-sage-400 focus:ring-offset-2 ${
                    explanationOn ? "bg-forest-600" : "bg-sage-200"
                  }`}
                >
                  <span
                    className={`pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition ${
                      explanationOn ? "translate-x-5" : "translate-x-1"
                    }`}
                  />
                </button>
              </label>
            </div>
            {explanationOn && (
              <div className="mb-6 rounded-xl border border-sage-200 bg-white shadow-leaf p-5">
                <h3 className="text-base font-semibold text-forest-800 mb-3 flex items-center gap-2">
                  <span className="text-lg">🌱</span>
                  Why these plants?
                </h3>
                <ExplanationDisplay explanation={explanation} loading={explanationLoading} />
              </div>
            )}
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
              {plants.map((p) => {
                const matchPct =
                  p.rerank_score != null
                    ? Math.round(p.rerank_score * 100)
                    : Math.round(((plants[0]?.score ?? 1) > 0 ? p.score / (plants[0]?.score ?? 1) : 0) * 100);
                return <PlantCard key={p.plant_id} p={p} matchPct={matchPct} onPick={() => setAddToGardenPlant(p)} />;
              })}
            </div>
          </div>
        )}

        <div className="mt-6 flex gap-3">
          <Link
            href="/"
            className="px-4 py-2 rounded-lg border border-sage-300 text-forest-700 text-sm font-medium hover:bg-sage-50"
          >
            Back to home
          </Link>
          <Link
            href="/profile"
            className="px-4 py-2 rounded-lg bg-forest-600 text-white text-sm font-medium hover:bg-forest-700"
          >
            View saved profile
          </Link>
        </div>
      </div>
      {addToGardenPlant && (
        <AddToGardenModal
          plant={addToGardenPlant}
          onClose={() => setAddToGardenPlant(null)}
        />
      )}
    </div>
  );
}
