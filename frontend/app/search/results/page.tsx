"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { getExplanation, getToken } from "@/lib/api";
import { useRouter } from "next/navigation";
import { ExplanationDisplay } from "@/app/components/ExplanationDisplay";
import { PlantCard, type PlantRec } from "@/app/components/PlantCard";
import { setChatContext } from "@/lib/chatContext";
import { addDirectlyToGarden } from "@/lib/addToGarden";

const SEARCH_RESULTS_KEY = "searchExtractedProfile";
const SEARCH_PLANTS_KEY = "searchExtractedPlants";
const SEARCH_SEMANTIC_PLANTS_KEY = "searchSemanticPlants";

type ProfileData = {
  username: string;
  climate?: string | null;
  usda_zone_min?: number | null;
  usda_zone_max?: number | null;
  environment?: {
    light_level?: string | null;
    soil_preference?: string | null;
    temperature_pref?: { min_f?: number | null; max_f?: number | null };
  };
  constraints?: { preferred_size?: string | null };
  preferences?: {
    care_level?: string | null;
    growth_pref?: string | null;
    care_preferences?: { watering_freq?: string | null };
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
  const router = useRouter();
  const [profile, setProfile] = useState<ProfileData | null>(null);
  const [plants, setPlants] = useState<PlantRec[]>([]);
  const [semanticPlants, setSemanticPlants] = useState<PlantRec[]>([]);
  const [explanationOn, setExplanationOn] = useState(false);
  const [explanation, setExplanation] = useState<string | null>(null);
  const [explanationLoading, setExplanationLoading] = useState(false);
  const [addError, setAddError] = useState<string | null>(null);
  const [addSuccessPlantId, setAddSuccessPlantId] = useState<number | null>(null);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const storedProfile = sessionStorage.getItem(SEARCH_RESULTS_KEY);
    const storedPlants = sessionStorage.getItem(SEARCH_PLANTS_KEY);
    const storedSemantic = sessionStorage.getItem(SEARCH_SEMANTIC_PLANTS_KEY);
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
    if (storedSemantic) {
      try {
        setSemanticPlants(JSON.parse(storedSemantic) as PlantRec[]);
      } catch {
        setSemanticPlants([]);
      }
    }
  }, []);

  useEffect(() => {
    if (addSuccessPlantId == null) return;
    const t = setTimeout(() => setAddSuccessPlantId(null), 2000);
    return () => clearTimeout(t);
  }, [addSuccessPlantId]);

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

  const { environment, climate, constraints, preferences, physical_desc, symbolism } = profile;
  const tempPref = environment?.temperature_pref;
  const carePref = preferences?.care_preferences;

  const sections: { title: string; items: [string, string | undefined | null][] }[] = [
    {
      title: "Climate & zones",
      items: [
        ["Climate", climate ?? undefined],
        [
          "USDA zones",
          profile.usda_zone_min != null && profile.usda_zone_max != null
            ? `${profile.usda_zone_min}–${profile.usda_zone_max}`
            : undefined,
        ],
      ],
    },
    {
      title: "Environment",
      items: [
        ["Light level", environment?.light_level ? formatLabel(environment.light_level) : undefined],
        ["Soil preference", environment?.soil_preference ? formatLabel(environment.soil_preference) : undefined],
        ["Temp range °F", tempPref?.min_f != null && tempPref?.max_f != null ? `${tempPref.min_f}–${tempPref.max_f}` : undefined],
      ],
    },
    {
      title: "Preferences",
      items: [
        ["Preferred size", constraints?.preferred_size ? formatLabel(constraints.preferred_size) : undefined],
        ["Care level", preferences?.care_level ? formatLabel(preferences.care_level) : undefined],
        ["Growth preference", preferences?.growth_pref ? formatLabel(preferences.growth_pref) : undefined],
        ["Watering", carePref?.watering_freq ? formatLabel(carePref.watering_freq) : undefined],
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
          <p className="text-forest-800 font-medium pb-4 border-b border-sage-200">@{profile.username}</p>

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

        {addError && (
          <p className="text-sm text-rose-600 mt-4">{addError}</p>
        )}

        {plants.length > 0 && (
          <div className="mt-8">
            <div className="flex items-center justify-between gap-4 mb-4">
              <h2 className="text-lg font-semibold text-forest-800">For your space</h2>
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
            <p className="text-sm text-forest-600 mb-4">
              Personalized by your profile and environment.
            </p>
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
                return (
                  <PlantCard
                    key={p.plant_id}
                    p={p}
                    isJustAdded={addSuccessPlantId === p.plant_id}
                    onAdd={async (plant) => {
                      try {
                        setAddError(null);
                        await addDirectlyToGarden(plant);
                        setAddSuccessPlantId(plant.plant_id);
                      } catch (err) {
                        setAddError(err instanceof Error ? err.message : "Failed to add");
                      }
                    }}
                    onTalkToAgent={(plant) => {
                      setChatContext(plant, plants.slice(0, 5));
                      router.push("/chat");
                    }}
                  />
                );
              })}
            </div>
          </div>
        )}

        {semanticPlants.length > 0 && (
          <div className="mt-10">
            <h2 className="text-lg font-semibold text-forest-800 mb-2">Matches your description</h2>
            <p className="text-sm text-forest-600 mb-4">
              By similarity to what you wrote.
            </p>
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
              {semanticPlants.map((p) => (
                  <PlantCard
                    key={p.plant_id}
                    p={p}
                    isJustAdded={addSuccessPlantId === p.plant_id}
                    onAdd={async (plant) => {
                      try {
                        setAddError(null);
                        await addDirectlyToGarden(plant);
                        setAddSuccessPlantId(plant.plant_id);
                      } catch (err) {
                        setAddError(err instanceof Error ? err.message : "Failed to add");
                      }
                    }}
                    onTalkToAgent={(plant) => {
                      setChatContext(plant, semanticPlants.slice(0, 5));
                      router.push("/chat");
                    }}
                  />
                ))}
            </div>
          </div>
        )}

        {plants.length === 0 && semanticPlants.length === 0 && (
          <p className="text-forest-600 text-sm mt-8">No plants found. Try a different search.</p>
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
    </div>
  );
}
