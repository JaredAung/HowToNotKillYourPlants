"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { getProfile, getToken } from "@/lib/api";

/** Matches GET /profile (two-tower fields only). */
type TowerProfile = {
  username: string;
  profile?: { name?: string | null };
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
};

function formatLabel(s: string): string {
  return s
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

export default function ProfilePage() {
  const [loading, setLoading] = useState(true);
  const [profile, setProfile] = useState<TowerProfile | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!getToken()) {
      setLoading(false);
      return;
    }
    getProfile()
      .then((data) => setProfile(data as TowerProfile))
      .catch((err) => setError(err instanceof Error ? err.message : "Failed to load"))
      .finally(() => setLoading(false));
  }, []);

  const isLoggedIn = !!getToken();

  if (!isLoggedIn) {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center px-4 bg-gradient-to-b from-sage-50 to-forest-50">
        <p className="text-forest-600 mb-4">Please sign in to view your profile.</p>
        <Link href="/auth" className="text-forest-700 font-medium underline">
          Sign in
        </Link>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center px-4 bg-gradient-to-b from-sage-50 to-forest-50">
        <p className="text-forest-600">Loading...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center px-4 bg-gradient-to-b from-sage-50 to-forest-50">
        <p className="text-rose-600 mb-4">{error}</p>
        <Link href="/" className="text-forest-700 font-medium underline">
          Back to home
        </Link>
      </div>
    );
  }

  const environment = profile!.environment;
  const constraints = profile!.constraints;
  const preferences = profile!.preferences;
  const tempPref = environment?.temperature_pref;
  const carePref = preferences?.care_preferences;

  const sections: { title: string; items: [string, string | undefined | null][] }[] = [
    {
      title: "Climate & zones",
      items: [
        ["Climate", profile!.climate ?? undefined],
        [
          "USDA zones",
          profile!.usda_zone_min != null && profile!.usda_zone_max != null
            ? `${profile!.usda_zone_min}–${profile!.usda_zone_max}`
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
  ];

  return (
    <div className="min-h-screen px-4 py-8 bg-gradient-to-b from-sage-50 to-forest-50">
      <div className="max-w-2xl mx-auto">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h1 className="text-2xl font-semibold text-forest-800">Recommendation profile</h1>
            <p className="text-sm text-forest-600 mt-1">Fields used for the same user tower as model training.</p>
          </div>
          <Link
            href="/onboarding"
            className="px-4 py-2 rounded-lg bg-forest-600 text-white text-sm font-medium hover:bg-forest-700 transition-colors"
          >
            Edit
          </Link>
        </div>

        <div className="rounded-xl border border-sage-200 bg-white shadow-leaf p-6 space-y-6">
          <div className="pb-4 border-b border-sage-200">
            <p className="text-forest-800 font-medium">
              {profile!.profile?.name?.trim() ? profile!.profile.name.trim() : "—"}
            </p>
            <p className="text-sm text-forest-600">@{profile!.username}</p>
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
      </div>
    </div>
  );
}
