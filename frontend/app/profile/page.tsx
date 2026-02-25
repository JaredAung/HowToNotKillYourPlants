"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { getProfile, getToken } from "@/lib/api";

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
};

function formatLabel(s: string): string {
  return s
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

export default function ProfilePage() {
  const [loading, setLoading] = useState(true);
  const [profile, setProfile] = useState<ProfileData | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!getToken()) {
      setLoading(false);
      return;
    }
    getProfile()
      .then((data) => setProfile(data as ProfileData))
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

  const { profile: p, location, environment, climate, safety, constraints, preferences } = profile!;
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
  ];

  return (
    <div className="min-h-screen px-4 py-8 bg-gradient-to-b from-sage-50 to-forest-50">
      <div className="max-w-2xl mx-auto">
        <div className="flex items-center justify-between mb-6">
          <h1 className="text-2xl font-semibold text-forest-800">Profile</h1>
          <Link
            href="/onboarding"
            className="px-4 py-2 rounded-lg bg-forest-600 text-white text-sm font-medium hover:bg-forest-700 transition-colors"
          >
            Edit profile
          </Link>
        </div>

        <div className="rounded-xl border border-sage-200 bg-white shadow-leaf p-6 space-y-6">
          <div className="flex items-center gap-4 pb-4 border-b border-sage-200">
            {p?.avatar_url ? (
              <img
                src={p.avatar_url}
                alt={p?.name || profile!.username}
                className="h-16 w-16 rounded-full object-cover"
              />
            ) : (
              <div className="h-16 w-16 rounded-full bg-sage-200 flex items-center justify-center text-2xl">
                🌱
              </div>
            )}
            <div>
              <p className="font-semibold text-forest-800">{p?.name || "—"}</p>
              <p className="text-sm text-forest-600">@{profile!.username}</p>
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
      </div>
    </div>
  );
}
