"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { getGarden, getToken } from "@/lib/api";

type GardenPlant = {
  plant_id: number;
  custom_name: string;
  added_at?: string;
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

export default function GardenPage() {
  const [plants, setPlants] = useState<GardenPlant[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const isLoggedIn = !!getToken();

  useEffect(() => {
    if (!isLoggedIn) {
      setLoading(false);
      return;
    }
    getGarden()
      .then((data: { plants?: GardenPlant[] }) => setPlants(data?.plants ?? []))
      .catch((err) => setError(err instanceof Error ? err.message : "Failed to load"))
      .finally(() => setLoading(false));
  }, [isLoggedIn]);

  if (!isLoggedIn) {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center px-4 bg-gradient-to-b from-sage-50 to-forest-50">
        <p className="text-forest-600 mb-4">Please sign in to view your garden.</p>
        <Link href="/auth" className="text-forest-700 font-medium underline">
          Sign in
        </Link>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center px-4 bg-gradient-to-b from-sage-50 to-forest-50">
        <p className="text-forest-600">Loading your garden...</p>
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

  return (
    <div className="min-h-screen px-4 py-8 bg-gradient-to-b from-sage-50 to-forest-50">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-2xl font-semibold text-forest-800 mb-2">Your Garden</h1>
        <p className="text-forest-600 text-sm mb-6">
          Plants you&apos;ve added from recommendations and search.
        </p>

        {plants.length === 0 ? (
          <div className="rounded-xl border border-sage-200 bg-white shadow-leaf p-8 text-center">
            <span className="text-4xl text-sage-400">🌱</span>
            <p className="text-forest-600 mt-3">Your garden is empty.</p>
            <p className="text-forest-500 text-sm mt-1">
              Click the + button on plant cards to add them here.
            </p>
            <Link
              href="/"
              className="inline-block mt-4 px-4 py-2 rounded-lg bg-forest-600 text-white text-sm font-medium hover:bg-forest-700"
            >
              Browse recommendations
            </Link>
          </div>
        ) : (
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {plants.map((p, i) => {
              const tempStr =
                p.temp_min != null && p.temp_max != null
                  ? `${Math.round(p.temp_min)}–${Math.round(p.temp_max)}°F`
                  : null;
              return (
                <Link
                  key={`${p.plant_id}-${p.added_at ?? i}`}
                  href={`/plant/${p.plant_id}`}
                  className="block rounded-xl border border-sage-200 bg-white shadow-leaf overflow-hidden hover:border-sage-300 hover:shadow-[0_6px_20px_rgba(45,58,42,0.1)] transition-all"
                >
                  <div className="aspect-square bg-sage-100 flex items-center justify-center">
                    {p.img_url ? (
                      <img
                        src={p.img_url}
                        alt={p.custom_name}
                        className="w-full h-full object-cover"
                        referrerPolicy="no-referrer"
                      />
                    ) : (
                      <span className="text-4xl text-sage-400">🌱</span>
                    )}
                  </div>
                  <div className="p-4">
                    <p className="font-bold text-forest-800">{p.custom_name}</p>
                    {p.latin && p.custom_name !== p.latin && (
                      <p className="text-forest-600 text-sm italic">{p.latin}</p>
                    )}
                    <div className="mt-2 text-xs text-sage-500 space-y-0.5">
                      {p.sunlight_type && <p>Sunlight: {p.sunlight_type}</p>}
                      {p.care_level && <p>Care: {p.care_level}</p>}
                      {tempStr && <p>Temp: {tempStr}</p>}
                    </div>
                  </div>
                </Link>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
