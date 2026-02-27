"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useParams } from "next/navigation";
import { useRouter } from "next/navigation";
import { getPlant } from "@/lib/api";
import { navigateToAddToGarden } from "@/lib/addToGarden";
import type { PlantRec } from "@/app/components/PlantCard";

type PlantDetail = {
  plant_id: number;
  img_url?: string;
  latin?: string;
  common_name?: string;
  category?: string;
  origin?: string;
  size?: string;
  growth_rate?: string;
  physical_desc?: string;
  symbolism?: string;
  sunlight_type?: string;
  ideal_light?: string;
  tolerated_light?: string;
  humidity?: string;
  humidity_req?: string;
  care_level?: string;
  water_req?: string;
  water_req_raw?: string;
  temp_min?: number;
  temp_max?: number;
  climate?: string;
  soil_type?: string;
  drainage_level?: string;
  bugs?: string[];
  disease?: string[];
};

function formatLabel(s: string): string {
  return s.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

export default function PlantDetailPage() {
  const params = useParams();
  const router = useRouter();
  const id = params?.id as string | undefined;
  const plantId = id ? parseInt(id, 10) : NaN;
  const [plant, setPlant] = useState<PlantDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!id || isNaN(plantId)) {
      setLoading(false);
      setError("Invalid plant ID");
      return;
    }
    getPlant(plantId)
      .then((data) => setPlant(data as PlantDetail))
      .catch((err) => setError(err instanceof Error ? err.message : "Failed to load"))
      .finally(() => setLoading(false));
  }, [id, plantId]);

  if (loading) {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center px-4 bg-gradient-to-b from-sage-50 to-forest-50">
        <p className="text-forest-600">Loading plant...</p>
      </div>
    );
  }

  if (error || !plant) {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center px-4 bg-gradient-to-b from-sage-50 to-forest-50">
        <p className="text-rose-600 mb-4">{error ?? "Plant not found"}</p>
        <Link href="/" className="text-forest-700 font-medium underline">
          Back to home
        </Link>
      </div>
    );
  }

  const tempStr =
    plant.temp_min != null && plant.temp_max != null
      ? `${Math.round(plant.temp_min)}–${Math.round(plant.temp_max)}°F`
      : null;

  const sections: { title: string; items: [string, string | number | string[] | undefined | null][] }[] = [
    {
      title: "About",
      items: [
        ["Latin name", plant.latin],
        ["Common name", plant.common_name],
        ["Category", plant.category ? formatLabel(plant.category) : undefined],
        ["Origin", plant.origin],
        ["Size", plant.size ? formatLabel(plant.size) : undefined],
        ["Growth rate", plant.growth_rate ? formatLabel(plant.growth_rate) : undefined],
      ],
    },
    {
      title: "Description",
      items: [
        ["Physical description", plant.physical_desc],
        ["Symbolism", plant.symbolism],
      ],
    },
    {
      title: "Care",
      items: [
        ["Sunlight", plant.sunlight_type ?? plant.ideal_light ?? plant.tolerated_light],
        ["Humidity", plant.humidity_req ?? plant.humidity],
        ["Watering", plant.water_req_raw ?? plant.water_req],
        ["Temperature", tempStr ?? undefined],
        ["Climate", plant.climate],
        ["Care level", plant.care_level ? formatLabel(plant.care_level) : undefined],
        ["Soil", plant.soil_type],
        ["Drainage", plant.drainage_level ? formatLabel(plant.drainage_level) : undefined],
      ],
    },
    {
      title: "Pests & diseases",
      items: [
        ["Bugs", plant.bugs?.length ? plant.bugs.join(", ") : undefined],
        ["Disease", plant.disease?.length ? plant.disease.join(", ") : undefined],
      ],
    },
  ];

  return (
    <div className="min-h-screen px-4 py-8 bg-gradient-to-b from-sage-50 to-forest-50">
      <div className="max-w-4xl mx-auto">
        <Link
          href="/"
          className="inline-flex items-center gap-1 text-forest-600 hover:text-forest-800 text-sm mb-6"
        >
          ← Back to home
        </Link>

        <div className="rounded-xl border border-sage-200 bg-white shadow-leaf overflow-hidden">
          <div className="p-6 lg:p-8 space-y-6">
            {/* Title + small image in same div */}
            <div className="flex items-center gap-6">
              <div className="w-48 h-48 shrink-0 rounded-lg overflow-hidden bg-sage-100 flex items-center justify-center">
                {plant.img_url ? (
                  <img
                    src={plant.img_url}
                    alt={plant.common_name ?? plant.latin ?? ""}
                    className="w-full h-full object-cover"
                    referrerPolicy="no-referrer"
                  />
                ) : (
                  <span className="text-2xl text-sage-400">🌱</span>
                )}
              </div>
              <div className="flex-1">
                <h1 className="text-2xl font-bold text-forest-800">
                  {plant.common_name ?? plant.latin ?? `Plant #${plant.plant_id}`}
                </h1>
                {plant.latin && plant.common_name && (
                  <p className="text-forest-600 text-base italic">{plant.latin}</p>
                )}
                <button
                    type="button"
                    onClick={() => navigateToAddToGarden({ plant_id: plant.plant_id, score: 0, latin: plant.latin, common_name: plant.common_name, img_url: plant.img_url } as PlantRec, router)}
                    className="mt-3 px-4 py-2 rounded-lg bg-forest-600 text-white text-sm font-medium hover:bg-forest-700 flex items-center gap-2"
                  >
                    <span>+</span> Add to garden
                  </button>
              </div>
            </div>

            {sections.map(({ title, items }) => {
              const filtered = items.filter(([, v]) => v != null && v !== "" && (Array.isArray(v) ? v.length > 0 : true));
              if (filtered.length === 0) return null;
              return (
                <div key={title}>
                  <h2 className="text-sm font-semibold text-forest-700 mb-3">{title}</h2>
                  <dl className="space-y-2">
                    {filtered.map(([label, value]) => (
                      <div key={label} className="flex flex-col sm:flex-row sm:gap-4">
                        <dt className="text-forest-600 text-sm shrink-0 sm:w-36">{formatLabel(label)}</dt>
                        <dd className="text-forest-800 text-sm leading-relaxed">
                          {Array.isArray(value) ? value.join(", ") : String(value)}
                        </dd>
                      </div>
                    ))}
                  </dl>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}
