"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useParams } from "next/navigation";
import { useRouter } from "next/navigation";
import { getPlant } from "@/lib/api";
import { navigateToAddToGarden } from "@/lib/addToGarden";
import { normalizePlantDetail, type PlantDetail } from "@/lib/plantDetail";
import { PlantCatalogTree } from "@/app/components/PlantCatalogTree";
import type { PlantRec } from "@/app/components/PlantCard";

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
      .then((data) => {
        const p = normalizePlantDetail(data);
        if (!p) {
          setError("Invalid plant data");
          return;
        }
        setPlant(p);
      })
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

  const displayName =
    plant.common_name ??
    plant.latin ??
    (typeof plant.catalog?.name === "string" ? plant.catalog.name : null) ??
    `Plant #${plant.plant_id}`;

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
                <h1 className="text-2xl font-bold text-forest-800">{displayName}</h1>
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

            <div className="rounded-2xl border border-sage-200/80 bg-gradient-to-b from-white to-sage-50/30 p-5 sm:p-6 max-h-[min(70vh,calc(100vh-12rem))] overflow-y-auto shadow-inner">
              {plant.catalog && Object.keys(plant.catalog).length > 0 ? (
                <PlantCatalogTree data={plant.catalog} />
              ) : (
                <p className="text-sage-500 text-sm">No catalog payload returned.</p>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
