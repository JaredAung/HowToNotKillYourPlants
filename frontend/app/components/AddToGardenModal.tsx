"use client";

import { useState } from "react";
import Link from "next/link";
import { addToGarden } from "@/lib/api";
import type { PlantRec } from "@/app/components/PlantCard";

type Step = "confirm" | "name" | "done";

export function AddToGardenModal({
  plant,
  onClose,
  onSuccess,
}: {
  plant: PlantRec;
  onClose: () => void;
  onSuccess?: () => void;
}) {
  const [step, setStep] = useState<Step>("confirm");
  const [customName, setCustomName] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const defaultName = plant.latin ?? plant.common_name ?? `Plant #${plant.plant_id}`;

  const handleConfirm = () => {
    setStep("name");
  };

  const handleSubmit = async () => {
    setLoading(true);
    setError(null);
    try {
      const name = customName.trim() || undefined;
      await addToGarden(plant.plant_id, name);
      setStep("done");
      onSuccess?.();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to add");
    } finally {
      setLoading(false);
    }
  };

  const handleSkipName = async () => {
    setLoading(true);
    setError(null);
    try {
      await addToGarden(plant.plant_id, undefined);
      setStep("done");
      onSuccess?.();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to add");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      <div className="absolute inset-0 bg-black/50" onClick={onClose} aria-hidden="true" />
      <div className="relative w-full max-w-md rounded-xl border border-sage-200 bg-white shadow-leaf p-6">
        <h2 className="text-lg font-semibold text-forest-800 mb-4">
          {step === "confirm" && "Add to garden"}
          {step === "name" && "Name your plant"}
          {step === "done" && "Added!"}
        </h2>

        {step === "confirm" && (
          <>
            <p className="text-forest-600 mb-4">
              Are you sure you want to plant this? ({plant.common_name ?? plant.latin ?? `Plant #${plant.plant_id}`})
            </p>
            <div className="flex gap-3 justify-end">
              <button
                type="button"
                onClick={onClose}
                className="px-4 py-2 rounded-lg border border-sage-300 text-forest-700 text-sm font-medium hover:bg-sage-50"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={handleConfirm}
                className="px-4 py-2 rounded-lg bg-forest-600 text-white text-sm font-medium hover:bg-forest-700"
              >
                Yes
              </button>
            </div>
          </>
        )}

        {step === "name" && (
          <>
            <p className="text-forest-600 mb-3">Would you like to name the plant?</p>
            <p className="text-sage-500 text-sm mb-3">If you skip, we&apos;ll use &quot;{defaultName}&quot;</p>
            <input
              type="text"
              placeholder={defaultName}
              value={customName}
              onChange={(e) => setCustomName(e.target.value)}
              className="w-full px-3 py-2 rounded-lg border border-sage-200 text-forest-800 text-sm placeholder:text-sage-400 focus:outline-none focus:ring-2 focus:ring-sage-400 focus:border-transparent mb-4"
            />
            {error && <p className="text-rose-600 text-sm mb-3">{error}</p>}
            <div className="flex gap-3 justify-end">
              <button
                type="button"
                onClick={handleSkipName}
                disabled={loading}
                className="px-4 py-2 rounded-lg border border-sage-300 text-forest-700 text-sm font-medium hover:bg-sage-50 disabled:opacity-50"
              >
                Skip
              </button>
              <button
                type="button"
                onClick={handleSubmit}
                disabled={loading}
                className="px-4 py-2 rounded-lg bg-forest-600 text-white text-sm font-medium hover:bg-forest-700 disabled:opacity-50"
              >
                {loading ? "Adding..." : "Add"}
              </button>
            </div>
          </>
        )}

        {step === "done" && (
          <>
            <p className="text-forest-600 mb-4">Your plant has been added to your garden.</p>
            <div className="flex gap-3">
              <button
                type="button"
                onClick={onClose}
                className="flex-1 px-4 py-2 rounded-lg border border-sage-300 text-forest-700 text-sm font-medium hover:bg-sage-50"
              >
                Done
              </button>
              <Link
                href="/garden"
                onClick={onClose}
                className="flex-1 px-4 py-2 rounded-lg bg-forest-600 text-white text-sm font-medium hover:bg-forest-700 text-center"
              >
                View garden
              </Link>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
