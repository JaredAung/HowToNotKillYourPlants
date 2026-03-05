"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { getGarden, getProfile, getToken, markPlantDead } from "@/lib/api";

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

const WHAT_HAPPENED_OPTIONS = [
  "Overwatered",
  "Underwatered",
  "Not enough light",
  "Too much sun",
  "Pests",
  "Temperature or drafts",
  "Root rot",
  "Not sure",
];

const WATERING_OPTIONS = [
  "Every day",
  "Every 2 days",
  "Weekly",
];

const LOCATION_OPTIONS = [
  "Direct sunlight",
  "Bright light",
  "Bright indirect light",
  "Medium indirect light",
  "Low light",
];

const HUMIDITY_OPTIONS = ["Low", "Medium", "High"];

const ROOM_TEMP_OPTIONS = ["Cold", "Comfortable", "Hot"];

function GardenPlantCard({
  p,
  onDeathClick,
}: {
  p: GardenPlant;
  onDeathClick: (plant: GardenPlant) => void;
}) {
  const [menuOpen, setMenuOpen] = useState(false);
  const tempStr =
    p.temp_min != null && p.temp_max != null
      ? `${Math.round(p.temp_min)}–${Math.round(p.temp_max)}°F`
      : null;

  return (
    <div className="relative rounded-xl border border-sage-200 bg-white shadow-leaf overflow-hidden hover:border-sage-300 hover:shadow-[0_6px_20px_rgba(45,58,42,0.1)] transition-all">
      <button
        type="button"
        onClick={(e) => {
          e.preventDefault();
          e.stopPropagation();
          setMenuOpen((o) => !o);
        }}
        className="absolute top-2 right-2 z-20 h-8 w-8 rounded-lg bg-white/90 shadow border border-sage-200 flex items-center justify-center text-forest-600 hover:bg-sage-100"
        aria-label="Options"
      >
        <span className="text-lg leading-none">⋮</span>
      </button>
      {menuOpen && (
        <>
          <div
            className="fixed inset-0 z-30"
            aria-hidden
            onClick={() => setMenuOpen(false)}
          />
          <div className="absolute top-12 right-2 z-40 min-w-[120px] rounded-lg border border-sage-200 bg-white shadow-lg py-1">
            <button
              type="button"
              onClick={(e) => {
                e.preventDefault();
                e.stopPropagation();
                setMenuOpen(false);
                onDeathClick(p);
              }}
              className="w-full px-4 py-2 text-left text-sm text-rose-600 hover:bg-rose-50"
            >
              Death
            </button>
          </div>
        </>
      )}
      <Link
        href={`/plant/${p.plant_id}`}
        className="block"
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
    </div>
  );
}

function DeathModal({
  plant,
  onClose,
  onSuccess,
}: {
  plant: GardenPlant | null;
  onClose: () => void;
  onSuccess: () => void;
}) {
  const [whatHappened, setWhatHappened] = useState<string[]>([]);
  const [wateringFrequency, setWateringFrequency] = useState<string>("");
  const [plantLocation, setPlantLocation] = useState<string>("");
  const [humidityLevel, setHumidityLevel] = useState<string>("");
  const [roomTemperature, setRoomTemperature] = useState<string>("");
  const [deathReason, setDeathReason] = useState("");
  const [userProfile, setUserProfile] = useState<Record<string, unknown> | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const toggleWhatHappened = (opt: string) => {
    setWhatHappened((prev) =>
      prev.includes(opt) ? prev.filter((x) => x !== opt) : [...prev, opt]
    );
  };

  useEffect(() => {
    if (!plant) return;
    setWhatHappened([]);
    setWateringFrequency("");
    setPlantLocation("");
    setHumidityLevel("");
    setRoomTemperature("");
    setDeathReason("");
    setErr(null);
    getProfile()
      .then((data) => setUserProfile(data ?? null))
      .catch(() => setUserProfile(null));
  }, [plant]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!plant) return;
    setSubmitting(true);
    setErr(null);
    const plantProfile: Record<string, unknown> = {
      plant_id: plant.plant_id,
      custom_name: plant.custom_name,
      latin: plant.latin,
      common_name: plant.common_name,
      sunlight_type: plant.sunlight_type,
      humidity: plant.humidity,
      care_level: plant.care_level,
      water_req: plant.water_req,
      temp_min: plant.temp_min,
      temp_max: plant.temp_max,
    };
    try {
      await markPlantDead({
        plant_id: plant.plant_id,
        what_happened: whatHappened,
        watering_frequency: wateringFrequency || undefined,
        plant_location: plantLocation || undefined,
        humidity_level: humidityLevel || undefined,
        room_temperature: roomTemperature || undefined,
        death_reason: deathReason.trim() || undefined,
        plant_profile: plantProfile,
        user_profile: userProfile ?? undefined,
      });
      onSuccess();
      onClose();
    } catch (error) {
      setErr(error instanceof Error ? error.message : "Failed to submit");
    } finally {
      setSubmitting(false);
    }
  };

  if (!plant) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/40" aria-modal>
      <div className="absolute inset-0" aria-hidden onClick={onClose} />
      <div className="relative w-full max-w-md max-h-[90vh] overflow-y-auto rounded-xl border border-sage-200 bg-white shadow-xl p-6">
        <h2 className="text-lg font-semibold text-forest-800 mb-1">Mark plant as dead</h2>
        <p className="text-sm text-forest-600 mb-4">{plant.custom_name}</p>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <p className="block text-sm font-medium text-forest-700 mb-2">
              What do you think happened? (optional)
            </p>
            <div className="space-y-2">
              {WHAT_HAPPENED_OPTIONS.map((opt) => (
                <label key={opt} className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={whatHappened.includes(opt)}
                    onChange={() => toggleWhatHappened(opt)}
                    className="rounded border-sage-300 text-forest-600 focus:ring-forest-500"
                  />
                  <span className="text-sm text-forest-700">{opt}</span>
                </label>
              ))}
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-forest-700 mb-2">
              How often were you watering the plant? (optional)
            </label>
            <div className="space-y-2">
              {WATERING_OPTIONS.map((opt) => (
                <label key={opt} className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="radio"
                    name="watering"
                    checked={wateringFrequency === opt}
                    onChange={() => setWateringFrequency(opt)}
                    className="border-sage-300 text-forest-600 focus:ring-forest-500"
                  />
                  <span className="text-sm text-forest-700">{opt}</span>
                </label>
              ))}
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-forest-700 mb-2">
              Where was the plant located? (optional)
            </label>
            <div className="space-y-2">
              {LOCATION_OPTIONS.map((opt) => (
                <label key={opt} className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="radio"
                    name="location"
                    checked={plantLocation === opt}
                    onChange={() => setPlantLocation(opt)}
                    className="border-sage-300 text-forest-600 focus:ring-forest-500"
                  />
                  <span className="text-sm text-forest-700">{opt}</span>
                </label>
              ))}
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-forest-700 mb-2">
              How humid was the room? (optional)
            </label>
            <div className="space-y-2">
              {HUMIDITY_OPTIONS.map((opt) => (
                <label key={opt} className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="radio"
                    name="humidity"
                    checked={humidityLevel === opt}
                    onChange={() => setHumidityLevel(opt)}
                    className="border-sage-300 text-forest-600 focus:ring-forest-500"
                  />
                  <span className="text-sm text-forest-700">{opt}</span>
                </label>
              ))}
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-forest-700 mb-2">
              Was the room usually cold, comfortable, or hot? (optional)
            </label>
            <div className="space-y-2">
              {ROOM_TEMP_OPTIONS.map((opt) => (
                <label key={opt} className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="radio"
                    name="roomTemp"
                    checked={roomTemperature === opt}
                    onChange={() => setRoomTemperature(opt)}
                    className="border-sage-300 text-forest-600 focus:ring-forest-500"
                  />
                  <span className="text-sm text-forest-700">{opt}</span>
                </label>
              ))}
            </div>
          </div>

          <div>
            <label htmlFor="death-reason" className="block text-sm font-medium text-forest-700 mb-1">
              Anything else? (optional)
            </label>
            <textarea
              id="death-reason"
              value={deathReason}
              onChange={(e) => setDeathReason(e.target.value)}
              placeholder="Additional notes..."
              rows={2}
              className="w-full rounded-lg border border-sage-200 px-3 py-2 text-sm text-forest-800 placeholder:text-sage-400 focus:border-forest-500 focus:outline-none focus:ring-1 focus:ring-forest-500"
            />
          </div>

          {err && <p className="text-sm text-rose-600">{err}</p>}

          <div className="flex gap-3 justify-end">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 text-sm font-medium text-forest-600 hover:bg-sage-100 rounded-lg"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={submitting}
              className="px-4 py-2 text-sm font-medium text-white bg-rose-600 hover:bg-rose-700 disabled:opacity-50 rounded-lg"
            >
              {submitting ? "Submitting..." : "Mark as dead"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

export default function GardenPage() {
  const [plants, setPlants] = useState<GardenPlant[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [deathPlant, setDeathPlant] = useState<GardenPlant | null>(null);
  const isLoggedIn = !!getToken();

  const refreshGarden = useCallback(() => {
    return getGarden()
      .then((data: { plants?: GardenPlant[] }) => {
        setPlants(data?.plants ?? []);
        setError(null);
      })
      .catch((err) => setError(err instanceof Error ? err.message : "Failed to load"));
  }, []);

  useEffect(() => {
    if (!isLoggedIn) {
      setLoading(false);
      return;
    }
    refreshGarden().finally(() => setLoading(false));
  }, [isLoggedIn, refreshGarden]);

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
            {plants.map((p, i) => (
              <GardenPlantCard
                key={`${p.plant_id}-${p.added_at ?? i}`}
                p={p}
                onDeathClick={setDeathPlant}
              />
            ))}
          </div>
        )}
      </div>

      {deathPlant && (
        <DeathModal
          plant={deathPlant}
          onClose={() => setDeathPlant(null)}
          onSuccess={refreshGarden}
        />
      )}
    </div>
  );
}
