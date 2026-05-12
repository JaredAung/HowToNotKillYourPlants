"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { getToken, updateProfile } from "@/lib/api";

/** Same strings as plant catalog / ``ORDINAL_ORDERS.light`` (stored as ``environment.light_level``). */
const LIGHT_OPTIONS = [
  { id: "full shade", label: "Full shade", icon: "🌫️", desc: "Low light, north window or deep shade" },
  { id: "partial sun/shade", label: "Partial sun / shade", icon: "🌤️", desc: "Bright indirect, east/west, filtered sun" },
  { id: "full sun", label: "Full sun", icon: "☀️", desc: "Direct sun several hours, south-facing" },
];

const CARE_LEVEL_OPTIONS = [
  { id: "easy", label: "Easy", desc: "Low maintenance, forgiving" },
  { id: "medium", label: "Medium", desc: "Moderate care needed" },
  { id: "hard", label: "Hard", desc: "Requires attention" },
];

const SIZE_OPTIONS = [
  { id: "small", label: "Small", desc: "Tabletop, shelf" },
  { id: "medium", label: "Medium", desc: "Desk, side table" },
  { id: "large", label: "Large", desc: "Floor, statement" },
];

const CLIMATE_OPTIONS = [
  { id: "alpine", label: "Alpine", desc: "Cool, high elevation" },
  { id: "arid", label: "Arid", desc: "Dry climates" },
  { id: "mediterranean", label: "Mediterranean", desc: "Mild, wet winters / dry summers" },
  { id: "temperate", label: "Temperate", desc: "Four seasons" },
  { id: "tropical", label: "Tropical", desc: "Warm, humid" },
];

const SOIL_OPTIONS = [
  { id: "light", label: "Light", desc: "Sandy, drains fast" },
  { id: "medium", label: "Medium", desc: "Loamy" },
  { id: "heavy", label: "Heavy", desc: "Clay-rich" },
];

const GROWTH_OPTIONS = [
  { id: "slow", label: "Slow", desc: "Gradual" },
  { id: "med", label: "Medium", desc: "Moderate pace" },
  { id: "fast", label: "Fast", desc: "Vigorous" },
];

const WATER_OPTIONS = [
  { id: "low", label: "Low" },
  { id: "medium", label: "Medium" },
  { id: "high", label: "High" },
];

function CardSelect<T extends string>({
  options,
  value,
  onChange,
}: {
  options: { id: T; label: string }[];
  value: T | null;
  onChange: (v: T) => void;
}) {
  return (
    <div className="flex flex-wrap gap-2">
      {options.map((opt) => {
        const isSel = value === opt.id;
        return (
          <button
            key={opt.id}
            type="button"
            onClick={() => onChange(opt.id)}
            className={`px-4 py-2.5 rounded-lg border-2 font-medium text-sm transition-all border-sage-200 text-forest-600 ${
              isSel ? "border-forest-600 bg-forest-50 text-forest-800" : ""
            }`}
          >
            {opt.label}
          </button>
        );
      })}
    </div>
  );
}

function LightCards({ value, onChange }: { value: string | null; onChange: (v: string) => void }) {
  const lightVisuals: Record<string, { bg: string; rays: string }> = {
    "full shade": { bg: "bg-sage-100", rays: "opacity-25" },
    "partial sun/shade": { bg: "bg-amber-100", rays: "opacity-60" },
    "full sun": { bg: "bg-amber-300", rays: "opacity-100" },
  };
  return (
    <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
      {LIGHT_OPTIONS.map((opt) => {
        const v = lightVisuals[opt.id] ?? lightVisuals["partial sun/shade"];
        return (
          <button
            key={opt.id}
            type="button"
            onClick={() => onChange(opt.id)}
            className={`p-4 rounded-xl border-2 text-center transition-all flex flex-col items-center ${
              value === opt.id ? "border-forest-600 bg-forest-50" : "border-sage-200 hover:border-sage-300 bg-white"
            }`}
          >
            <div className={`w-14 h-14 rounded-full ${v.bg} flex items-center justify-center mb-2 shadow-inner`}>
              <span className={`text-2xl ${v.rays}`}>{opt.icon}</span>
            </div>
            <span className="font-medium text-forest-800 text-sm">{opt.label}</span>
            <span className="block text-xs text-forest-600 mt-0.5">{opt.desc}</span>
          </button>
        );
      })}
    </div>
  );
}

/** Payload aligned with GET /profile and two-tower feature_loader. */
function buildTowerProfileJson(username: string, form: {
  displayName: string;
  climate: string | null;
  usdaZoneMin: string;
  usdaZoneMax: string;
  lightLevel: string | null;
  soilPreference: string | null;
  growthPref: string | null;
  tempMinF: string;
  tempMaxF: string;
  careLevel: string | null;
  preferredSize: string | null;
  wateringFreq: string | null;
}) {
  const parseZone = (s: string) => {
    const n = parseInt(s, 10);
    return Number.isFinite(n) ? Math.min(13, Math.max(1, n)) : null;
  };
  const zmin = form.usdaZoneMin.trim() ? parseZone(form.usdaZoneMin) : null;
  const zmax = form.usdaZoneMax.trim() ? parseZone(form.usdaZoneMax) : null;
  return {
    username,
    profile: {
      name: form.displayName.trim() || null,
    },
    climate: form.climate,
    ...(zmin != null ? { usda_zone_min: zmin } : {}),
    ...(zmax != null ? { usda_zone_max: zmax } : {}),
    environment: {
      light_level: form.lightLevel,
      soil_preference: form.soilPreference,
      temperature_pref: {
        min_f: form.tempMinF ? parseFloat(form.tempMinF) : null,
        max_f: form.tempMaxF ? parseFloat(form.tempMaxF) : null,
      },
    },
    constraints: {
      preferred_size: form.preferredSize,
    },
    preferences: {
      care_level: form.careLevel,
      growth_pref: form.growthPref,
      care_preferences: {
        watering_freq: form.wateringFreq,
      },
    },
  };
}

export default function OnboardingPage() {
  const router = useRouter();
  const [step, setStep] = useState(1);
  const [username, setUsername] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [extractedJson, setExtractedJson] = useState<Record<string, unknown> | null>(null);
  const [submitError, setSubmitError] = useState<string | null>(null);

  const [displayName, setDisplayName] = useState("");
  const [climate, setClimate] = useState<string | null>(null);
  const [usdaZoneMin, setUsdaZoneMin] = useState("");
  const [usdaZoneMax, setUsdaZoneMax] = useState("");

  const [lightLevel, setLightLevel] = useState<string | null>(null);
  const [tempMinF, setTempMinF] = useState("");
  const [tempMaxF, setTempMaxF] = useState("");
  const [soilPreference, setSoilPreference] = useState<string | null>(null);
  const [growthPref, setGrowthPref] = useState<string | null>(null);

  const [careLevel, setCareLevel] = useState<string | null>(null);
  const [preferredSize, setPreferredSize] = useState<string | null>(null);
  const [wateringFreq, setWateringFreq] = useState<string | null>(null);

  const [userLoading, setUserLoading] = useState(true);
  useEffect(() => {
    const loadUser = async () => {
      if (typeof window === "undefined") return;
      const stored = window.sessionStorage.getItem("userUsername");
      if (stored) {
        setUsername(stored);
        setUserLoading(false);
        return;
      }
      try {
        const { getMe, getProfile } = await import("@/lib/api");
        const me = await getMe();
        if (me?.username) {
          setUsername(me.username);
          window.sessionStorage.setItem("userUsername", me.username);
        }
        try {
          const prof = await getProfile();
          const n = prof && typeof prof === "object" && "profile" in prof ? (prof as { profile?: { name?: string } }).profile?.name : undefined;
          if (typeof n === "string" && n.trim()) setDisplayName(n.trim());
        } catch {
          // ignore
        }
      } catch {
        // No token or expired
      } finally {
        setUserLoading(false);
      }
    };
    loadUser();
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!getToken()) {
      router.push("/auth");
      return;
    }
    if (!username) {
      setSubmitError("Loading user...");
      return;
    }
    setLoading(true);
    setSubmitError(null);
    try {
      await updateProfile({
        name: displayName.trim(),
        climate: climate || undefined,
        light_level: lightLevel || undefined,
        soil_preference: soilPreference || undefined,
        temp_min_f: tempMinF ? parseFloat(tempMinF) : undefined,
        temp_max_f: tempMaxF ? parseFloat(tempMaxF) : undefined,
        care_level: careLevel || undefined,
        growth_pref: growthPref || undefined,
        preferred_size: preferredSize || undefined,
        watering_freq: wateringFreq || undefined,
        usda_zone_min: usdaZoneMin.trim() ? parseInt(usdaZoneMin, 10) : undefined,
        usda_zone_max: usdaZoneMax.trim() ? parseInt(usdaZoneMax, 10) : undefined,
      });
      setExtractedJson(
        buildTowerProfileJson(username, {
          displayName,
          climate,
          usdaZoneMin,
          usdaZoneMax,
          lightLevel,
          soilPreference,
          growthPref,
          tempMinF,
          tempMaxF,
          careLevel,
          preferredSize,
          wateringFreq,
        }) as Record<string, unknown>
      );
    } catch (err) {
      setSubmitError(err instanceof Error ? err.message : "Profile update failed");
    } finally {
      setLoading(false);
    }
  };

  const inputCls =
    "w-full px-4 py-2.5 rounded-lg border border-sage-300 bg-white text-forest-800 placeholder:text-sage-400 focus:outline-none focus:ring-2 focus:ring-sage-400 focus:border-transparent";
  const labelCls = "block text-sm font-medium text-forest-700 mb-2";
  const sectionCls = "mb-6";

  if (userLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center px-4 bg-gradient-to-b from-sage-50 to-forest-50">
        <p className="text-forest-600">Loading...</p>
      </div>
    );
  }
  if (!getToken()) {
    return (
      <div className="min-h-screen flex items-center justify-center px-4 bg-gradient-to-b from-sage-50 to-forest-50">
        <div className="text-center">
          <p className="text-forest-600 mb-4">Please sign in first.</p>
          <Link href="/auth" className="text-forest-700 font-medium underline">
            Go to sign in
          </Link>
        </div>
      </div>
    );
  }

  if (extractedJson) {
    return (
      <div className="min-h-screen px-4 py-8 bg-gradient-to-b from-sage-50 to-forest-50">
        <div className="max-w-2xl mx-auto">
          <Link href="/" className="inline-flex items-center gap-2 text-forest-700 hover:text-forest-800 mb-8">
            <span className="text-xl">🌿</span>
            <span className="font-medium">How to Keep Your Plants Alive</span>
          </Link>
          <div className="bg-white/80 backdrop-blur rounded-2xl shadow-leaf border border-sage-200/60 p-8">
            <h1 className="text-xl font-semibold text-forest-800 mb-2">Your recommendation profile</h1>
            <p className="text-sm text-forest-600 mb-4">Saved fields used for the same user tower as model training:</p>
            <pre className="p-4 rounded-lg bg-sage-100 text-forest-800 text-sm overflow-x-auto overflow-y-auto max-h-[60vh] border border-sage-200">
              {JSON.stringify(extractedJson, null, 2)}
            </pre>
            <Link
              href="/"
              className="mt-6 inline-block w-full py-3 rounded-lg bg-forest-600 text-white font-medium text-center hover:bg-forest-700 transition-colors"
            >
              Continue to home
            </Link>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen px-4 py-8 bg-gradient-to-b from-sage-50 to-forest-50">
      <div className="max-w-lg mx-auto">
        <Link href="/" className="inline-flex items-center gap-2 text-forest-700 hover:text-forest-800 mb-8">
          <span className="text-xl">🌿</span>
          <span className="font-medium">How to Keep Your Plants Alive</span>
        </Link>

        <div className="bg-white/80 backdrop-blur rounded-2xl shadow-leaf border border-sage-200/60 p-8">
          <h1 className="text-xl font-semibold text-forest-800 mb-2">Your growing conditions</h1>
          <p className="text-sm text-forest-600 mb-6">
            Same inputs as the two-tower recommendation model (climate, light, soil, growth, temps, care, size,
            watering, USDA zones).
          </p>

          <div className="flex gap-2 mb-6">
            {[1, 2, 3].map((s) => (
              <div key={s} className={`h-1.5 flex-1 rounded-full ${step >= s ? "bg-forest-600" : "bg-sage-200"}`} />
            ))}
          </div>

          <form onSubmit={handleSubmit}>
            {step === 1 && (
              <>
                <div className={sectionCls}>
                  <label className={labelCls}>Name (optional)</label>
                  <p className="text-xs text-forest-600 mb-2">
                    How we&apos;ll greet you.
                    {username ? (
                      <>
                        {" "}
                        Login username: <span className="font-medium text-forest-700">@{username}</span>
                      </>
                    ) : null}
                  </p>
                  <input
                    type="text"
                    value={displayName}
                    onChange={(e) => setDisplayName(e.target.value)}
                    className={inputCls}
                    placeholder="e.g. Jordan"
                    autoComplete="name"
                  />
                </div>
                <div className={sectionCls}>
                  <label className={labelCls}>Climate (plant origin category)</label>
                  <div className="flex flex-col gap-2">
                    {CLIMATE_OPTIONS.map((opt) => (
                      <button
                        key={opt.id}
                        type="button"
                        onClick={() => setClimate(opt.id)}
                        className={`p-3 rounded-lg border-2 text-left transition-all ${
                          climate === opt.id ? "border-forest-600 bg-forest-50" : "border-sage-200"
                        }`}
                      >
                        <span className="font-medium text-forest-800">{opt.label}</span>
                        <span className="block text-xs text-forest-600">{opt.desc}</span>
                      </button>
                    ))}
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className={labelCls}>USDA zone min (1–13, optional)</label>
                    <input
                      type="number"
                      min={1}
                      max={13}
                      value={usdaZoneMin}
                      onChange={(e) => setUsdaZoneMin(e.target.value)}
                      className={inputCls}
                      placeholder="e.g. 6"
                    />
                  </div>
                  <div>
                    <label className={labelCls}>USDA zone max (optional)</label>
                    <input
                      type="number"
                      min={1}
                      max={13}
                      value={usdaZoneMax}
                      onChange={(e) => setUsdaZoneMax(e.target.value)}
                      className={inputCls}
                      placeholder="e.g. 9"
                    />
                  </div>
                </div>
              </>
            )}

            {step === 2 && (
              <>
                <div className={sectionCls}>
                  <label className={labelCls}>Light at your plants</label>
                  <LightCards value={lightLevel} onChange={setLightLevel} />
                </div>
                <div className={sectionCls}>
                  <label className={labelCls}>Soil you prefer</label>
                  <div className="flex flex-wrap gap-2">
                    {SOIL_OPTIONS.map((opt) => (
                      <button
                        key={opt.id}
                        type="button"
                        onClick={() => setSoilPreference(opt.id)}
                        className={`p-3 rounded-lg border-2 text-left transition-all ${
                          soilPreference === opt.id ? "border-forest-600 bg-forest-50" : "border-sage-200"
                        }`}
                      >
                        <span className="font-medium text-forest-800">{opt.label}</span>
                        <span className="block text-xs text-forest-600">{opt.desc}</span>
                      </button>
                    ))}
                  </div>
                </div>
                <div className={sectionCls}>
                  <label className={labelCls}>Growth pace you like</label>
                  <div className="flex flex-wrap gap-2">
                    {GROWTH_OPTIONS.map((opt) => (
                      <button
                        key={opt.id}
                        type="button"
                        onClick={() => setGrowthPref(opt.id)}
                        className={`p-3 rounded-lg border-2 text-left transition-all ${
                          growthPref === opt.id ? "border-forest-600 bg-forest-50" : "border-sage-200"
                        }`}
                      >
                        <span className="font-medium text-forest-800">{opt.label}</span>
                        <span className="block text-xs text-forest-600">{opt.desc}</span>
                      </button>
                    ))}
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className={labelCls}>Indoor temp min °F (optional)</label>
                    <input type="number" value={tempMinF} onChange={(e) => setTempMinF(e.target.value)} className={inputCls} placeholder="65" />
                  </div>
                  <div>
                    <label className={labelCls}>Indoor temp max °F (optional)</label>
                    <input type="number" value={tempMaxF} onChange={(e) => setTempMaxF(e.target.value)} className={inputCls} placeholder="78" />
                  </div>
                </div>
              </>
            )}

            {step === 3 && (
              <>
                <div className={sectionCls}>
                  <label className={labelCls}>Care level you want</label>
                  <div className="flex flex-wrap gap-2">
                    {CARE_LEVEL_OPTIONS.map((opt) => (
                      <button
                        key={opt.id}
                        type="button"
                        onClick={() => setCareLevel(opt.id)}
                        className={`p-3 rounded-lg border-2 text-left transition-all ${
                          careLevel === opt.id ? "border-forest-600 bg-forest-50" : "border-sage-200"
                        }`}
                      >
                        <span className="font-medium text-forest-800">{opt.label}</span>
                        <span className="block text-xs text-forest-600">{opt.desc}</span>
                      </button>
                    ))}
                  </div>
                </div>
                <div className={sectionCls}>
                  <label className={labelCls}>Preferred plant size</label>
                  <div className="flex flex-wrap gap-2">
                    {SIZE_OPTIONS.map((opt) => (
                      <button
                        key={opt.id}
                        type="button"
                        onClick={() => setPreferredSize(opt.id)}
                        className={`p-3 rounded-lg border-2 text-left transition-all ${
                          preferredSize === opt.id ? "border-forest-600 bg-forest-50" : "border-sage-200"
                        }`}
                      >
                        <span className="font-medium text-forest-800">{opt.label}</span>
                        <span className="block text-xs text-forest-600">{opt.desc}</span>
                      </button>
                    ))}
                  </div>
                </div>
                <div className={sectionCls}>
                  <label className={labelCls}>Watering you can offer (low / medium / high)</label>
                  <CardSelect options={WATER_OPTIONS} value={wateringFreq} onChange={(v) => setWateringFreq(v)} />
                </div>
              </>
            )}

            <div className="flex gap-3 mt-8">
              {step > 1 && (
                <button type="button" onClick={() => setStep((s) => s - 1)} className="flex-1 py-3 rounded-lg border-2 border-sage-300 text-forest-700 font-medium">
                  Back
                </button>
              )}
              {step < 3 ? (
                <button type="button" onClick={() => setStep((s) => s + 1)} className="flex-1 py-3 rounded-lg bg-forest-600 text-white font-medium">
                  Next
                </button>
              ) : (
                <div className="flex-1 flex flex-col gap-2">
                  {submitError && <p className="text-sm text-rose-600">{submitError}</p>}
                  <button type="submit" disabled={loading} className="py-3 rounded-lg bg-forest-600 text-white font-medium disabled:opacity-60">
                    {loading ? "Saving..." : "Finish"}
                  </button>
                </div>
              )}
            </div>
          </form>
        </div>
      </div>
    </div>
  );
}
