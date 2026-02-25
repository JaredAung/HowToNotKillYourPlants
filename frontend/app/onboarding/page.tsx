"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { getToken, updateProfile } from "@/lib/api";

const LIGHT_OPTIONS = [
  { id: "direct", label: "Direct", icon: "🌞", desc: "Full sun, south-facing window" },
  { id: "bright_light", label: "Bright light", icon: "☀️", desc: "Several hours of sun" },
  { id: "bright_indirect", label: "Bright indirect", icon: "🌤️", desc: "Sun filtered through sheer" },
  { id: "indirect", label: "Indirect", icon: "🌥️", desc: "No direct rays" },
  { id: "diffused", label: "Diffused", icon: "🌫️", desc: "Soft, even light" },
];

const HUMIDITY_OPTIONS = [
  { id: "low", label: "Low", desc: "Dry air" },
  { id: "medium", label: "Medium", desc: "Moderate humidity" },
  { id: "high", label: "High", desc: "Humid" },
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
  { id: "Arid Tropical", label: "Arid Tropical" },
  { id: "Subtropical", label: "Subtropical" },
  { id: "Subtropical arid", label: "Subtropical arid" },
  { id: "Tropical", label: "Tropical" },
  { id: "Tropical humid", label: "Tropical humid" },
];

const HARD_NO_OPTIONS = [
  { id: "frequent_watering", label: "Frequent watering" },
];

const CARD_BTN = "px-4 py-2.5 rounded-lg border-2 font-medium text-sm transition-all border-sage-200 text-forest-600";
const CARD_BTN_SEL = "border-forest-600 bg-forest-50 text-forest-800";

function CardSelect<T extends string>({
  options,
  value,
  onChange,
  multi,
}: {
  options: { id: T; label: string; icon?: string; desc?: string }[];
  value: T | T[] | null;
  onChange: (v: T | T[]) => void;
  multi?: boolean;
}) {
  const sel = (id: T) => {
    if (multi && Array.isArray(value)) {
      const next = value.includes(id) ? value.filter((x) => x !== id) : [...value, id];
      onChange(next);
    } else {
      onChange(id);
    }
  };
  return (
    <div className="flex flex-wrap gap-2">
      {options.map((opt) => {
        const isSel = multi && Array.isArray(value) ? value.includes(opt.id) : value === opt.id;
        return (
          <button
            key={opt.id}
            type="button"
            onClick={() => sel(opt.id)}
            className={`${CARD_BTN} ${isSel ? CARD_BTN_SEL : ""}`}
          >
            {opt.icon && <span className="mr-1">{opt.icon}</span>}
            {opt.label}
          </button>
        );
      })}
    </div>
  );
}

function LightCards({ value, onChange }: { value: string | null; onChange: (v: string) => void }) {
  const lightVisuals: Record<string, { bg: string; rays: string }> = {
    direct: { bg: "bg-amber-300", rays: "opacity-100" },
    bright_light: { bg: "bg-amber-200", rays: "opacity-90" },
    bright_indirect: { bg: "bg-amber-100", rays: "opacity-60" },
    indirect: { bg: "bg-sage-200", rays: "opacity-40" },
    diffused: { bg: "bg-sage-100", rays: "opacity-25" },
  };
  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
      {LIGHT_OPTIONS.map((opt) => {
        const v = lightVisuals[opt.id] || lightVisuals.diffused;
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

function YesNo({ value, onChange }: { value: boolean | null; onChange: (v: boolean) => void }) {
  return (
    <div className="flex gap-3">
      <button
        type="button"
        onClick={() => onChange(true)}
        className={`flex-1 py-3 rounded-lg border-2 font-medium transition-all ${
          value === true ? "border-forest-600 bg-forest-50 text-forest-800" : "border-sage-200 text-forest-600"
        }`}
      >
        Yes
      </button>
      <button
        type="button"
        onClick={() => onChange(false)}
        className={`flex-1 py-3 rounded-lg border-2 font-medium transition-all ${
          value === false ? "border-forest-600 bg-forest-50 text-forest-800" : "border-sage-200 text-forest-600"
        }`}
      >
        No
      </button>
    </div>
  );
}

function buildProfileJson(username: string, form: {
  name: string; avatarUrl: string;
  city: string; state: string; postalCode: string; country: string;
  climate: string | null;
  lightLevel: string | null; humidityLevel: string | null; tempMinF: string; tempMaxF: string;
  hasKids: boolean | null;
  careLevel: string | null; preferredSize: string | null; hardNo: string[];
  wateringFreq: string | null; careFreq: string | null;
}) {
  const obj: Record<string, unknown> = {
    auth: { username },
    profile: {
      name: form.name || null,
      avatar_url: form.avatarUrl || null,
    },
    location: {
      city: form.city || null,
      state: form.state || null,
      postal_code: form.postalCode || null,
      country: form.country || null,
    },
    climate: form.climate,
    environment: {
      light_level: form.lightLevel,
      humidity_level: form.humidityLevel,
      temperature_pref: {
        min_f: form.tempMinF ? parseFloat(form.tempMinF) : null,
        max_f: form.tempMaxF ? parseFloat(form.tempMaxF) : null,
      },
    },
    safety: {
      has_kids: form.hasKids,
    },
    constraints: {
      preferred_size: form.preferredSize,
      hard_no: form.hardNo,
    },
    preferences: {
      care_level: form.careLevel,
      care_preferences: {
        watering_freq: form.wateringFreq,
        care_freq: form.careFreq,
      },
    },
    gamification: { care_points: 0, care_level: 0, streak_days: 0, multiplier: 1, badges: [] },
    social: { friends: [], neighborhood_id: null },
    history: { owned_plants_count: 0, deaths_count: 0, average_health_score: 0, last_death_reason: null },
  };
  return obj;
}

export default function OnboardingPage() {
  const router = useRouter();
  const [step, setStep] = useState(1);
  const [username, setUsername] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [extractedJson, setExtractedJson] = useState<Record<string, unknown> | null>(null);
  const [submitError, setSubmitError] = useState<string | null>(null);

  // Profile
  const [name, setName] = useState("");
  const [avatarUrl, setAvatarUrl] = useState("");

  // Location
  const [city, setCity] = useState("");
  const [state, setState] = useState("");
  const [postalCode, setPostalCode] = useState("");
  const [country, setCountry] = useState("US");

  // Climate
  const [climate, setClimate] = useState<string | null>(null);

  // Environment
  const [lightLevel, setLightLevel] = useState<string | null>(null);
  const [humidityLevel, setHumidityLevel] = useState<string | null>(null);
  const [tempMinF, setTempMinF] = useState<string>("");
  const [tempMaxF, setTempMaxF] = useState<string>("");

  // Safety
  const [hasKids, setHasKids] = useState<boolean | null>(null);

  // Constraints
  const [careLevel, setCareLevel] = useState<string | null>(null);
  const [preferredSize, setPreferredSize] = useState<string | null>(null);
  const [hardNo, setHardNo] = useState<string[]>([]);

  // Preferences
  const [wateringFreq, setWateringFreq] = useState<string | null>(null);
  const [careFreq, setCareFreq] = useState<string | null>(null);

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
        const { getMe } = await import("@/lib/api");
        const me = await getMe();
        if (me?.username) {
          setUsername(me.username);
          window.sessionStorage.setItem("userUsername", me.username);
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
        name: name || undefined,
        avatar_url: avatarUrl || undefined,
        city: city || undefined,
        state: state || undefined,
        postal_code: postalCode || undefined,
        country: country || undefined,
        climate: climate || undefined,
        light_level: lightLevel || undefined,
        humidity_level: humidityLevel || undefined,
        temp_min_f: tempMinF ? parseFloat(tempMinF) : undefined,
        temp_max_f: tempMaxF ? parseFloat(tempMaxF) : undefined,
        has_kids: hasKids,
        care_level: careLevel || undefined,
        preferred_size: preferredSize || undefined,
        hard_no: hardNo.length ? hardNo : undefined,
        watering_freq: wateringFreq || undefined,
        care_freq: careFreq || undefined,
      });
      const json = buildProfileJson(username, {
        name, avatarUrl,
        city, state, postalCode, country,
        climate,
        lightLevel, humidityLevel, tempMinF, tempMaxF,
        hasKids,
        careLevel, preferredSize, hardNo,
        wateringFreq, careFreq,
      });
      setExtractedJson(json);
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

  // Show extracted JSON after successful submit
  if (extractedJson) {
    return (
      <div className="min-h-screen px-4 py-8 bg-gradient-to-b from-sage-50 to-forest-50">
        <div className="max-w-2xl mx-auto">
          <Link href="/" className="inline-flex items-center gap-2 text-forest-700 hover:text-forest-800 mb-8">
            <span className="text-xl">🌿</span>
            <span className="font-medium">How to Keep Your Plants Alive</span>
          </Link>
          <div className="bg-white/80 backdrop-blur rounded-2xl shadow-leaf border border-sage-200/60 p-8">
            <h1 className="text-xl font-semibold text-forest-800 mb-2">Your profile (extracted)</h1>
            <p className="text-sm text-forest-600 mb-4">Here&apos;s your info in the schema format:</p>
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
          <h1 className="text-xl font-semibold text-forest-800 mb-2">Tell us about your space</h1>
          <p className="text-sm text-forest-600 mb-6">We&apos;ll use this to recommend the right plants for you.</p>

          {/* Step indicator */}
          <div className="flex gap-2 mb-6">
            {[1, 2, 3, 4].map((s) => (
              <div
                key={s}
                className={`h-1.5 flex-1 rounded-full ${step >= s ? "bg-forest-600" : "bg-sage-200"}`}
              />
            ))}
          </div>

          <form onSubmit={handleSubmit}>
            {step === 1 && (
              <>
                <div className={sectionCls}>
                  <label className={labelCls}>Name (optional)</label>
                  <input type="text" value={name} onChange={(e) => setName(e.target.value)} className={inputCls} placeholder="Your name" />
                </div>
                <div className={sectionCls}>
                  <label className={labelCls}>Avatar URL (optional)</label>
                  <input type="url" value={avatarUrl} onChange={(e) => setAvatarUrl(e.target.value)} className={inputCls} placeholder="https://..." />
                </div>
                <div className={sectionCls}>
                  <label className={labelCls}>City</label>
                  <input type="text" value={city} onChange={(e) => setCity(e.target.value)} className={inputCls} placeholder="City" required />
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className={labelCls}>State</label>
                    <input type="text" value={state} onChange={(e) => setState(e.target.value)} className={inputCls} placeholder="State" required />
                  </div>
                  <div>
                    <label className={labelCls}>Postal code</label>
                    <input type="text" value={postalCode} onChange={(e) => setPostalCode(e.target.value)} className={inputCls} placeholder="ZIP" required />
                  </div>
                </div>
                <div className={sectionCls}>
                  <label className={labelCls}>Country</label>
                  <input type="text" value={country} onChange={(e) => setCountry(e.target.value)} className={inputCls} placeholder="US" />
                </div>
                <div className={sectionCls}>
                  <label className={labelCls}>Climate</label>
                  <CardSelect options={CLIMATE_OPTIONS} value={climate} onChange={(v) => setClimate(v as string)} />
                </div>
              </>
            )}

            {step === 2 && (
              <>
                <div className={sectionCls}>
                  <label className={labelCls}>Light level</label>
                  <LightCards value={lightLevel} onChange={setLightLevel} />
                </div>
                <div className={sectionCls}>
                  <label className={labelCls}>Humidity level (low / med / high)</label>
                  <div className="flex flex-wrap gap-2">
                    {HUMIDITY_OPTIONS.map((opt) => (
                      <button
                        key={opt.id}
                        type="button"
                        onClick={() => setHumidityLevel(opt.id)}
                        className={`p-3 rounded-lg border-2 text-left transition-all ${humidityLevel === opt.id ? "border-forest-600 bg-forest-50" : "border-sage-200"}`}
                      >
                        <span className="font-medium text-forest-800">{opt.label}</span>
                        <span className="block text-xs text-forest-600">{opt.desc}</span>
                      </button>
                    ))}
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className={labelCls}>Min temp °F (optional)</label>
                    <input type="number" value={tempMinF} onChange={(e) => setTempMinF(e.target.value)} className={inputCls} placeholder="65" />
                  </div>
                  <div>
                    <label className={labelCls}>Max temp °F (optional)</label>
                    <input type="number" value={tempMaxF} onChange={(e) => setTempMaxF(e.target.value)} className={inputCls} placeholder="78" />
                  </div>
                </div>
                <div className={sectionCls}>
                  <label className={labelCls}>Do you have kids?</label>
                  <YesNo value={hasKids} onChange={setHasKids} />
                </div>
              </>
            )}

            {step === 3 && (
              <>
                <div className={sectionCls}>
                  <label className={labelCls}>Care level (matches plant difficulty: easy / medium / hard)</label>
                  <div className="flex flex-wrap gap-2">
                    {CARE_LEVEL_OPTIONS.map((opt) => (
                      <button
                        key={opt.id}
                        type="button"
                        onClick={() => setCareLevel(opt.id)}
                        className={`p-3 rounded-lg border-2 text-left transition-all ${careLevel === opt.id ? "border-forest-600 bg-forest-50" : "border-sage-200"}`}
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
                        className={`p-3 rounded-lg border-2 text-left transition-all ${preferredSize === opt.id ? "border-forest-600 bg-forest-50" : "border-sage-200"}`}
                      >
                        <span className="font-medium text-forest-800">{opt.label}</span>
                        <span className="block text-xs text-forest-600">{opt.desc}</span>
                      </button>
                    ))}
                  </div>
                </div>
                <div className={sectionCls}>
                  <label className={labelCls}>Hard no (optional)</label>
                  <CardSelect options={HARD_NO_OPTIONS} value={hardNo} onChange={(v) => setHardNo(v as string[])} multi />
                </div>
              </>
            )}

            {step === 4 && (
              <>
                <div className={sectionCls}>
                  <label className={labelCls}>Watering frequency (optional: low / medium / high)</label>
                  <CardSelect options={[{ id: "low", label: "Low" }, { id: "medium", label: "Medium" }, { id: "high", label: "High" }]} value={wateringFreq} onChange={(v) => setWateringFreq(v as string)} />
                </div>
                <div className={sectionCls}>
                  <label className={labelCls}>Care frequency (optional: low / medium / high)</label>
                  <CardSelect options={[{ id: "low", label: "Low" }, { id: "medium", label: "Medium" }, { id: "high", label: "High" }]} value={careFreq} onChange={(v) => setCareFreq(v as string)} />
                </div>
              </>
            )}

            <div className="flex gap-3 mt-8">
              {step > 1 && (
                <button type="button" onClick={() => setStep((s) => s - 1)} className="flex-1 py-3 rounded-lg border-2 border-sage-300 text-forest-700 font-medium">
                  Back
                </button>
              )}
              {step < 4 ? (
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
