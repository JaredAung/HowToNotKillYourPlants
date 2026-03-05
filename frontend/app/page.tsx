"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { getExplanation, getRecommendations, getToken } from "@/lib/api";
import { useRouter } from "next/navigation";
import { PlantCard, type PlantRec } from "@/app/components/PlantCard";
import { ExplanationDisplay } from "@/app/components/ExplanationDisplay";
import { setChatContext } from "@/lib/chatContext";

type RecResponse = { username?: string; plants?: PlantRec[]; message?: string; explanation?: string };

export default function Home() {
  const router = useRouter();
  const [loading, setLoading] = useState(true);
  const [recommendations, setRecommendations] = useState<RecResponse | null>(null);
  const [explanationOn, setExplanationOn] = useState(false);
  const [explanation, setExplanation] = useState<string | null>(null);
  const [explanationLoading, setExplanationLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchRecs = (forceRefresh = false) => {
    const token = getToken();
    if (!token) {
      setLoading(false);
      return;
    }
    setLoading(true);
    setError(null);
    getRecommendations({ use_rerank: false, forceRefresh })
      .then((data) => setRecommendations(data as RecResponse))
      .catch((err) => setError(err instanceof Error ? err.message : "Failed to load"))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    fetchRecs();
  }, []);

  useEffect(() => {
    if (!explanationOn) {
      setExplanation(null);
      return;
    }
    const plants = recommendations?.plants ?? [];
    if (plants.length === 0) return;
    setExplanationLoading(true);
    const top5Ids = plants.slice(0, 5).map((p) => p.plant_id);
    getExplanation(top5Ids)
      .then((res) => setExplanation(res.explanation ?? ""))
      .catch(() => setExplanation(""))
      .finally(() => setExplanationLoading(false));
  }, [explanationOn, recommendations?.plants]);

  const isLoggedIn = !!getToken();
  const plants = recommendations?.plants ?? [];

  return (
    <div className="min-h-screen flex flex-col px-4 py-8 bg-gradient-to-b from-sage-50 to-forest-50">
      <main className="flex flex-col items-center gap-8 max-w-4xl mx-auto w-full">
        <div className="flex items-center gap-3">
          <span className="text-4xl">🌿</span>
          <h1 className="text-2xl font-semibold text-forest-800">
            How to Keep Your Plants Alive
          </h1>
        </div>
        <p className="text-forest-600 leading-relaxed text-center">
          Track your plants, get care reminders, and never forget to water again.
        </p>

        {loading ? (
          <p className="text-forest-600">Loading...</p>
        ) : isLoggedIn ? (
          <div className="flex flex-col gap-6 w-full">
            {error && (
              <p className="text-sm text-rose-600">{error}</p>
            )}
            {recommendations && (
              <>
                <div className="flex items-center justify-between gap-4 flex-wrap">
                  <h2 className="text-lg font-semibold text-forest-800">
                    For you, {recommendations.username ?? "you"}
                  </h2>
                  <div className="flex items-center gap-3">
                    {plants.length > 0 && (
                    <button
                      type="button"
                      onClick={() => fetchRecs(true)}
                      disabled={loading}
                      className="text-sm text-forest-600 hover:text-forest-800 underline disabled:opacity-50"
                    >
                      Refresh
                    </button>
                    )}
                    {plants.length > 0 && (
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
                    )}
                  </div>
                </div>
                {plants.length > 0 && explanationOn && (
                  <div className="w-full max-w-2xl mx-auto rounded-xl border border-sage-200 bg-white shadow-leaf p-5">
                    <h3 className="text-base font-semibold text-forest-800 mb-3 flex items-center gap-2">
                      <span className="text-lg">🌱</span>
                      Why these plants?
                    </h3>
                    <ExplanationDisplay explanation={explanation} loading={explanationLoading} />
                  </div>
                )}
                {recommendations.message ? (
                  <p className="text-forest-600 text-sm">{recommendations.message}</p>
                ) : plants.length > 0 ? (
                  <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
                    {plants.map((p) => {
                      const matchPct =
                        p.rerank_score != null
                          ? Math.round(p.rerank_score * 100)
                          : Math.round(((plants[0]?.score ?? 1) > 0 ? p.score / (plants[0]?.score ?? 1) : 0) * 100);
                      return <PlantCard key={p.plant_id} p={p} matchPct={matchPct} onPick={(plant) => { setChatContext(plant, plants.slice(0, 5)); router.push("/chat"); }} />;
                    })}
                  </div>
                ) : (
                  <p className="text-forest-600 text-sm">
                    Complete your profile to get personalized plant recommendations.
                  </p>
                )}
              </>
            )}
          </div>
        ) : (
          <div className="flex flex-col sm:flex-row gap-4 w-full sm:w-auto">
            <Link
              href="/auth"
              className="px-6 py-3 rounded-lg bg-forest-600 text-white font-medium hover:bg-forest-700 transition-colors text-center"
            >
              Sign in / Log in
            </Link>
          </div>
        )}
      </main>
    </div>
  );
}
