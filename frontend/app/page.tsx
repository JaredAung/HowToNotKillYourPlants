"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import {
  cacheHomeRecommendations,
  getExplanation,
  getRecommendations,
  getRecommendationsExplore,
  getToken,
  type RecommendationsResponse,
} from "@/lib/api";
import { useRouter } from "next/navigation";
import { PlantCard, type PlantRec } from "@/app/components/PlantCard";
import { ExplanationDisplay } from "@/app/components/ExplanationDisplay";
import { setChatContext } from "@/lib/chatContext";
import { addDirectlyToGarden } from "@/lib/addToGarden";

export default function Home() {
  const router = useRouter();
  const [loading, setLoading] = useState(true);
  const [recommendations, setRecommendations] = useState<RecommendationsResponse | null>(null);
  const [explanationOn, setExplanationOn] = useState(false);
  const [explanation, setExplanation] = useState<string | null>(null);
  const [explanationLoading, setExplanationLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [addSuccessPlantId, setAddSuccessPlantId] = useState<number | null>(null);
  const [loadMoreLoading, setLoadMoreLoading] = useState(false);

  const fetchRecs = (forceRefresh = false) => {
    const token = getToken();
    if (!token) {
      setLoading(false);
      return;
    }
    setLoading(true);
    setError(null);
    getRecommendations({ forceRefresh })
      .then((data) => setRecommendations(data))
      .catch((err) => setError(err instanceof Error ? err.message : "Failed to load"))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    fetchRecs();
  }, []);

  useEffect(() => {
    if (addSuccessPlantId == null) return;
    const t = setTimeout(() => setAddSuccessPlantId(null), 2000);
    return () => clearTimeout(t);
  }, [addSuccessPlantId]);

  useEffect(() => {
    if (!explanationOn) {
      setExplanation(null);
      return;
    }
    const top5 = (recommendations?.top_recommended ?? []).slice(0, 5);
    if (top5.length === 0) {
      setExplanation(null);
      setExplanationLoading(false);
      return;
    }
    setExplanationLoading(true);
    const top5Ids = top5.map((p) => p.plant_id);
    getExplanation(top5Ids)
      .then((res) => setExplanation(res.explanation ?? ""))
      .catch(() => setExplanation(""))
      .finally(() => setExplanationLoading(false));
  }, [explanationOn, recommendations?.top_recommended]);

  const isLoggedIn = !!getToken();
  const topRecommended = recommendations?.top_recommended ?? [];
  const plants = recommendations?.plants ?? [];
  const hasRecs = topRecommended.length > 0 || plants.length > 0;
  const showGridRefresh = plants.length > 0;
  const showHeaderRefresh = hasRecs && !showGridRefresh;
  const chatContextPlants: PlantRec[] =
    topRecommended.length > 0 ? topRecommended.slice(0, 5) : plants.slice(0, 5);

  const exploreHasMore = recommendations?.explore_has_more === true;

  const handleLoadMoreExplore = async () => {
    if (!recommendations || loadMoreLoading) return;
    setLoadMoreLoading(true);
    setError(null);
    try {
      const next = await getRecommendationsExplore(plants.length, 20);
      const merged: RecommendationsResponse = {
        ...recommendations,
        plants: [...plants, ...next.plants],
        explore_has_more: next.has_more,
      };
      setRecommendations(merged);
      cacheHomeRecommendations(merged);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load more");
    } finally {
      setLoadMoreLoading(false);
    }
  };

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
                    {showHeaderRefresh && (
                    <button
                      type="button"
                      onClick={() => fetchRecs(true)}
                      disabled={loading}
                      className="text-sm text-forest-600 hover:text-forest-800 underline disabled:opacity-50"
                    >
                      Refresh
                    </button>
                    )}
                    {hasRecs && (
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
                {hasRecs && explanationOn && (
                  <div className="w-full max-w-2xl mx-auto rounded-xl border border-sage-200 bg-white shadow-leaf p-5">
                    <h3 className="text-base font-semibold text-forest-800 mb-3 flex items-center gap-2">
                      <span className="text-lg">🌱</span>
                      Why these plants?
                    </h3>
                    <ExplanationDisplay explanation={explanation} loading={explanationLoading} />
                  </div>
                )}
                {recommendations.message && !hasRecs ? (
                  <p className="text-forest-600 text-sm">{recommendations.message}</p>
                ) : hasRecs ? (
                  <>
                    {topRecommended.length > 0 && (
                      <section className="w-full flex flex-col gap-4">
                        <h3 className="text-base font-semibold text-forest-800">
                          Top recommended
                        </h3>
                        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
                          {topRecommended.map((p) => {
                            return (
                              <PlantCard
                                key={p.plant_id}
                                p={p}
                                isJustAdded={addSuccessPlantId === p.plant_id}
                                onAdd={async (plant) => {
                                  try {
                                    setError(null);
                                    await addDirectlyToGarden(plant);
                                    setAddSuccessPlantId(plant.plant_id);
                                  } catch (err) {
                                    setError(err instanceof Error ? err.message : "Failed to add");
                                  }
                                }}
                                onTalkToAgent={(plant) => {
                                  setChatContext(plant, chatContextPlants);
                                  router.push("/chat");
                                }}
                              />
                            );
                          })}
                        </div>
                      </section>
                    )}
                    {plants.length > 0 && (
                      <section
                        className={`w-full flex flex-col gap-4 ${
                          topRecommended.length > 0 ? "mt-10" : ""
                        }`}
                      >
                        <div className="flex items-center justify-between gap-4 flex-wrap">
                          <h3 className="text-base font-semibold text-forest-800">
                            More picks
                          </h3>
                          <button
                            type="button"
                            onClick={() => fetchRecs(true)}
                            disabled={loading}
                            className="text-sm text-forest-600 hover:text-forest-800 underline disabled:opacity-50 shrink-0"
                          >
                            Refresh
                          </button>
                        </div>
                        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
                          {plants.map((p) => {
                            return (
                              <PlantCard
                                key={p.plant_id}
                                p={p}
                                isJustAdded={addSuccessPlantId === p.plant_id}
                                onAdd={async (plant) => {
                                  try {
                                    setError(null);
                                    await addDirectlyToGarden(plant);
                                    setAddSuccessPlantId(plant.plant_id);
                                  } catch (err) {
                                    setError(err instanceof Error ? err.message : "Failed to add");
                                  }
                                }}
                                onTalkToAgent={(plant) => {
                                  setChatContext(plant, chatContextPlants);
                                  router.push("/chat");
                                }}
                              />
                            );
                          })}
                        </div>
                        {exploreHasMore && (
                          <div className="flex justify-center pt-2">
                            <button
                              type="button"
                              onClick={() => void handleLoadMoreExplore()}
                              disabled={loadMoreLoading}
                              className="px-5 py-2.5 rounded-lg border border-sage-300 bg-white text-sm font-medium text-forest-800 hover:bg-sage-50 disabled:opacity-50"
                            >
                              {loadMoreLoading ? "Loading…" : "Load more"}
                            </button>
                          </div>
                        )}
                      </section>
                    )}
                  </>
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
