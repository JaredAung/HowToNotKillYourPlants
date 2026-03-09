"use client";

import { useState } from "react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { extractSearchProfile, getToken, logOut, semanticSearch } from "@/lib/api";

const SEARCH_RESULTS_KEY = "searchExtractedProfile";
const SEARCH_PLANTS_KEY = "searchExtractedPlants";
const SEARCH_SEMANTIC_PLANTS_KEY = "searchSemanticPlants";

const navItems = [
  { href: "/profile", label: "Profile" },
  { href: "/garden", label: "Garden" },
  { href: "/agent", label: "Agent" },
];

export default function Navbar() {
  const pathname = usePathname();
  const router = useRouter();
  const isLoggedIn = !!getToken();
  const [searchOpen, setSearchOpen] = useState(false);
  const [searchText, setSearchText] = useState("");
  const [searchLoading, setSearchLoading] = useState(false);
  const [searchError, setSearchError] = useState<string | null>(null);

  const handleSearchSubmit = async () => {
    if (!searchText.trim()) return;
    setSearchLoading(true);
    setSearchError(null);
    const text = searchText.trim();
    try {
      const [extractRes, semanticRes] = await Promise.all([
        extractSearchProfile(text),
        semanticSearch(text, 20),
      ]);
      if (typeof window !== "undefined") {
        sessionStorage.setItem(SEARCH_RESULTS_KEY, JSON.stringify(extractRes.profile));
        sessionStorage.setItem(SEARCH_PLANTS_KEY, JSON.stringify(extractRes.plants ?? []));
        sessionStorage.setItem(SEARCH_SEMANTIC_PLANTS_KEY, JSON.stringify(semanticRes.plants ?? []));
      }
      setSearchOpen(false);
      setSearchText("");
      router.push("/search/results");
    } catch (err) {
      setSearchError(err instanceof Error ? err.message : "Search failed");
    } finally {
      setSearchLoading(false);
    }
  };

  if (!isLoggedIn) return null;

  return (
    <nav className="sticky top-0 z-50 w-full border-b border-sage-200 bg-white/95 backdrop-blur shadow-sm">
      <div className="mx-auto flex h-14 max-w-4xl items-center justify-between px-4">
        <Link
          href="/"
          className="flex items-center gap-2 text-forest-800 font-semibold hover:text-forest-600 transition-colors"
        >
          <span className="text-xl">🌿</span>
          <span className="hidden sm:inline">How to Keep Your Plants Alive</span>
        </Link>
        <div className="flex items-center gap-1 sm:gap-2">
          <div className="relative">
            <button
              type="button"
              onClick={() => setSearchOpen((o) => !o)}
              className={`px-3 py-2 rounded-lg text-sm font-medium transition-colors flex items-center gap-1.5 ${
                searchOpen
                  ? "bg-forest-100 text-forest-800"
                  : "text-forest-600 hover:bg-sage-100 hover:text-forest-800"
              }`}
            >
              <span>🔍</span>
              Search
            </button>
            {searchOpen && (
              <>
                <div
                  className="fixed inset-0 z-40"
                  aria-hidden="true"
                  onClick={() => setSearchOpen(false)}
                />
                <div className="absolute right-0 top-full mt-1 z-50 w-80 rounded-xl border border-sage-200 bg-white shadow-leaf p-3">
                  <textarea
                    placeholder="Describe what you want: appearance (e.g. tall palm, glossy leaves), symbolism (e.g. peace, good luck), or care needs (light, humidity...)"
                    rows={4}
                    value={searchText}
                    onChange={(e) => setSearchText(e.target.value)}
                    className="w-full resize-y min-h-[80px] max-h-48 px-3 py-2 rounded-lg border border-sage-200 text-forest-800 text-sm placeholder:text-sage-400 focus:outline-none focus:ring-2 focus:ring-sage-400 focus:border-transparent"
                    autoFocus
                  />
                  {searchError && (
                    <p className="text-xs text-rose-600 mt-1">{searchError}</p>
                  )}
                  <button
                    type="button"
                    onClick={handleSearchSubmit}
                    disabled={searchLoading || !searchText.trim()}
                    className="mt-2 w-full py-2 rounded-lg bg-forest-600 text-white text-sm font-medium hover:bg-forest-700 disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {searchLoading ? "Searching..." : "Search"}
                  </button>
                </div>
              </>
            )}
          </div>
          {navItems.map((item) => {
            const isActive = pathname === item.href || pathname.startsWith(item.href + "/");
            return (
              <Link
                key={item.href}
                href={item.href}
                className={`px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                  isActive
                    ? "bg-forest-100 text-forest-800"
                    : "text-forest-600 hover:bg-sage-100 hover:text-forest-800"
                }`}
              >
                {item.label}
              </Link>
            );
          })}
          <button
            type="button"
            onClick={async () => {
              await logOut();
              if (typeof window !== "undefined") sessionStorage.removeItem("userUsername");
              router.push("/");
            }}
            className="px-3 py-2 rounded-lg text-sm font-medium text-forest-600 hover:bg-sage-100 hover:text-forest-800 transition-colors"
          >
            Log out
          </button>
        </div>
      </div>
    </nav>
  );
}
