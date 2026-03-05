const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const TOKEN_KEY = "auth_token";

export function getToken(): string | null {
  if (typeof window === "undefined") return null;
  return sessionStorage.getItem(TOKEN_KEY);
}

export function setToken(token: string) {
  if (typeof window !== "undefined") sessionStorage.setItem(TOKEN_KEY, token);
}

export function clearToken() {
  if (typeof window !== "undefined") sessionStorage.removeItem(TOKEN_KEY);
}

export async function logOut() {
  const token = getToken();
  if (!token) return;
  try {
    await fetch(`${API_BASE}/auth/logout`, {
      method: "POST",
      headers: { Authorization: `Bearer ${token}` },
    });
  } finally {
    clearToken();
  }
}

export async function signUp(username: string, password: string, email?: string) {
  const res = await fetch(`${API_BASE}/auth/signup`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password, email: email || undefined }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    const msg = Array.isArray(err.detail) ? err.detail[0]?.msg : err.detail;
    throw new Error(msg || "Sign up failed");
  }
  const data = await res.json();
  if (data?.token) setToken(data.token);
  return data;
}

export async function logIn(username: string, password: string) {
  const res = await fetch(`${API_BASE}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    const msg = Array.isArray(err.detail) ? err.detail[0]?.msg : err.detail;
    throw new Error(msg || "Login failed");
  }
  const data = await res.json();
  if (data?.token) setToken(data.token);
  return data;
}

export async function getMe() {
  const token = getToken();
  if (!token) throw new Error("Not logged in");
  const res = await fetch(`${API_BASE}/auth/me`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  if (!res.ok) {
    if (res.status === 401) {
      clearToken();
      throw new Error("Session expired");
    }
    throw new Error("Failed to get user");
  }
  return res.json();
}

const HOME_REC_CACHE_KEY = "homeRecommendationsCache";

function getCachedRecommendations(): Record<string, unknown> | null {
  if (typeof window === "undefined") return null;
  try {
    const stored = sessionStorage.getItem(HOME_REC_CACHE_KEY);
    if (!stored) return null;
    const parsed = JSON.parse(stored) as Record<string, unknown>;
    const plants = parsed?.plants as unknown[] | undefined;
    if (!plants || plants.length === 0) return null;
    return parsed;
  } catch {
    sessionStorage.removeItem(HOME_REC_CACHE_KEY);
    return null;
  }
}

function setCachedRecommendations(data: Record<string, unknown>) {
  if (typeof window === "undefined") return;
  try {
    sessionStorage.setItem(HOME_REC_CACHE_KEY, JSON.stringify(data));
  } catch {
    sessionStorage.removeItem(HOME_REC_CACHE_KEY);
  }
}

export function clearRecommendationsCache() {
  if (typeof window === "undefined") return;
  sessionStorage.removeItem(HOME_REC_CACHE_KEY);
}

export async function getRecommendations(options?: {
  k?: number;
  use_rerank?: boolean;
  forceRefresh?: boolean;
}) {
  const token = getToken();
  if (!token) throw new Error("Not logged in");

  // Use cached recs when available; only call API when cache is empty or forceRefresh
  if (!options?.forceRefresh) {
    const cached = getCachedRecommendations();
    if (cached) return cached;
  } else {
    clearRecommendationsCache();
  }

  const params = new URLSearchParams();
  if (options?.k !== undefined) params.set("k", String(options.k));
  if (options?.use_rerank === false) params.set("use_rerank", "false");
  const qs = params.toString();
  const url = `${API_BASE}/recommend/${qs ? `?${qs}` : ""}`;
  const res = await fetch(url, {
    headers: { Authorization: `Bearer ${token}` },
  });
  if (!res.ok) {
    if (res.status === 401) {
      clearToken();
      throw new Error("Session expired");
    }
    const err = await res.json().catch(() => ({}));
    const msg = Array.isArray(err.detail) ? err.detail : err.detail;
    throw new Error(typeof msg === "string" ? msg : "Failed to load recommendations");
  }
  const data = (await res.json()) as Record<string, unknown>;
  setCachedRecommendations(data);
  return data;
}

export async function getExplanation(plantIds: number[]) {
  const token = getToken();
  if (!token) throw new Error("Not logged in");
  const ids = plantIds.slice(0, 5).join(",");
  const res = await fetch(`${API_BASE}/recommend/explanation?plant_ids=${encodeURIComponent(ids)}`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  if (!res.ok) {
    if (res.status === 401) {
      clearToken();
      throw new Error("Session expired");
    }
    const err = await res.json().catch(() => ({}));
    throw new Error(err?.detail || "Failed to load explanation");
  }
  return res.json();
}

export async function getProfile() {
  const token = getToken();
  if (!token) throw new Error("Not logged in");
  const res = await fetch(`${API_BASE}/profile/`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  if (!res.ok) {
    if (res.status === 401) {
      clearToken();
      throw new Error("Session expired");
    }
    throw new Error("Failed to get profile");
  }
  return res.json();
}

export async function getPlant(plantId: number) {
  const token = getToken();
  if (!token) throw new Error("Not logged in");
  const res = await fetch(`${API_BASE}/plant/${plantId}`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  if (!res.ok) {
    if (res.status === 401) {
      clearToken();
      throw new Error("Session expired");
    }
    if (res.status === 404) throw new Error("Plant not found");
    throw new Error("Failed to load plant");
  }
  return res.json();
}

export async function extractSearchProfile(text: string) {
  const token = getToken();
  if (!token) throw new Error("Not logged in");
  const res = await fetch(`${API_BASE}/search/extract`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`,
    },
    body: JSON.stringify({ text }),
  });
  if (!res.ok) {
    if (res.status === 401) {
      clearToken();
      throw new Error("Session expired");
    }
    const err = await res.json().catch(() => ({}));
    throw new Error(err?.detail || "Failed to extract profile");
  }
  return res.json();
}

export async function addToGarden(plantId: number, customName?: string) {
  const token = getToken();
  if (!token) throw new Error("Not logged in");
  const res = await fetch(`${API_BASE}/garden/add`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`,
    },
    body: JSON.stringify({ plant_id: plantId, custom_name: customName || undefined }),
  });
  if (!res.ok) {
    if (res.status === 401) {
      clearToken();
      throw new Error("Session expired");
    }
    const err = await res.json().catch(() => ({}));
    const msg = err?.detail;
    if (res.status === 404) {
      throw new Error(typeof msg === "string" ? msg : "Garden service unavailable. Is the backend running?");
    }
    throw new Error(typeof msg === "string" ? msg : "Failed to add to garden");
  }
  return res.json();
}

export async function getGarden() {
  const token = getToken();
  if (!token) throw new Error("Not logged in");
  const res = await fetch(`${API_BASE}/garden/`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  if (!res.ok) {
    if (res.status === 401) {
      clearToken();
      throw new Error("Session expired");
    }
    throw new Error("Failed to load garden");
  }
  return res.json();
}

export type DeathReportPayload = {
  plant_id: number;
  what_happened: string[];
  watering_frequency?: string;
  plant_location?: string;
  humidity_level?: string;
  room_temperature?: string;
  death_reason?: string;
  plant_profile?: Record<string, unknown>;
  user_profile?: Record<string, unknown>;
};

export async function markPlantDead(payload: DeathReportPayload) {
  const token = getToken();
  if (!token) throw new Error("Not logged in");
  const res = await fetch(`${API_BASE}/garden/death`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`,
    },
    body: JSON.stringify({
      plant_id: payload.plant_id,
      what_happened: payload.what_happened,
      watering_frequency: payload.watering_frequency ?? null,
      plant_location: payload.plant_location ?? null,
      humidity_level: payload.humidity_level ?? null,
      room_temperature: payload.room_temperature ?? null,
      death_reason: payload.death_reason ?? null,
      plant_profile: payload.plant_profile ?? null,
      user_profile: payload.user_profile ?? null,
    }),
  });
  if (!res.ok) {
    if (res.status === 401) {
      clearToken();
      throw new Error("Session expired");
    }
    const err = await res.json().catch(() => ({}));
    const msg = err?.detail;
    throw new Error(typeof msg === "string" ? msg : "Failed to mark plant as dead");
  }
  return res.json();
}

export type ChatMessage = { role: "user" | "assistant"; content: string };

export type InvokeChatOptions = {
  messages: ChatMessage[];
  selectedPlant?: Record<string, unknown>;
  recommendedPlants?: Record<string, unknown>[];
};

export async function invokeChat(options: InvokeChatOptions) {
  const token = getToken();
  if (!token) throw new Error("Not logged in");
  const res = await fetch(`${API_BASE}/chat/invoke`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`,
    },
    body: JSON.stringify({
      messages: options.messages,
      selected_plant: options.selectedPlant,
      recommended_plants: options.recommendedPlants,
    }),
  });
  if (!res.ok) {
    if (res.status === 401) {
      clearToken();
      throw new Error("Session expired");
    }
    const err = await res.json().catch(() => ({}));
    throw new Error(err?.detail || "Chat failed");
  }
  return res.json();
}

export async function updateProfile(data: Record<string, unknown>) {
  const token = getToken();
  if (!token) throw new Error("Not logged in");
  const res = await fetch(`${API_BASE}/profile/update`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`,
    },
    body: JSON.stringify(data),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    const msg = Array.isArray(err.detail) ? err.detail[0]?.msg : err.detail;
    throw new Error(msg || "Profile update failed");
  }
  return res.json();
}
