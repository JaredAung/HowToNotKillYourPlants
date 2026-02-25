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

export async function getRecommendations(options?: { k?: number }) {
  const token = getToken();
  if (!token) throw new Error("Not logged in");
  const params = new URLSearchParams();
  if (options?.k !== undefined) params.set("k", String(options.k));
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
  return res.json();
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
