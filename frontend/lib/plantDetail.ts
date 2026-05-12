/**
 * GET /plant/{id} — matches backend ``plant.plant`` + Permapeople-style catalog flattening.
 */

export type PlantDetail = {
  plant_id: number;
  img_url?: string;
  latin?: string;
  common_name?: string;
  category?: string;
  origin?: string;
  size?: string;
  growth_rate?: string;
  physical_desc?: string;
  symbolism?: string;
  sunlight_type?: string;
  ideal_light?: string;
  tolerated_light?: string;
  humidity?: string;
  humidity_req?: string;
  care_level?: string;
  water_req?: string;
  water_req_raw?: string;
  temp_min?: number;
  temp_max?: number;
  /** Backend-combined temp / USDA / "cool" note from ``environment_care``. */
  temperature_display?: string;
  climate?: string;
  soil_type?: string;
  drainage_level?: string;
  bugs?: string[];
  disease?: string[];
  /** Full Mongo document (ids and embeddings stripped) for the catalog tree. */
  catalog?: Record<string, unknown>;
};

function toNum(v: unknown): number | undefined {
  if (typeof v === "number" && Number.isFinite(v)) return v;
  if (typeof v === "string" && v.trim() !== "") {
    const n = Number(v);
    if (Number.isFinite(n)) return n;
  }
  return undefined;
}

function optStr(o: Record<string, unknown>, k: string): string | undefined {
  const v = o[k];
  if (v == null) return undefined;
  const s = String(v).trim();
  return s || undefined;
}

function optStrArr(o: Record<string, unknown>, k: string): string[] | undefined {
  const v = o[k];
  if (!Array.isArray(v)) return undefined;
  const xs = v.filter((x): x is string => typeof x === "string" && x.trim() !== "");
  return xs.length ? xs : undefined;
}

/** Coerce JSON from GET /plant/{id} into a typed detail object (handles loose/nullable fields). */
export function normalizePlantDetail(raw: unknown): PlantDetail | null {
  if (!raw || typeof raw !== "object") return null;
  const o = raw as Record<string, unknown>;
  const plant_id = toNum(o.plant_id);
  if (plant_id === undefined) return null;

  const d: PlantDetail = { plant_id };

  const strKeys: (keyof PlantDetail)[] = [
    "img_url",
    "latin",
    "common_name",
    "category",
    "origin",
    "size",
    "growth_rate",
    "physical_desc",
    "symbolism",
    "sunlight_type",
    "ideal_light",
    "tolerated_light",
    "humidity",
    "humidity_req",
    "care_level",
    "water_req",
    "water_req_raw",
    "temperature_display",
    "climate",
    "soil_type",
    "drainage_level",
  ];
  for (const k of strKeys) {
    const s = optStr(o, k as string);
    if (s !== undefined) (d as Record<string, unknown>)[k as string] = s;
  }

  const tmin = toNum(o.temp_min);
  const tmax = toNum(o.temp_max);
  if (tmin !== undefined) d.temp_min = tmin;
  if (tmax !== undefined) d.temp_max = tmax;

  const bugs = optStrArr(o, "bugs");
  const disease = optStrArr(o, "disease");
  if (bugs) d.bugs = bugs;
  if (disease) d.disease = disease;

  const cat = o.catalog;
  if (cat && typeof cat === "object" && !Array.isArray(cat)) {
    d.catalog = cat as Record<string, unknown>;
  }

  return d;
}
