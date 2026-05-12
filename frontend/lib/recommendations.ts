import type { PlantRec } from "@/app/components/PlantCard";

/** Aligns with GET /recommend/: `{ username, plants, top_recommended?, message? }`. */
export type RecommendationsResponse = {
  username?: string;
  /** Main grid: first page from explore queue (Redis) or tail of ranked list (non-Redis). */
  plants: PlantRec[];
  /** Best reranked picks (typically 5), shown above the grid when present. */
  top_recommended?: PlantRec[];
  /** When true, more plants can load via ``GET /recommend/explore`` (Redis deck only). */
  explore_has_more?: boolean;
  message?: string;
};

function toNum(v: unknown, fallback: number): number {
  if (typeof v === "number" && Number.isFinite(v)) return v;
  if (typeof v === "string" && v.trim() !== "") {
    const n = Number(v);
    if (Number.isFinite(n)) return n;
  }
  return fallback;
}

/** Normalize nested catalog fields into the flat keys the UI expects (matches backend ``flatten_catalog_plant_for_api``). */
function flattenMongoPlantShape(src: Record<string, unknown>): Record<string, unknown> {
  const o: Record<string, unknown> = { ...src };

  if (o.latin == null && o.scientific_name != null) o.latin = o.scientific_name;
  if (o.common_name == null && typeof o.name === "string") o.common_name = o.name;

  const pickImageUrl = (obj: Record<string, unknown>): unknown => {
    const imgs = obj.images;
    if (imgs && typeof imgs === "object" && !Array.isArray(imgs)) {
      const im = imgs as Record<string, unknown>;
      return im.thumb ?? im.thumbnail ?? im.title ?? im.primary ?? im.url;
    }
    return undefined;
  };

  if (o.img_url == null) {
    const u = pickImageUrl(o);
    if (typeof u === "string" && u.trim()) o.img_url = u;
  }

  const info = o.info as Record<string, unknown> | undefined;
  if (info && typeof info === "object") {
    if (o.latin == null && info.latin != null) o.latin = info.latin;
    if (o.common_name == null && info.common_name != null) o.common_name = info.common_name;
    if (o.img_url == null) {
      const u = pickImageUrl(info);
      if (typeof u === "string" && u.trim()) o.img_url = u;
    }
    const desc = info.desc as Record<string, unknown> | undefined;
    if (desc && typeof desc === "object") {
      if (o.physical_desc == null && desc.physical_desc != null) o.physical_desc = desc.physical_desc;
      if (o.symbolism == null && desc.symbolism != null) o.symbolism = desc.symbolism;
    }
    if (o.size == null && info.size != null) o.size = info.size;
    if (o.category == null && info.category != null) o.category = info.category;
  }

  const ec = o.environment_care as Record<string, unknown> | undefined;
  if (ec && typeof ec === "object") {
    const lighting = ec.lighting as Record<string, unknown> | undefined;
    if (lighting && typeof lighting === "object") {
      const il = lighting.ideal_light;
      const tl = lighting.tolerated_light;
      if (o.sunlight_type == null) {
        if (typeof il === "string" && il.trim()) o.sunlight_type = il;
        else if (typeof tl === "string" && tl.trim()) o.sunlight_type = tl;
      }
      if (o.ideal_light == null && typeof il === "string" && il.trim()) o.ideal_light = il;
      if (o.tolerated_light == null && typeof tl === "string" && tl.trim()) o.tolerated_light = tl;
    }
    if (o.humidity == null) {
      const h = ec["Humidity requirement"] ?? ec.humidity_req;
      if (h != null) o.humidity = h;
    }
    if (o.care_level == null && ec.care_level != null) o.care_level = ec.care_level;
    const infoFlat = o.info as Record<string, unknown> | undefined;
    if (o.difficulty_score == null && infoFlat && infoFlat.difficulty_score != null) {
      o.difficulty_score = infoFlat.difficulty_score;
    }
    if (o.water_req == null) {
      const water = ec.water as Record<string, unknown> | undefined;
      if (water && typeof water === "object") {
        const w = water.ideal_water ?? water.tolerated_water;
        if (w != null) o.water_req = w;
      }
      if (o.water_req == null && typeof ec["Water requirement"] === "string") {
        o.water_req = ec["Water requirement"];
      }
    }
    const tr = ec.temp_req as Record<string, unknown> | undefined;
    if (tr && typeof tr === "object") {
      if (o.temp_min == null && tr.min_temp != null) o.temp_min = tr.min_temp;
      if (o.temp_max == null && tr.max_temp != null) o.temp_max = tr.max_temp;
    }
    if (o.climate == null && ec.origin_climate != null) o.climate = ec.origin_climate;
  }

  return o;
}

/** Build a strict PlantRec from API JSON (handles string ids from loose serializers). */
export function coercePlantRec(raw: unknown): PlantRec | null {
  if (!raw || typeof raw !== "object") return null;
  const o = flattenMongoPlantShape(raw as Record<string, unknown>);
  if (Array.isArray(o.img_url) && o.img_url.length > 0 && typeof o.img_url[0] === "string") {
    o.img_url = o.img_url[0];
  }
  const plant_id = toNum(o.plant_id, NaN);
  if (!Number.isFinite(plant_id)) return null;

  const p: PlantRec = {
    plant_id,
    score: toNum(o.score, 0),
  };

  if (o.rerank_score != null) {
    const r = toNum(o.rerank_score, NaN);
    if (Number.isFinite(r)) p.rerank_score = r;
  }

  const optStr = (key: keyof PlantRec) => {
    const v = o[key as string];
    if (v == null) return;
    const s = String(v).trim();
    if (s) (p as Record<string, unknown>)[key as string] = s;
  };
  optStr("img_url");
  optStr("latin");
  optStr("common_name");
  optStr("sunlight_type");
  optStr("humidity");
  optStr("care_level");
  optStr("water_req");

  if (o.temp_min != null) {
    const t = toNum(o.temp_min, NaN);
    if (Number.isFinite(t)) p.temp_min = t;
  }
  if (o.temp_max != null) {
    const t = toNum(o.temp_max, NaN);
    if (Number.isFinite(t)) p.temp_max = t;
  }

  if (o.difficulty_score != null) {
    const d = toNum(o.difficulty_score, NaN);
    if (Number.isFinite(d)) p.difficulty_score = d;
  }

  return p;
}

/** Normalize any JSON body from GET /recommend/ for safe UI use. */
export function normalizeRecommendationsResponse(data: unknown): RecommendationsResponse {
  if (!data || typeof data !== "object") {
    return { plants: [] };
  }
  const d = data as Record<string, unknown>;
  const plants: PlantRec[] = [];
  if (Array.isArray(d.plants)) {
    for (const item of d.plants) {
      const c = coercePlantRec(item);
      if (c) plants.push(c);
    }
  }
  const top_recommended: PlantRec[] = [];
  if (Array.isArray(d.top_recommended)) {
    for (const item of d.top_recommended) {
      const c = coercePlantRec(item);
      if (c) top_recommended.push(c);
    }
  }
  const explore_has_more =
    typeof d.explore_has_more === "boolean" ? d.explore_has_more : undefined;

  return {
    username: typeof d.username === "string" ? d.username : undefined,
    plants,
    top_recommended: top_recommended.length > 0 ? top_recommended : undefined,
    explore_has_more,
    message: typeof d.message === "string" ? d.message : undefined,
  };
}
