"use client";

import type { ReactNode } from "react";

/** Lowercase words, then capitalize the first letter of each word (labels). */
function formatKeyLabel(s: string): string {
  const spaced = s.replace(/_/g, " ").trim();
  if (!spaced) return "";
  return spaced
    .split(/\s+/)
    .map((w) => (w.length === 0 ? w : w.charAt(0).toUpperCase() + w.slice(1).toLowerCase()))
    .join(" ");
}

/** Sentence-style: capitalize only the first character of the string. */
export function capitalizeFirstLetter(text: string): string {
  const t = text.trim();
  if (!t) return t;
  return t.charAt(0).toUpperCase() + t.slice(1);
}

/** Keys removed everywhere in the catalog tree (display only). */
const STRIP_FROM_CATALOG = new Set([
  "images",
  "link",
  "created_at",
  "updated_at",
  "life_cycle_req",
  "layer_req",
]);

/** Remove strip-list keys at every object level; recurse into arrays and nested objects. */
export function omitKeysDeep(value: unknown, keys: Set<string>): unknown {
  if (Array.isArray(value)) {
    return value.map((x) => omitKeysDeep(x, keys));
  }
  if (value !== null && typeof value === "object") {
    const o = value as Record<string, unknown>;
    const out: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(o)) {
      if (keys.has(k)) continue;
      out[k] = omitKeysDeep(v, keys);
    }
    return out;
  }
  return value;
}

export type NameBlock = {
  common?: string;
  slug?: string;
  scientific_name?: string;
};

export type OverviewBlock = {
  difficulty_score?: number;
  care_level?: string;
};

function parseDifficultyScore(raw: unknown): number | undefined {
  if (typeof raw === "number" && Number.isFinite(raw)) return raw;
  if (typeof raw === "string" && raw.trim()) {
    const n = Number(raw);
    if (Number.isFinite(n)) return n;
  }
  return undefined;
}

/** Pull name, slug, scientific_name, difficulty_score, care_level up front; strip noisy keys from the rest. */
export function preparePlantCatalogDisplay(data: Record<string, unknown>): {
  nameBlock: NameBlock | null;
  overviewBlock: OverviewBlock | null;
  rest: Record<string, unknown>;
} {
  const common = typeof data.name === "string" ? data.name : undefined;
  const slug = typeof data.slug === "string" ? data.slug : undefined;
  const scientific_name =
    typeof data.scientific_name === "string" ? data.scientific_name : undefined;

  const difficulty_score = parseDifficultyScore(data.difficulty_score);
  const care_level =
    typeof data.care_level === "string" && data.care_level.trim()
      ? data.care_level.trim()
      : undefined;

  const {
    name: _n,
    slug: _s,
    scientific_name: _sci,
    difficulty_score: _ds,
    care_level: _cl,
    ...restShallow
  } = data;
  const stripped = omitKeysDeep(restShallow, STRIP_FROM_CATALOG) as Record<string, unknown>;

  const nameBlock: NameBlock | null =
    common?.trim() || slug?.trim() || scientific_name?.trim()
      ? {
          ...(common?.trim() ? { common: common.trim() } : {}),
          ...(slug?.trim() ? { slug: slug.trim() } : {}),
          ...(scientific_name?.trim()
            ? { scientific_name: scientific_name.trim() }
            : {}),
        }
      : null;

  const overviewBlock: OverviewBlock | null =
    difficulty_score !== undefined || care_level !== undefined
      ? {
          ...(difficulty_score !== undefined ? { difficulty_score } : {}),
          ...(care_level !== undefined ? { care_level } : {}),
        }
      : null;

  return { nameBlock, overviewBlock, rest: stripped };
}

/** Drop null, undefined, empty strings, and empty objects/arrays after recursion. */
export function pruneCatalog(value: unknown): unknown {
  if (value === null || value === undefined) return undefined;
  if (typeof value === "string" && value.trim() === "") return undefined;
  if (Array.isArray(value)) {
    const next = value
      .map(pruneCatalog)
      .filter((v): v is NonNullable<typeof v> => v !== undefined);
    return next.length === 0 ? undefined : next;
  }
  if (typeof value === "object") {
    const o = value as Record<string, unknown>;
    const out: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(o)) {
      const p = pruneCatalog(v);
      if (p !== undefined) out[k] = p;
    }
    return Object.keys(out).length === 0 ? undefined : out;
  }
  return value;
}

function PrimitiveInline({ value }: { value: unknown }) {
  if (typeof value === "boolean") return <>{capitalizeFirstLetter(value ? "yes" : "no")}</>;
  if (typeof value === "number")
    return <>{Number.isFinite(value) ? String(value) : "—"}</>;
  if (typeof value === "string")
    return (
      <span className="break-words whitespace-pre-wrap">{capitalizeFirstLetter(value)}</span>
    );
  const s = String(value);
  return <span className="break-all">{capitalizeFirstLetter(s)}</span>;
}

/** Shared shell for Name, Care & difficulty, and catalog field sections. */
const SECTION_CARD_CLASS =
  "rounded-xl border-2 border-forest-300/70 bg-gradient-to-br from-emerald-50/95 via-white to-sage-50/50 p-6 shadow-md ring-1 ring-forest-900/[0.06]";
const SECTION_TITLE_CLASS =
  "text-base sm:text-lg font-bold tracking-wide text-forest-800 mb-5";

function FieldRow({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-[minmax(7rem,11rem)_minmax(0,1fr)] gap-x-4 gap-y-1 py-2 border-b border-sage-100/80 last:border-0">
      <div className="text-xs font-medium text-sage-600 shrink-0">{formatKeyLabel(label)}</div>
      <div className="text-sm text-forest-800 leading-relaxed min-w-0">{children}</div>
    </div>
  );
}

/** Nested object or array — no top-level card. */
function CatalogBranch({ value, path }: { value: unknown; path: string }) {
  if (value === null || value === undefined) return null;

  if (Array.isArray(value)) {
    if (value.length === 0) return null;
    return (
      <ul className="list-none space-y-2 m-0 p-0">
        {value.map((item, i) => (
          <li
            key={`${path}-${i}`}
            className="rounded-lg bg-sage-50/60 px-3 py-2 border border-sage-100/60"
          >
            {typeof item === "object" && item !== null ? (
              <CatalogBranch value={item} path={`${path}[${i}]`} />
            ) : (
              <PrimitiveInline value={item} />
            )}
          </li>
        ))}
      </ul>
    );
  }

  if (typeof value === "object") {
    const o = value as Record<string, unknown>;
    const keys = Object.keys(o);
    if (keys.length === 0) return null;
    return (
      <div className="space-y-0">
        {keys.map((k) => {
          const v = o[k];
          const isNestedObj =
            v !== null &&
            typeof v === "object" &&
            !Array.isArray(v) &&
            Object.keys(v as object).length > 0;
          const isArr = Array.isArray(v);

          if (isNestedObj || isArr) {
            return (
              <div key={`${path}.${k}`} className="py-2 border-b border-sage-100/80 last:border-0">
                <div className="text-sm font-semibold text-forest-800 mb-2">{formatKeyLabel(k)}</div>
                <div className="ml-0 sm:ml-2 pl-3 border-l-2 border-forest-400/25">
                  <CatalogBranch value={v} path={`${path}.${k}`} />
                </div>
              </div>
            );
          }

          return (
            <FieldRow key={`${path}.${k}`} label={k}>
              <PrimitiveInline value={v} />
            </FieldRow>
          );
        })}
      </div>
    );
  }

  return <PrimitiveInline value={value} />;
}

function TopSection({ title, value }: { title: string; value: unknown }) {
  return (
    <section className={SECTION_CARD_CLASS}>
      <h3 className={SECTION_TITLE_CLASS}>{formatKeyLabel(title)}</h3>
      <CatalogBranch value={value} path={title} />
    </section>
  );
}

function NameSection({ block }: { block: NameBlock }) {
  return (
    <section className={SECTION_CARD_CLASS}>
      <h3 className={SECTION_TITLE_CLASS}>Name</h3>
      <div className="space-y-0">
        {block.common != null ? (
          <FieldRow label="Common name">
            <PrimitiveInline value={block.common} />
          </FieldRow>
        ) : null}
        {block.slug != null ? (
          <FieldRow label="Slug">
            <PrimitiveInline value={block.slug} />
          </FieldRow>
        ) : null}
        {block.scientific_name != null ? (
          <FieldRow label="Scientific name">
            <PrimitiveInline value={block.scientific_name} />
          </FieldRow>
        ) : null}
      </div>
    </section>
  );
}

function OverviewSection({ block }: { block: OverviewBlock }) {
  return (
    <section className={SECTION_CARD_CLASS}>
      <h3 className={SECTION_TITLE_CLASS}>Care & difficulty</h3>
      <div className="space-y-0">
        {block.difficulty_score !== undefined ? (
          <FieldRow label="Difficulty score">
            <span className="text-lg font-semibold tabular-nums text-forest-900">
              {block.difficulty_score}
            </span>
          </FieldRow>
        ) : null}
        {block.care_level != null ? (
          <FieldRow label="Care level">
            <span className="text-base font-medium text-forest-900">
              {capitalizeFirstLetter(block.care_level)}
            </span>
          </FieldRow>
        ) : null}
      </div>
    </section>
  );
}

export function PlantCatalogTree({ data }: { data: Record<string, unknown> }) {
  const { nameBlock, overviewBlock, rest } = preparePlantCatalogDisplay(data);
  const pruned = pruneCatalog(rest) as Record<string, unknown> | undefined;

  const restEntries = pruned ? Object.entries(pruned) : [];
  const showRest = restEntries.length > 0;
  const showName = nameBlock != null;
  const showOverview = overviewBlock != null;

  if (!showName && !showOverview && !showRest) {
    return <p className="text-sage-500 text-sm">No details to show.</p>;
  }

  return (
    <div className="flex flex-col gap-4">
      {showName ? <NameSection block={nameBlock!} /> : null}
      {showOverview ? <OverviewSection block={overviewBlock!} /> : null}
      {restEntries.map(([k, v]) => (
        <TopSection key={k} title={k} value={v} />
      ))}
    </div>
  );
}
