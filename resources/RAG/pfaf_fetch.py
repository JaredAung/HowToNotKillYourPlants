"""Fetch PFAF plant pages (requests); parsing lives in ``extract_normalize``."""
from __future__ import annotations

import argparse
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote_plus

import requests
from dotenv import load_dotenv

_RAG_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_RAG_DIR) not in sys.path:
    sys.path.insert(0, str(_RAG_DIR))

from convert_to_chunks import chunks_from_structured
from extract_normalize import PfafStructured, parse_pfaf_html

_DEFAULT_UA = "Mozilla/5.0 (compatible; HowToKeepYourPlantsAlive/1.0; +research)"
_PFAF_PLANT_BASE = "https://pfaf.org/user/Plant.aspx"

# One Session per fetch thread (connection reuse; ``requests.Session`` is not thread-safe shared).
_fetch_tls = threading.local()


def _thread_fetch_session() -> requests.Session:
    s = getattr(_fetch_tls, "session", None)
    if s is None:
        s = requests.Session()
        s.headers.update({"User-Agent": _DEFAULT_UA})
        _fetch_tls.session = s
    return s


def _ensure_backend_on_path() -> None:
    """So ``from database import get_db`` resolves (``backend/`` package)."""
    backend = _PROJECT_ROOT / "backend"
    if str(backend) not in sys.path:
        sys.path.insert(0, str(backend))


def pfaf_plant_url(latin_name: str) -> str:
    """
    Build a PFAF plant URL like
    ``https://pfaf.org/user/Plant.aspx?LatinName=Morus+rubra``.
    """
    latin = (latin_name or "").strip()
    if not latin:
        raise ValueError("latin_name is empty")
    q = quote_plus(latin)
    return f"{_PFAF_PLANT_BASE}?LatinName={q}"


def mongo_plant_entries_for_pfaf(*, limit: int | None = None) -> list[tuple[int | str | None, str]]:
    """
    Load ``(plant_id, scientific_name)`` pairs from NewPlantCollection (first row per distinct name).

    Uses ``MONGO_URI``, ``MONGO_DATABASE`` from ``.env``, and collection
    ``NEW_PLANT_COLLECTION`` (defaults to ``NewPlantCollection`` if unset).
    """
    _ensure_backend_on_path()
    load_dotenv(_PROJECT_ROOT / ".env")
    from database import get_db

    collection_name = (os.environ.get("NEW_PLANT_COLLECTION") or "NewPlantCollection").strip()
    coll = get_db()[collection_name]
    seen: set[str] = set()
    entries: list[tuple[int | str | None, str]] = []
    cursor = coll.find(
        {"scientific_name": {"$exists": True, "$nin": [None, ""]}},
        {"scientific_name": 1, "plant_id": 1},
    )
    for doc in cursor:
        name = doc.get("scientific_name")
        if not isinstance(name, str) or not name.strip():
            continue
        name = name.strip()
        if name in seen:
            continue
        seen.add(name)
        entries.append((doc.get("plant_id"), name))
        if limit is not None and len(entries) >= limit:
            break
    entries.sort(key=lambda x: x[1])
    return entries


@dataclass
class PfafMongoFetchReport:
    """Result of batch-fetching PFAF pages for Mongo latin names."""

    succeeded: list[str]
    failed: list[tuple[str, str]]
    skipped_already_in_pinecone: list[str]


@dataclass(frozen=True)
class PfafPlantExtract:
    """HTTP metadata + structured parse of a PFAF ``Plant.aspx`` page."""

    url: str
    status_code: int
    encoding: str | None
    structured: PfafStructured
    chunks: tuple[str, ...]
    chunks_from_window: int


def fetch_pfaf_plant(
    url: str,
    *,
    timeout: float = 30,
    session: requests.Session | None = None,
) -> PfafPlantExtract:
    """
    GET ``url``, parse the HTML into ``PfafStructured``.

    Pass ``session`` for connection reuse (e.g. one ``Session`` per thread in a pool).

    Raises ``requests.HTTPError`` if the response is not successful.
    """
    headers = {"User-Agent": _DEFAULT_UA}
    if session is None:
        r = requests.get(url, headers=headers, timeout=timeout)
    else:
        r = session.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()

    structured = parse_pfaf_html(r.text)
    stats = chunks_from_structured(structured)
    return PfafPlantExtract(
        url=str(r.url),
        status_code=r.status_code,
        encoding=r.encoding,
        structured=structured,
        chunks=tuple(stats.chunks),
        chunks_from_window=stats.chunks_from_window,
    )


def _fetch_one_mongo_entry(
    entry: tuple[int | str | None, str],
    *,
    timeout: float,
) -> tuple[int | str | None, str, PfafPlantExtract | None, Exception | None]:
    """Fetch + parse one plant (for thread pool). Uses a thread-local ``requests.Session``."""
    plant_id, scientific_name = entry
    try:
        url = pfaf_plant_url(scientific_name)
        out = fetch_pfaf_plant(url, timeout=timeout, session=_thread_fetch_session())
        return (plant_id, scientific_name, out, None)
    except Exception as e:
        return (plant_id, scientific_name, None, e)


def fetch_pfaf_plants_from_mongo(
    *,
    limit: int | None = None,
    timeout: float = 30,
    upsert_pinecone: bool = True,
    max_fetch_workers: int = 4,
    skip_if_latin_in_pinecone: bool = True,
) -> PfafMongoFetchReport:
    """
    For each ``(plant_id, scientific_name)`` from Mongo, build the PFAF URL and ``GET`` it via
    :func:`fetch_pfaf_plant`. Fetches run concurrently (bounded by ``max_fetch_workers``) with
    connection reuse per thread. After each successful fetch (as completions arrive), optionally
    upsert chunks to Pinecone using **reused** Voyage + Pinecone clients.

    When ``upsert_pinecone`` and ``skip_if_latin_in_pinecone`` are True, plants whose
    ``scientific_name`` already appears as ``latin_name`` metadata in Pinecone are skipped (no HTTP,
    no embed).

    Catches all exceptions per plant and records failures (including Pinecone upsert errors).
    """
    entries = mongo_plant_entries_for_pfaf(limit=limit)
    succeeded: list[str] = []
    failed: list[tuple[str, str]] = []
    skipped_already: list[str] = []

    voyage_client = None
    pc_index = None
    pine_ns = ""
    pine_dim = 0
    embed_fn = None
    if upsert_pinecone:
        from embed_pinecone import (
            embed_and_upsert_pfaf_chunks,
            make_pfaf_embed_clients_with_dimension,
            pinecone_has_latin_name,
        )

        voyage_client, pc_index, pine_ns, pine_dim = make_pfaf_embed_clients_with_dimension()
        embed_fn = embed_and_upsert_pfaf_chunks

        if skip_if_latin_in_pinecone:
            pending: list[tuple[int | str | None, str]] = []
            for e in entries:
                _pid, scientific_name = e
                if pinecone_has_latin_name(
                    pc_index,
                    latin_name=scientific_name,
                    namespace=pine_ns,
                    dimension=pine_dim,
                ):
                    skipped_already.append(scientific_name)
                else:
                    pending.append(e)
            entries = pending

    def _embed_and_record(
        plant_id: int | str | None,
        scientific_name: str,
        out: PfafPlantExtract,
    ) -> None:
        # Pinecone ``latin_name`` metadata = Mongo ``scientific_name``, not PFAF HTML title.
        if embed_fn is not None:
            embed_fn(
                out.chunks,
                out.structured,
                out.url,
                plant_id=plant_id,
                voyage_client=voyage_client,
                pinecone_index_obj=pc_index,
                metadata_latin_name=scientific_name,
            )
        succeeded.append(scientific_name)

    workers = max(1, max_fetch_workers)
    if workers == 1:
        for entry in entries:
            plant_id, scientific_name, out, err = _fetch_one_mongo_entry(entry, timeout=timeout)
            if err is not None:
                failed.append((scientific_name, f"{type(err).__name__}: {err}"))
                continue
            assert out is not None
            try:
                _embed_and_record(plant_id, scientific_name, out)
            except Exception as e:
                failed.append((scientific_name, f"{type(e).__name__}: {e}"))
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            future_map = {ex.submit(_fetch_one_mongo_entry, e, timeout=timeout): e[1] for e in entries}
            for fut in as_completed(future_map):
                plant_id, scientific_name, out, err = fut.result()
                if err is not None:
                    failed.append((scientific_name, f"{type(err).__name__}: {err}"))
                    continue
                assert out is not None
                try:
                    _embed_and_record(plant_id, scientific_name, out)
                except Exception as e:
                    failed.append((scientific_name, f"{type(e).__name__}: {e}"))

    return PfafMongoFetchReport(
        succeeded=succeeded,
        failed=failed,
        skipped_already_in_pinecone=skipped_already,
    )


def _demo_single() -> None:
    _demo = "https://pfaf.org/user/Plant.aspx?LatinName=Morus+rubra"
    out = fetch_pfaf_plant(_demo)
    print("status:", out.status_code, "encoding:", out.encoding)
    print("chunks:", len(out.chunks), "| from sliding window:", out.chunks_from_window)
    print("--- first chunk ---")
    print(out.chunks[0][:600] if out.chunks else "(none)")

    try:
        from embed_pinecone import embed_and_upsert_pfaf_chunks, make_pfaf_embed_clients

        voyage_client, pc_index = make_pfaf_embed_clients()
        pc_res = embed_and_upsert_pfaf_chunks(
            out.chunks,
            out.structured,
            out.url,
            plant_id=4,
            voyage_client=voyage_client,
            pinecone_index_obj=pc_index,
        )
        print(
            "pinecone upserted:",
            pc_res.vectors_upserted,
            "index:",
            pc_res.index_name,
            "namespace:",
            repr(pc_res.namespace),
            "model:",
            pc_res.embedding_model,
        )
    except Exception as e:
        print("pinecone / voyage upsert skipped or failed:", e)

def _demo_mongo_batch(
    limit: int | None,
    *,
    upsert_pinecone: bool,
    max_fetch_workers: int,
    skip_if_latin_in_pinecone: bool,
) -> None:
    load_dotenv(_PROJECT_ROOT / ".env")
    coll_name = (os.environ.get("NEW_PLANT_COLLECTION") or "NewPlantCollection").strip()
    print(f"Loading scientific_name values from Mongo (collection: {coll_name})…")
    print(f"Concurrent fetch workers: {max(1, max_fetch_workers)}")
    report = fetch_pfaf_plants_from_mongo(
        limit=limit,
        upsert_pinecone=upsert_pinecone,
        max_fetch_workers=max_fetch_workers,
        skip_if_latin_in_pinecone=skip_if_latin_in_pinecone,
    )
    n_ok = len(report.succeeded)
    n_fail = len(report.failed)
    n_skip = len(report.skipped_already_in_pinecone)
    print(f"Successful: {n_ok}")
    print(f"Failed: {n_fail}")
    if upsert_pinecone and skip_if_latin_in_pinecone:
        print(f"Skipped (already in Pinecone): {n_skip}")
    if report.failed:
        print("\nFailed plant names:")
        for latin, _ in report.failed:
            print(f"  - {latin}")
        print("\nFailed (with error):")
        for latin, err in report.failed:
            print(f"  - {latin}: {err}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch PFAF plant pages; optional Mongo batch.")
    parser.add_argument(
        "--mongo",
        action="store_true",
        help="Fetch PFAF for every distinct scientific_name in Mongo (tracks failures)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Max plants when using --mongo")
    parser.add_argument(
        "--no-pinecone",
        action="store_true",
        help="With --mongo: fetch and parse only; do not embed/upsert to Pinecone",
    )
    parser.add_argument(
        "--fetch-workers",
        type=int,
        default=4,
        metavar="N",
        help="Max concurrent PFAF HTTP fetches when using --mongo (default: 4)",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="With --mongo and Pinecone: re-fetch and upsert even if latin_name already exists",
    )
    args = parser.parse_args()
    if args.mongo:
        _demo_mongo_batch(
            limit=args.limit,
            upsert_pinecone=not args.no_pinecone,
            max_fetch_workers=args.fetch_workers,
            skip_if_latin_in_pinecone=not args.no_skip_existing,
        )
    else:
        _demo_single()
