from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client

# ---------- Config ----------
BASE_DIR = Path(__file__).resolve().parents[2]  # project root
CLEAN_DIR = BASE_DIR / "data" / "clean"

FILES: List[Tuple[str, str]] = [
    ("blue_ridge_clean.csv", "Blue Ridge GA"),
    ("bradenton_clean.csv", "Bradenton FL"),
    ("indianapolis_clean.csv", "Indianapolis IN"),
]

# keep small to avoid PostgREST payload issues
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "200"))


# ---------- Helpers ----------
def to_int(v: Any) -> Optional[int]:
    v = py(v)
    if v is None:
        return None
    try:
        return int(float(v))  # handles "26.0"
    except Exception:
        return None


def py(v: Any) -> Any:
    """Convert pandas/numpy scalars to plain Python + turn NaN/NA into None."""
    if v is None:
        return None
    # pandas NA / numpy nan
    if v is pd.NA:
        return None
    try:
        if isinstance(v, float) and np.isnan(v):
            return None
    except Exception:
        pass
    # numpy scalars
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def chunked(items: List[Dict[str, Any]], size: int) -> List[List[Dict[str, Any]]]:
    return [items[i: i + size] for i in range(0, len(items), size)]


def upsert_rows(sb, table: str, rows: List[Dict[str, Any]], on_conflict: str) -> None:
    if not rows:
        return
    for ch in chunked(rows, BATCH_SIZE):
        sb.table(table).upsert(ch, on_conflict=on_conflict).execute()


def get_market_id(sb, market_name: str) -> str:
    res = sb.table("markets").select("market_id").eq(
        "market_name", market_name).limit(1).execute()
    data = res.data or []
    if not data:
        raise RuntimeError(
            f"Could not find market_id for market_name={market_name}")
    return data[0]["market_id"]


def bool_columns(df: pd.DataFrame) -> List[str]:
    # In your clean CSV, bool columns are lowercased like has_pool, system_firepit, etc.
    bools = []
    for c in df.columns:
        if c.startswith(("has_", "system_")) or c in {"instant_book", "superhost", "is_guest_favorite", "is_super_host"}:
            bools.append(c)
    return bools


# ---------- Main ingestion ----------
def ingest_one(sb, csv_path: Path, market_name: str) -> None:
    print(f"\nIngesting market: {market_name}")
    print(f"File: {csv_path}")

    df = pd.read_csv(csv_path, low_memory=False)

    # Market row (use first non-null city/state from file)
    city = df["city"].dropna(
    ).iloc[0] if "city" in df.columns and df["city"].notna().any() else None
    state = df["state"].dropna(
    ).iloc[0] if "state" in df.columns and df["state"].notna().any() else None

    sb.table("markets").upsert(
        [{"market_name": market_name, "city": py(city), "state": py(state)}],
        on_conflict="market_name",
    ).execute()

    market_id = get_market_id(sb, market_name)
    print("market_id:", market_id)

    bcols = bool_columns(df)

    core_rows: List[Dict[str, Any]] = []
    perf_rows: List[Dict[str, Any]] = []
    feat_rows: List[Dict[str, Any]] = []

    for _, r in df.iterrows():
        property_id = str(r["property_id"]).strip()

        # properties_core
        core = {
            "property_id": property_id,
            "market_id": market_id,
            "title": py(r.get("title")),
            "host_id": py(r.get("host_id")),
            "airbnb_listing_url": py(r.get("airbnb_listing_url")),
            "vrbo_listing_url": py(r.get("vrbo_listing_url")),
            "airbnb_host_url": py(r.get("airbnb_host_url")),
            "city": py(r.get("city")),
            "state": py(r.get("state")),
            "zipcode": py(r.get("zipcode")),
            "latitude": py(r.get("latitude")),
            "longitude": py(r.get("longitude")),
            "description": py(r.get("description")),
            "amenities_text": py(r.get("amenities_text")),
            # optional: store the whole cleaned row for traceability
            "raw": {k: py(r.get(k)) for k in df.columns},
        }
        core_rows.append(core)

        # performance
        perf = {
            "property_id": property_id,
            "revenue": py(r.get("revenue")),
            "revenue_potential": py(r.get("revenue_potential")),
            "adr": py(r.get("adr")),
            "occupancy": py(r.get("occupancy")),
            "cleaning_fee": py(r.get("cleaning_fee")),
            "available_nights": to_int(r.get("available_nights")),
            "price_tier": py(r.get("price_tier")),
            "total_reviews": to_int(r.get("total_reviews")),
            "property_rating": py(r.get("property_rating")),
            "stars": py(r.get("stars")),
        }
        perf_rows.append(perf)

        # property_features (basics + booleans)
        feat = {
            "property_id": property_id,
            "bedrooms": to_int(r.get("bedrooms")),
            "accommodates": to_int(r.get("accommodates")),
            "bathrooms": to_int(r.get("bathrooms")),
            "minimum_stay": to_int(r.get("minimum_stay")),
        }
        for c in bcols:
            feat[c] = py(r.get(c))
        feat_rows.append(feat)

    print(f"Upserting properties_core: {len(core_rows)} rows")
    upsert_rows(sb, "properties_core", core_rows, on_conflict="property_id")

    print(f"Upserting performance: {len(perf_rows)} rows")
    upsert_rows(sb, "performance", perf_rows, on_conflict="property_id")

    print(f"Upserting property_features: {len(feat_rows)} rows")
    upsert_rows(sb, "property_features", feat_rows, on_conflict="property_id")

    print("✅ Done:", market_name)


def main():
    load_dotenv()

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        raise RuntimeError(
            "Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY in .env")

    sb = create_client(url, key)

    for filename, market_name in FILES:
        path = CLEAN_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing clean CSV: {path}")
        ingest_one(sb, path, market_name)


if __name__ == "__main__":
    main()
