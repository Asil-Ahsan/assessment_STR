from __future__ import annotations

import os
import math
from typing import Any, Dict, List, Optional, Set, Tuple

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError(
        "Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY in .env")

sb = create_client(SUPABASE_URL, SUPABASE_KEY)

app = FastAPI(title="STR Search Take-Home API", version="1.1.0")


# -----------------------------
# Small UX fixes
# -----------------------------
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home():
    return """
    <html>
      <head><title>STR Search Take-Home</title></head>
      <body style="font-family: Arial; max-width: 900px; margin: 40px auto; line-height: 1.4;">
        <h1>STR Search Take-Home API</h1>
        <p>Supabase-backed STR scoring + investor insights.</p>
        <ul>
          <li><a href="/docs">Swagger UI</a></li>
          <li><a href="/redoc">ReDoc</a></li>
          <li><a href="/markets">Markets</a></li>
          <li><a href="/insights/top-performers">Top performers</a></li>
        </ul>
      </body>
    </html>
    """


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)


# -----------------------------
# Helpers
# -----------------------------
def chunked(xs: List[str], n: int) -> List[List[str]]:
    return [xs[i: i + n] for i in range(0, len(xs), n)]


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * \
        math.cos(phi2) * math.sin(dlmb / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def apply_intersection(cur: Optional[Set[str]], nxt: Set[str]) -> Set[str]:
    return nxt if cur is None else (cur & nxt)


def fetch_all_property_ids(table: str = "property_scores") -> Set[str]:
    out: Set[str] = set()
    start = 0
    batch = 1000

    while True:
        resp = sb.table(table).select("property_id").range(
            start, start + batch - 1).execute()
        data = resp.data or []

        for r in data:
            pid = r.get("property_id")
            if pid:
                out.add(pid)

        if len(data) < batch:
            break
        start += batch

    return out


def fetch_ids(table: str, col: str, op: str, value: Any) -> Set[str]:
    out: Set[str] = set()
    start = 0
    batch = 1000

    while True:
        q = sb.table(table).select("property_id")

        if op == "eq":
            q = q.eq(col, value)
        elif op == "gte":
            q = q.gte(col, value)
        elif op == "lte":
            q = q.lte(col, value)
        else:
            raise ValueError(f"Unsupported op: {op}")

        resp = q.range(start, start + batch - 1).execute()
        data = resp.data or []

        for r in data:
            pid = r.get("property_id")
            if pid:
                out.add(pid)

        if len(data) < batch:
            break
        start += batch

    return out


def get_filtered_property_ids(
    market_id: Optional[str],
    bedrooms: Optional[int],
    min_revenue: Optional[float],
    min_score: Optional[float],
    price_tier: Optional[str],
    # amenity / flags
    has_aircon: Optional[bool],
    has_pool: Optional[bool],
    has_hottub: Optional[bool],
    has_waterfront: Optional[bool],
    has_beach_access: Optional[bool],
    superhost: Optional[bool],
    instant_book: Optional[bool],
    is_guest_favorite: Optional[bool],
) -> Set[str]:
    """
    Filter using per-table server-side filters and intersect IDs in Python.
    Keeps it join-free and stable on Supabase REST.
    """
    ids: Optional[Set[str]] = fetch_all_property_ids("property_scores")

    if market_id:
        ids = apply_intersection(ids, fetch_ids(
            "properties_core", "market_id", "eq", market_id))

    if bedrooms is not None:
        ids = apply_intersection(ids, fetch_ids(
            "property_features", "bedrooms", "eq", bedrooms))

    if min_revenue is not None:
        ids = apply_intersection(ids, fetch_ids(
            "performance", "revenue", "gte", min_revenue))

    if min_score is not None:
        ids = apply_intersection(ids, fetch_ids(
            "property_scores", "investment_score", "gte", min_score))

    if price_tier:
        ids = apply_intersection(ids, fetch_ids(
            "performance", "price_tier", "eq", price_tier))

    # amenity / flags live in property_features
    def flag_filter(col: str, val: Optional[bool]):
        nonlocal ids
        if val is not None:
            ids = apply_intersection(ids, fetch_ids(
                "property_features", col, "eq", val))

    flag_filter("has_aircon", has_aircon)
    flag_filter("has_pool", has_pool)
    flag_filter("has_hottub", has_hottub)
    flag_filter("has_waterfront", has_waterfront)
    flag_filter("has_beach_access", has_beach_access)
    flag_filter("superhost", superhost)
    flag_filter("instant_book", instant_book)
    flag_filter("is_guest_favorite", is_guest_favorite)

    return ids or set()


def batch_fetch_by_ids(table: str, cols: str, ids: List[str], batch: int = 500) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for ch in chunked(ids, batch):
        resp = sb.table(table).select(cols).in_("property_id", ch).execute()
        for r in resp.data or []:
            pid = r.get("property_id")
            if pid:
                out[pid] = r
    return out


def get_market_bedroom_bucket_ids(market_id: str, bedrooms: int) -> List[str]:
    core = sb.table("properties_core").select("property_id").eq(
        "market_id", market_id).execute().data or []
    ids_market = {r["property_id"] for r in core if r.get("property_id")}

    feats = sb.table("property_features").select("property_id").eq(
        "bedrooms", bedrooms).execute().data or []
    ids_bed = {r["property_id"] for r in feats if r.get("property_id")}

    return list(ids_market & ids_bed)


def summarize_top_factors(score_breakdown: Dict[str, Any]) -> List[str]:
    norm = (score_breakdown or {}).get("normalized") or {}
    items = []
    for k, v in norm.items():
        if isinstance(v, (int, float)):
            items.append((k, v))
    items.sort(key=lambda x: x[1], reverse=True)
    top = items[:3]
    labels = {
        "rev_gap_s": "Revenue above market avg",
        "adr_gap_s": "ADR above market avg",
        "occ_gap_s": "Occupancy above market avg",
        "capture_s": "Strong revenue capture vs potential",
        "review_s": "Strong rating/reviews signal",
        "amenity_s": "Strong amenity package",
        "trust_s": "Trust/booking friction advantage",
    }
    return [labels.get(k, k) for k, _ in top]


# -----------------------------
# Response models
# -----------------------------
class PropertyListItem(BaseModel):
    property_id: str
    market_id: Optional[str] = None
    title: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zipcode: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None

    bedrooms: Optional[int] = None
    accommodates: Optional[int] = None
    property_rating: Optional[float] = None
    total_reviews: Optional[int] = None
    revenue_potential: Optional[float] = None

    revenue: Optional[float] = None
    adr: Optional[float] = None
    occupancy: Optional[float] = None
    price_tier: Optional[str] = None

    investment_score: Optional[float] = None
    has_aircon: Optional[bool] = None

    has_pool: Optional[bool] = None
    has_hottub: Optional[bool] = None
    has_waterfront: Optional[bool] = None
    has_beach_access: Optional[bool] = None
    system_view_ocean: Optional[bool] = None
    system_view_mountain: Optional[bool] = None
    system_firepit: Optional[bool] = None


class PropertiesResponse(BaseModel):
    page: int
    page_size: int
    returned: int
    items: List[PropertyListItem]


# -----------------------------
# Endpoints
# -----------------------------
@app.get("/markets")
def markets():
    data = (
        sb.table("markets")
        .select("market_id,market_name,city,state")
        .order("market_name", desc=False)
        .execute()
        .data
        or []
    )
    return {"markets": data}


@app.get("/properties", response_model=PropertiesResponse)
def list_properties(
    market: Optional[str] = Query(None, description="market_id (uuid)"),
    bedrooms: Optional[int] = Query(None, ge=0),
    price_tier: Optional[str] = Query(
        None, description="e.g. LUXURY / UPSCALE / MIDSCALE"),
    min_revenue: Optional[float] = Query(None, ge=0),
    min_score: Optional[float] = Query(None, ge=0),

    # amenity / flags
    has_aircon: Optional[bool] = Query(None),
    has_pool: Optional[bool] = Query(None),
    has_hottub: Optional[bool] = Query(None),
    has_waterfront: Optional[bool] = Query(None),
    has_beach_access: Optional[bool] = Query(None),
    superhost: Optional[bool] = Query(None),
    instant_book: Optional[bool] = Query(None),
    is_guest_favorite: Optional[bool] = Query(None),

    sort_by: str = Query("investment_score",
                         description="investment_score|revenue|adr|occupancy"),
    sort_dir: str = Query("desc", description="asc|desc"),
    page: int = Query(1, ge=1),
    page_size: int = Query(200, ge=1, le=1000),  # keep this sane
):
    """
    Robust paging:
    1) Page over property_scores (stable order, stable paging)
    2) Fetch detail tables ONLY for those page ids
    3) Apply extra filters via intersections, but never with huge IN lists
    """

    # Validate sort fields
    allowed_sort = {"investment_score", "revenue", "adr", "occupancy"}
    if sort_by not in allowed_sort:
        raise HTTPException(
            status_code=400, detail=f"sort_by must be one of {sorted(allowed_sort)}")
    if sort_dir not in {"asc", "desc"}:
        raise HTTPException(
            status_code=400, detail="sort_dir must be asc|desc")

    # First: figure out candidate ids via filters (but do NOT pass huge IN to Supabase later)
    # We'll build a set of ids and then page the sorted list in Python *only if needed*.
    ids = get_filtered_property_ids(
        market, bedrooms, min_revenue, min_score, price_tier,
        has_aircon, has_pool, has_hottub, has_waterfront, has_beach_access,
        superhost, instant_book, is_guest_favorite
    )
    if not ids:
        return PropertiesResponse(page=page, page_size=page_size, returned=0, items=[])

    ids = list(ids)

    # Sort ids by score/performance in a stable way (Python sort but safe because only 2025 rows)
    scores = batch_fetch_by_ids(
        "property_scores", "property_id,investment_score", ids)
    perf = batch_fetch_by_ids(
        "performance", "property_id,revenue,adr,occupancy,price_tier,property_rating,total_reviews,revenue_potential", ids)

    def sort_key(pid: str):
        if sort_by == "revenue":
            return perf.get(pid, {}).get("revenue", -1e18)
        if sort_by == "adr":
            return perf.get(pid, {}).get("adr", -1e18)
        if sort_by == "occupancy":
            return perf.get(pid, {}).get("occupancy", -1e18)
        return scores.get(pid, {}).get("investment_score", -1e18)

    reverse = (sort_dir == "desc")
    ids_sorted = sorted(ids, key=sort_key, reverse=reverse)

    # Page slice in Python (robust for this dataset size)
    start = (page - 1) * page_size
    end = start + page_size
    page_ids = ids_sorted[start:end]
    if not page_ids:
        return PropertiesResponse(page=page, page_size=page_size, returned=0, items=[])

    # Fetch core + features for JUST this page (small IN lists => no 1000 cap issues)
    core = batch_fetch_by_ids(
        "properties_core",
        "property_id,market_id,title,city,state,zipcode,latitude,longitude",
        page_ids
    )
    feats = batch_fetch_by_ids(
        "property_features",
        "property_id,bedrooms,accommodates,has_aircon,has_pool,has_hottub,has_waterfront,has_beach_access,system_view_ocean,system_view_mountain,system_firepit,superhost,instant_book,is_guest_favorite",
        page_ids
    )

    # Build response items
    items: List[PropertyListItem] = []
    for pid in page_ids:
        c = core.get(pid, {})
        p = perf.get(pid, {})
        f = feats.get(pid, {})
        s = scores.get(pid, {})

        items.append(
            PropertyListItem(
                property_id=pid,
                market_id=c.get("market_id"),
                title=c.get("title"),
                city=c.get("city"),
                state=c.get("state"),
                zipcode=c.get("zipcode"),
                latitude=c.get("latitude"),
                longitude=c.get("longitude"),
                bedrooms=f.get("bedrooms"),
                accommodates=f.get("accommodates"),
                revenue=p.get("revenue"),
                adr=p.get("adr"),
                occupancy=p.get("occupancy"),
                price_tier=p.get("price_tier"),
                property_rating=p.get("property_rating"),
                total_reviews=p.get("total_reviews"),
                revenue_potential=p.get("revenue_potential"),
                investment_score=s.get("investment_score"),
                has_aircon=f.get("has_aircon"),
                has_pool=f.get("has_pool"),
                has_hottub=f.get("has_hottub"),
                has_waterfront=f.get("has_waterfront"),
                has_beach_access=f.get("has_beach_access"),
                system_view_ocean=f.get("system_view_ocean"),
                system_view_mountain=f.get("system_view_mountain"),
                system_firepit=f.get("system_firepit"),
            )
        )

    return PropertiesResponse(page=page, page_size=page_size, returned=len(items), items=items)


@app.get("/properties/{property_id}/analysis")
def property_analysis(property_id: str):
    core = sb.table("properties_core").select(
        "*").eq("property_id", property_id).execute().data
    if not core:
        raise HTTPException(status_code=404, detail="property not found")
    core = core[0]

    perf = sb.table("performance").select(
        "*").eq("property_id", property_id).execute().data
    perf = (perf or [None])[0]

    feats = sb.table("property_features").select(
        "*").eq("property_id", property_id).execute().data
    feats = (feats or [None])[0]

    score = sb.table("property_scores").select(
        "*").eq("property_id", property_id).execute().data
    score = (score or [None])[0]

    market_id = core.get("market_id")
    bedrooms = feats.get("bedrooms") if feats else None
    if not market_id or bedrooms is None:
        return {
            "property": {"core": core, "performance": perf, "features": feats},
            "score": score,
            "market_bedroom_bucket": None,
            "comparables": [],
        }

    bucket_ids = get_market_bedroom_bucket_ids(market_id, int(bedrooms))

    bucket_perf = []
    bucket_core = []
    for ch in chunked(bucket_ids, 500):
        bucket_perf.extend(
            sb.table("performance")
            .select("property_id,revenue,adr,occupancy")
            .in_("property_id", ch)
            .execute()
            .data
            or []
        )
        bucket_core.extend(
            sb.table("properties_core")
            .select("property_id,latitude,longitude,city,state,title")
            .in_("property_id", ch)
            .execute()
            .data
            or []
        )

    def mean_num(rows, k):
        vals = [r.get(k) for r in rows if isinstance(r.get(k), (int, float))]
        return sum(vals) / len(vals) if vals else None

    avg_rev = mean_num(bucket_perf, "revenue")
    avg_adr = mean_num(bucket_perf, "adr")
    avg_occ = mean_num(bucket_perf, "occupancy")

    gaps = {}
    if perf:
        if isinstance(perf.get("revenue"), (int, float)) and isinstance(avg_rev, (int, float)):
            gaps["revenue_gap"] = perf["revenue"] - avg_rev
        if isinstance(perf.get("adr"), (int, float)) and isinstance(avg_adr, (int, float)):
            gaps["adr_gap"] = perf["adr"] - avg_adr
        if isinstance(perf.get("occupancy"), (int, float)) and isinstance(avg_occ, (int, float)):
            gaps["occupancy_gap"] = perf["occupancy"] - avg_occ

    bucket_summary = {
        "market_id": market_id,
        "bedrooms": int(bedrooms),
        "count": len(bucket_ids),
        "avg_revenue": avg_rev,
        "avg_adr": avg_adr,
        "avg_occupancy": avg_occ,
        "gaps_for_property": gaps,
    }

    lat = core.get("latitude")
    lon = core.get("longitude")

    comparables = []
    if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
        core_map = {r["property_id"]                    : r for r in bucket_core if r.get("property_id")}

        bucket_scores = []
        for ch in chunked(bucket_ids, 500):
            bucket_scores.extend(
                sb.table("property_scores")
                .select("property_id,investment_score")
                .in_("property_id", ch)
                .execute()
                .data
                or []
            )
        score_map = {r["property_id"]                     : r for r in bucket_scores if r.get("property_id")}

        candidates = []
        for pid, r in core_map.items():
            if pid == property_id:
                continue
            la, lo = r.get("latitude"), r.get("longitude")
            if not (isinstance(la, (int, float)) and isinstance(lo, (int, float))):
                continue
            d = haversine_km(float(lat), float(lon), float(la), float(lo))
            candidates.append((d, pid))

        candidates.sort(key=lambda x: x[0])
        for d, pid in candidates[:10]:
            r = core_map.get(pid, {})
            srow = score_map.get(pid, {})
            comparables.append(
                {
                    "property_id": pid,
                    "distance_km": round(d, 3),
                    "title": r.get("title"),
                    "city": r.get("city"),
                    "state": r.get("state"),
                    "investment_score": srow.get("investment_score"),
                }
            )
    else:
        bucket_scores = []
        for ch in chunked(bucket_ids, 500):
            bucket_scores.extend(
                sb.table("property_scores")
                .select("property_id,investment_score")
                .in_("property_id", ch)
                .execute()
                .data
                or []
            )
        bucket_scores = [r for r in bucket_scores if r.get(
            "property_id") != property_id]
        bucket_scores.sort(key=lambda r: r.get(
            "investment_score") or -1e18, reverse=True)
        comparables = bucket_scores[:10]

    return {
        "property": {"core": core, "performance": perf, "features": feats},
        "score": score,
        "market_bedroom_bucket": bucket_summary,
        "comparables": comparables,
    }


@app.get("/insights/top-performers")
def top_performers():
    top = (
        sb.table("property_scores")
        .select("property_id,market_id,investment_score,score_breakdown")
        .order("investment_score", desc=True)
        .limit(20)
        .execute()
        .data
        or []
    )

    ids = [r["property_id"] for r in top if r.get("property_id")]
    core = batch_fetch_by_ids(
        "properties_core", "property_id,title,city,state,market_id", ids)
    perf = batch_fetch_by_ids(
        "performance", "property_id,revenue,adr,occupancy,price_tier", ids)
    feats = batch_fetch_by_ids(
        "property_features", "property_id,bedrooms,accommodates", ids)

    items = []
    groups: Dict[Tuple[str, int], int] = {}

    for r in top:
        pid = r.get("property_id")
        if not pid:
            continue
        c = core.get(pid, {})
        p = perf.get(pid, {})
        f = feats.get(pid, {})
        br = f.get("bedrooms")
        mk = c.get("market_id") or r.get("market_id")

        if mk is not None and br is not None:
            groups[(mk, int(br))] = groups.get((mk, int(br)), 0) + 1

        breakdown = r.get("score_breakdown") or {}
        items.append(
            {
                "property_id": pid,
                "market_id": mk,
                "title": c.get("title"),
                "city": c.get("city"),
                "state": c.get("state"),
                "bedrooms": br,
                "accommodates": f.get("accommodates"),
                "revenue": p.get("revenue"),
                "adr": p.get("adr"),
                "occupancy": p.get("occupancy"),
                "price_tier": p.get("price_tier"),
                "investment_score": r.get("investment_score"),
                "top_factors": summarize_top_factors(breakdown),
                "score_breakdown": r.get("score_breakdown"),
            }
        )

    grouped = [
        {"market_id": mk, "bedrooms": br, "count_in_top20": cnt}
        for (mk, br), cnt in sorted(groups.items(), key=lambda x: (-x[1], x[0][0], x[0][1]))
    ]

    return {"top20": items, "grouped_summary": grouped}
