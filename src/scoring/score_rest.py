from __future__ import annotations

import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# ----------------------------
# Helpers
# ----------------------------
def py(v: Any) -> Any:
    if v is None:
        return None
    try:
        if isinstance(v, float) and np.isnan(v):
            return None
    except Exception:
        pass
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def chunked(xs: List[Dict[str, Any]], n: int) -> List[List[Dict[str, Any]]]:
    return [xs[i:i + n] for i in range(0, len(xs), n)]


def pct_rank(s: pd.Series) -> pd.Series:
    if s.notna().sum() == 0:
        return pd.Series(np.nan, index=s.index)
    return s.rank(pct=True, method="average")


def winsorize(s: pd.Series, lo=0.01, hi=0.99) -> pd.Series:
    if s.notna().sum() == 0:
        return s
    return s.clip(s.quantile(lo), s.quantile(hi))


def norm_bucket(df, group_cols, col):
    return df.groupby(group_cols, dropna=False)[col] \
             .transform(lambda s: pct_rank(winsorize(s)))


# ----------------------------
# Main
# ----------------------------
def main():
    load_dotenv()
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

    if not url or not key:
        raise RuntimeError("Missing Supabase credentials")

    sb = create_client(url, key)

    print("Fetching data...")

    props = sb.table("properties_core").select(
        "property_id,market_id").execute().data
    perf = sb.table("performance").select(
        "property_id,revenue,revenue_potential,adr,occupancy,"
        "total_reviews,property_rating,price_tier"
    ).execute().data
    feats = sb.table("property_features").select(
        "property_id,bedrooms,accommodates,"
        "has_aircon,has_kitchen,has_parking,"
        "has_pool,has_hottub,has_waterfront,has_beach_access,"
        "system_view_ocean,system_view_mountain,system_firepit,system_grill,"
        "superhost,instant_book,is_guest_favorite"
    ).execute().data

    df = (
        pd.DataFrame(props)
        .merge(pd.DataFrame(perf), on="property_id", how="left")
        .merge(pd.DataFrame(feats), on="property_id", how="left")
    )

    print("Scoring frame:", df.shape)

    # ----------------------------
    # Basic cleanup
    # ----------------------------
    num_cols = [
        "revenue", "revenue_potential", "adr", "occupancy",
        "total_reviews", "property_rating", "bedrooms", "accommodates"
    ]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["occ_01"] = np.where(df["occupancy"] > 1.5,
                            df["occupancy"] / 100, df["occupancy"])

    group_cols = ["market_id", "bedrooms"]

    # ----------------------------
    # Bucket-relative performance
    # ----------------------------
    df["rev_gap"] = df["revenue"] - \
        df.groupby(group_cols)["revenue"].transform("mean")
    df["adr_gap"] = df["adr"] - df.groupby(group_cols)["adr"].transform("mean")
    df["occ_gap"] = df["occ_01"] - \
        df.groupby(group_cols)["occ_01"].transform("mean")

    df["rev_gap_s"] = norm_bucket(df, group_cols, "rev_gap")
    df["adr_gap_s"] = norm_bucket(df, group_cols, "adr_gap")
    df["occ_gap_s"] = norm_bucket(df, group_cols, "occ_gap")

    # ----------------------------
    # Revenue capture
    # ----------------------------
    df["capture"] = df["revenue"] / df["revenue_potential"]
    df["capture"] = df["capture"].clip(0, 2)

    df["capture_s"] = norm_bucket(df, group_cols, "capture")

    # ----------------------------
    # Reviews: Bayesian shrinkage
    # ----------------------------
    global_mean = df["property_rating"].mean()
    prior = df.groupby(group_cols)["property_rating"].transform(
        "mean").fillna(global_mean)

    m = 20
    r = df["property_rating"].fillna(prior)
    n = df["total_reviews"].fillna(0)

    df["adj_rating"] = (prior * m + r * n) / (m + n)

    conf = np.log1p(n) / np.log1p(50)
    conf = 0.4 + 0.6 * conf

    df["review_raw"] = (df["adj_rating"] / 5) * conf
    df["review_s"] = norm_bucket(df, group_cols, "review_raw")

    # ----------------------------
    # Amenities (aggregate, not biased)
    # ----------------------------
    amen_cols = [
        "has_aircon", "has_pool", "has_hottub",
        "has_waterfront", "has_beach_access",
        "system_view_ocean", "system_view_mountain",
        "system_firepit", "system_grill"
    ]
    for c in amen_cols:
        df[c] = df[c].fillna(False).astype(int)

    df["amenity_raw"] = df[amen_cols].sum(
        axis=1) / (1 + 0.5 * df["bedrooms"].fillna(0))
    df["amenity_s"] = norm_bucket(df, group_cols, "amenity_raw")

    # ----------------------------
    # Trust signals
    # ----------------------------
    df["trust_raw"] = (
        df["superhost"].fillna(False).astype(int) * 1.0 +
        df["instant_book"].fillna(False).astype(int) * 0.7 +
        df["is_guest_favorite"].fillna(False).astype(int) * 0.8
    )
    df["trust_s"] = norm_bucket(df, group_cols, "trust_raw")

    # ----------------------------
    # PCA (diagnostic, not dominant)
    # ----------------------------
    pca_features = [
        "rev_gap_s", "adr_gap_s", "occ_gap_s",
        "capture_s", "review_s", "amenity_s", "trust_s"
    ]
    X = df[pca_features].fillna(0)
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=1)
    df["pca_1"] = pca.fit_transform(Xs).ravel()
    df["pca_s"] = pct_rank(df["pca_1"])

    # ----------------------------
    # Final weighting (explainable)
    # ----------------------------
    weights = {
        "rev_gap_s": 0.22,
        "capture_s": 0.18,
        "review_s": 0.18,
        "amenity_s": 0.16,
        "adr_gap_s": 0.10,
        "occ_gap_s": 0.08,
        "trust_s": 0.05,
        "pca_s": 0.03,
    }

    score01 = sum(weights[k] * df[k].fillna(0) for k in weights)

    df["investment_score"] = 10 + 90 * (score01 ** 0.65)
    df["investment_score"] = df["investment_score"].clip(0, 100).round(2)

    # ----------------------------
    # Write back
    # ----------------------------
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "property_id": r["property_id"],
            "market_id": r["market_id"],
            "investment_score": py(r["investment_score"]),
            "score_breakdown": {
                "normalized": {k: py(r[k]) for k in weights},
                "weights": weights,
                "raw": {
                    "revenue": py(r["revenue"]),
                    "adr": py(r["adr"]),
                    "occupancy": py(r["occ_01"]),
                    "rating": py(r["property_rating"]),
                },
                "pca": {
                    "explained_variance": float(pca.explained_variance_ratio_[0])
                }
            }
        })

    print("Upserting:", len(rows))
    for ch in chunked(rows, 250):
        sb.table("property_scores").upsert(
            ch, on_conflict="property_id").execute()

    print("Done.")


if __name__ == "__main__":
    main()
