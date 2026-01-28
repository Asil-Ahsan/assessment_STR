from __future__ import annotations
import os
import requests
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import pydeck as pdk


st.set_page_config(
    page_title="STR Search: Assessment", layout="wide")

API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000").rstrip("/")


# ----------------------------
# API helpers
# ----------------------------
@st.cache_data(ttl=60)
def api_get(path: str, params: dict | None = None):
    url = f"{API_BASE}{path}"
    r = requests.get(url, params=params, timeout=60)
    if not r.ok:
        raise RuntimeError(f"{r.status_code} {r.text}")
    return r.json()


@st.cache_data(ttl=120)
def get_markets():
    data = api_get("/markets")
    mk = data.get("markets", [])
    df = pd.DataFrame(mk)
    if df.empty:
        return df
    df["label"] = df["market_name"].fillna(
        "") + " (" + df["city"].fillna("") + ", " + df["state"].fillna("") + ")"
    return df.sort_values("market_name")


@st.cache_data(ttl=60)
def get_properties_flat(params: dict):
    params = {k: v for k, v in dict(
        params).items() if v is not None and v != ""}

    all_items = []
    page = 1
    page_size = 500  # <= 1000, safe

    while True:
        p = dict(params)
        p["page"] = page
        p["page_size"] = page_size

        data = api_get("/properties", params=p)
        items = data.get("items", []) or []
        all_items.extend(items)

        if len(items) < page_size:
            break

        page += 1
        if page > 50:  # safety (50*500 = 25,000)
            break

    return pd.DataFrame(all_items)


def fmt_money(x):
    try:
        return f"${float(x):,.0f}"
    except Exception:
        return str(x)


# ----------------------------
# Sidebar filters
# ----------------------------
st.title("STR Search: Data Engineer Take-Home Assessment")
st.caption(f"API: {API_BASE}")

with st.sidebar:
    st.header("Filters")

    markets_df = get_markets()
    market_id = None
    if not markets_df.empty:
        market_choice = st.selectbox(
            "Market",
            ["All markets"] + markets_df["label"].tolist(),
            index=0,
        )
        if market_choice != "All markets":
            market_id = markets_df.loc[markets_df["label"]
                                       == market_choice, "market_id"].iloc[0]

    cA, cB = st.columns(2)
    with cA:
        min_score = st.number_input(
            "Min Score", min_value=0.0, max_value=100.0, value=0.0, step=1.0)
    with cB:
        min_revenue = st.number_input(
            "Min Revenue", min_value=0.0, value=0.0, step=10000.0)

    bedrooms = st.multiselect(
        "Bedrooms", options=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], default=[])

    st.subheader("Amenities / Flags")
    has_aircon = st.checkbox("Has A/C")
    has_pool = st.checkbox("Has Pool")
    has_hottub = st.checkbox("Has Hot Tub")
    has_waterfront = st.checkbox("Waterfront")
    has_beach_access = st.checkbox("Beach Access")
    superhost = st.checkbox("Superhost")
    instant_book = st.checkbox("Instant Book")
    is_guest_favorite = st.checkbox("Guest Favorite")

    sort_by = st.selectbox(
        "Sort by", ["investment_score", "revenue", "adr", "occupancy"], index=0)
    sort_dir = st.selectbox("Sort direction", ["desc", "asc"], index=0)

params = {
    "sort_by": sort_by,
    "sort_dir": sort_dir,
    "min_score": min_score if min_score > 0 else None,
    "min_revenue": min_revenue if min_revenue > 0 else None,
    "market": market_id,
    "has_aircon": True if has_aircon else None,
    "has_pool": True if has_pool else None,
    "has_hottub": True if has_hottub else None,
    "has_waterfront": True if has_waterfront else None,
    "has_beach_access": True if has_beach_access else None,
    "superhost": True if superhost else None,
    "instant_book": True if instant_book else None,
    "is_guest_favorite": True if is_guest_favorite else None,
}
params = {k: v for k, v in params.items() if v is not None and v != ""}

df = get_properties_flat(params)

if df.empty:
    st.warning("No properties matched your filters.")
    st.stop()

# client-side bedroom filter
if bedrooms:
    df = df[df["bedrooms"].isin(bedrooms)]

# client-side price tier filter
tiers = sorted([t for t in df.get("price_tier", pd.Series(
    [])).dropna().unique().tolist() if str(t).strip() != ""])
if tiers:
    tier_choice = st.multiselect("Price Tier", options=tiers, default=tiers)
    df = df[df["price_tier"].isin(tier_choice)]

# ----------------------------
# KPIs
# ----------------------------
k1, k2, k3, k4 = st.columns(4)
k1.metric("Listings", f"{len(df):,}")
k2.metric("Avg Score",
          f"{pd.to_numeric(df['investment_score'], errors='coerce').mean():.1f}")
k3.metric("Median Revenue", fmt_money(
    pd.to_numeric(df["revenue"], errors="coerce").median()))
k4.metric("Top Score",
          f"{pd.to_numeric(df['investment_score'], errors='coerce').max():.1f}")

st.divider()

# ----------------------------
# Charts
# ----------------------------
c1, c2 = st.columns(2)

with c1:
    st.subheader("Score vs Property Rating")

    tmp = df.copy()
    tmp["investment_score"] = pd.to_numeric(
        tmp.get("investment_score"), errors="coerce")
    tmp["property_rating"] = pd.to_numeric(
        tmp.get("property_rating"), errors="coerce")

    tmp = tmp.dropna(subset=["investment_score", "property_rating"])
    if tmp.empty:
        st.info("No rating/score data available for the current filters.")
    else:
        # Bin to 1..5 (nearest whole star)
        tmp["rating_bin"] = tmp["property_rating"].clip(
            1, 5).round(0).astype(int)

        agg = (
            tmp.groupby("rating_bin")
            .agg(
                avg_score=("investment_score", "mean"),
                n=("investment_score", "size"),
                sd=("investment_score", "std"),
            )
            .reset_index()
            .sort_values("rating_bin")
        )
        # Standard error for optional error bars
        agg["se"] = agg["sd"] / (agg["n"] ** 0.5)

        fig = plt.figure()
        x = agg["rating_bin"].astype(int).tolist()
        y = agg["avg_score"].tolist()

        # Bars (avg score)
        plt.bar(x, y)

        # Error bars (optional, looks pro)
        plt.errorbar(x, y, yerr=agg["se"].fillna(
            0).tolist(), fmt="none", capsize=4)

        # Line overlay (trend)
        plt.plot(x, y, marker="o")

        plt.xticks(x)
        plt.xlabel("Property Rating")
        plt.ylabel("Avg Score")

        # Add n labels on top of bars
        for xi, yi, ni in zip(x, y, agg["n"].tolist()):
            plt.text(xi, yi + 0.8, f"n={ni}",
                     ha="center", va="bottom", fontsize=9)

        st.pyplot(fig, clear_figure=True)

        st.caption(
            "Bars show average score per rating bucket. Line shows the improvement trend")


with c2:
    st.subheader("Revenue vs Score")
    fig = plt.figure()
    y = pd.to_numeric(df["revenue"], errors="coerce").fillna(0)
    x = pd.to_numeric(df["investment_score"], errors="coerce").fillna(0)
    plt.scatter(x, y)
    plt.xlabel("Score")
    plt.ylabel("Revenue")
    st.pyplot(fig, clear_figure=True)

c3, c4 = st.columns(2)

with c3:
    st.subheader("Avg Score by Bedrooms")
    tmp = df.dropna(subset=["bedrooms", "investment_score"]).copy()
    tmp["investment_score"] = pd.to_numeric(
        tmp["investment_score"], errors="coerce")
    tmp = tmp.groupby("bedrooms")["investment_score"].mean().reset_index()
    fig = plt.figure()
    plt.bar(tmp["bedrooms"].astype(str), tmp["investment_score"])
    plt.xlabel("Bedrooms")
    plt.ylabel("Avg Score")
    st.pyplot(fig, clear_figure=True)

with c4:
    st.subheader("Amenity Impact on Score")

    df_u = df.copy()
    df_u["investment_score"] = pd.to_numeric(
        df_u["investment_score"], errors="coerce")
    base = float(df_u["investment_score"].dropna().mean()
                 ) if df_u["investment_score"].notna().any() else np.nan

    flags = [
        ("A/C", "has_aircon"),
        ("Pool", "has_pool"),
        ("Hot tub", "has_hottub"),
        ("Waterfront", "has_waterfront"),
        ("Beach access", "has_beach_access"),
        ("Ocean view", "system_view_ocean"),
        ("Mountain view", "system_view_mountain"),
        ("Firepit", "system_firepit"),
    ]

    labels = ["Avg score"]
    vals = [base]
    notes = [f"baseline (n={int(df_u['investment_score'].notna().sum())})"]

    colors = ["#9aa0a6"]
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
               "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]

    idx = 0
    for label, col in flags:
        if col not in df_u.columns:
            continue

        has = df_u[df_u[col] == True]["investment_score"].dropna()
        no = df_u[df_u[col] == False]["investment_score"].dropna()

        # sample size guardrail
        if len(has) < 20 or len(no) < 20:
            continue

        mean_has = float(has.mean())
        delta = mean_has - base

        labels.append(f"+ {label}")
        vals.append(mean_has)
        notes.append(f"Δ={delta:+.2f} | has={len(has)} no={len(no)}")
        colors.append(palette[idx % len(palette)])
        idx += 1

    fig = plt.figure()
    plt.bar(labels, vals, color=colors)
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("Score with amenity present")
    st.pyplot(fig, clear_figure=True)

    st.caption(
        "Baseline avg score and avg score for listings that HAVE the amenity."
    )
    st.write({labels[i]: notes[i] for i in range(len(labels))})

    # end amenity block

with st.expander("CTE Insight: Top 3 Revenue per Bedroom vs Market Avg"):
    if market_id is None:
        st.info("Select a market in the sidebar to view this CTE insight.")
    else:
        t = df.copy()
        t["revenue"] = pd.to_numeric(t.get("revenue"), errors="coerce")
        t["bedrooms"] = pd.to_numeric(t.get("bedrooms"), errors="coerce")
        t = t.dropna(subset=["bedrooms", "revenue"])
        t["bedrooms"] = t["bedrooms"].astype(int)

        if t.empty:
            st.info("No data available for this market.")
        else:
            # Market avg by bedroom
            avg = (
                t.groupby("bedrooms")
                 .agg(avg_rev=("revenue", "mean"), n=("revenue", "size"))
                 .reset_index()
                 .sort_values("bedrooms")
            )

            # Top 3 by bedroom
            top = (
                t.sort_values(["bedrooms", "revenue"], ascending=[True, False])
                 .groupby("bedrooms")
                 .head(3)
                 .copy()
            )
            top["rank"] = top.groupby("bedrooms")["revenue"].rank(
                ascending=False, method="first"
            ).astype(int)

            # Merge avg into top for gap calc (optional for tooltip/table)
            top = top.merge(avg[["bedrooms", "avg_rev"]],
                            on="bedrooms", how="left")
            top["gap"] = top["revenue"] - top["avg_rev"]

            # --- Plot
            fig = plt.figure(figsize=(8, 4))

            # baseline avg line
            plt.plot(avg["bedrooms"], avg["avg_rev"],
                     marker="o", linewidth=2, label="Market avg")

            # top3 scatter with offsets so points don't overlap
            offsets = {1: -0.18, 2: 0.0, 3: 0.18}
            for r in [1, 2, 3]:
                sub = top[top["rank"] == r]
                if sub.empty:
                    continue
                x = sub["bedrooms"] + offsets[r]
                y = sub["revenue"]
                plt.scatter(x, y, s=40, label=f"Top #{r}")

            plt.xticks(sorted(avg["bedrooms"].unique()))
            plt.xlabel("Bedrooms")
            plt.ylabel("Revenue ($)")
            plt.legend(loc="upper left", fontsize=8)
            st.pyplot(fig, clear_figure=True)

            # Optional: show the computed table (super useful for explanation)
            show = top[["bedrooms", "rank", "property_id", "revenue",
                        "avg_rev", "gap"]].sort_values(["bedrooms", "rank"])
            st.dataframe(show, use_container_width=True)
            st.caption(
                "Dots = top 3 listings per bedroom. Line = market-bedroom average. Table shows the revenue gap.")
st.divider()

# ----------------------------
# Results table
# ----------------------------
st.subheader("Top 50 Listings")
show_cols = [
    "property_id", "market_id", "title", "city", "state", "price_tier",
    "bedrooms", "accommodates", "revenue", "adr", "occupancy", "investment_score",
]
show_cols = [c for c in show_cols if c in df.columns]
st.dataframe(df.sort_values("investment_score", ascending=False)
             [show_cols].head(50), use_container_width=True)

st.divider()

# ----------------------------
# Deep dive + improvements
# ----------------------------
st.subheader("Property Deep Dive")

top_df = df.sort_values("investment_score", ascending=False).head(300)
pid = st.selectbox("Select property_id (optional)", options=[
                   "(none)"] + top_df["property_id"].tolist(), index=0)

show_all = st.checkbox("Show all filtered properties on map", value=True)

st.markdown("### Map")

mdf = df.dropna(subset=["latitude", "longitude"]).copy()
if mdf.empty:
    st.info("No latitude/longitude available for current filters.")
elif show_all:
    # default color: red
    mdf["color"] = mdf.apply(lambda _: [255, 0, 0, 80], axis=1)

    # if a property is selected, color it blue
    if pid != "(none)":
        mdf.loc[mdf["property_id"] == pid, "color"] = mdf.loc[
            mdf["property_id"] == pid, "color"
        ].apply(lambda _: [0, 90, 255, 200])  # BLUE

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=mdf,
        get_position="[longitude, latitude]",
        get_fill_color="color",
        get_radius=120,
        pickable=True,
    )

    # center map
    view = pdk.ViewState(
        latitude=float(mdf["latitude"].mean()),
        longitude=float(mdf["longitude"].mean()),
        zoom=10,
    )

    tooltip = {"text": "{title}\nScore: {investment_score}\nRevenue: {revenue}"}

    st.pydeck_chart(
        pdk.Deck(layers=[layer], initial_view_state=view, tooltip=tooltip))
else:
    st.info("Disable 'Show all filtered properties' and select a property to see its exact location.")

if pid == "(none)":
    st.info("Select a property to see analysis + action plan.")
    st.stop()

analysis = api_get(f"/properties/{pid}/analysis")
core = analysis["property"]["core"]
perf = analysis["property"]["performance"]
feats = analysis["property"]["features"]
score = analysis.get("score") or {}
bucket = analysis.get("market_bedroom_bucket") or {}
comps = analysis.get("comparables") or []

left, right = st.columns(2)

with left:
    st.markdown("### Summary")
    st.write({
        "property_id": pid,
        "title": core.get("title"),
        "city": core.get("city"),
        "state": core.get("state"),
        "bedrooms": feats.get("bedrooms") if feats else None,
        "accommodates": feats.get("accommodates") if feats else None,
        "price_tier": perf.get("price_tier") if perf else None,
        "revenue": perf.get("revenue") if perf else None,
        "adr": perf.get("adr") if perf else None,
        "occupancy": perf.get("occupancy") if perf else None,
        "investment_score": score.get("investment_score"),
    })

with right:
    st.markdown("### Market+Bedrooms Benchmark")
    if bucket:
        st.write({
            "bucket_count": bucket.get("count"),
            "avg_revenue": bucket.get("avg_revenue"),
            "avg_adr": bucket.get("avg_adr"),
            "avg_occupancy": bucket.get("avg_occupancy"),
            "gaps": bucket.get("gaps_for_property"),
        })
    else:
        st.info("No bucket benchmark available (missing market_id or bedrooms).")

st.markdown("### Listing Summary: Pros, Cons, Action Plan")

bd = (score.get("score_breakdown") or {})
norm = (bd.get("normalized") or {})
gaps = (bd.get("gaps") or {})
amen = (bd.get("amenities") or {})
potential = (bd.get("potential") or {})
reviews = (bd.get("reviews") or {})

label_map = {
    "rev_gap_s": "Revenue advantage vs similar listings",
    "adr_gap_s": "ADR (pricing power) vs similar listings",
    "occ_gap_s": "Occupancy strength vs similar listings",
    "capture_s": "Revenue capture vs predicted potential",
    "review_s": "Rating & review signal",
    "amenity_s": "Amenity package strength",
    "trust_s": "Booking trust (Superhost/Instant Book/Favorite)",
}


def rank_factors(d: dict):
    pairs = [(k, float(v))
             for k, v in d.items() if isinstance(v, (int, float))]
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs


pairs = rank_factors(norm)
top3 = pairs[:3]
bot3 = list(reversed(pairs[-3:]))

# Quick facts (human)
facts = []
cap = potential.get("capture")
if isinstance(cap, (int, float)):
    facts.append(f"Revenue capture: **{cap:.2f}** (actual / potential)")

if isinstance(gaps, dict):
    if isinstance(gaps.get("rev_gap"), (int, float)):
        facts.append(
            f"Revenue gap vs market-bedroom avg: **{fmt_money(gaps['rev_gap'])}**")
    if isinstance(gaps.get("adr_gap"), (int, float)):
        facts.append(
            f"ADR gap vs market-bedroom avg: **{gaps['adr_gap']:.0f}**")
    if isinstance(gaps.get("occ_gap"), (int, float)):
        facts.append(
            f"Occupancy gap vs market-bedroom avg: **{gaps['occ_gap']:.2f}**")

has_ac = amen.get("has_aircon")
facts.append(
    f"A/C present: **{'YES' if has_ac is True else 'NO' if has_ac is False else 'Unknown'}**")

rating_val = perf.get("property_rating") if perf else None
if rating_val is not None:
    facts.append(f"Property rating: **{rating_val}**")

review_ct = perf.get("total_reviews") if perf else None
if review_ct is not None:
    facts.append(f"Total reviews: **{review_ct}**")

if facts:
    st.markdown("**Key facts:**")
    for f in facts:
        st.write("• " + f)

# Pros / Cons
c1, c2 = st.columns(2)

with c1:
    st.markdown("#### Pros")
    if not top3:
        st.write("No factor breakdown available.")
    else:
        for k, v in top3:
            st.write(f"• **{label_map.get(k,k)}** — strong (score={v:.2f})")

with c2:
    st.markdown("#### Cons")
    if not bot3:
        st.write("No factor breakdown available.")
    else:
        for k, v in bot3:
            st.write(f"• **{label_map.get(k,k)}** — weak (score={v:.2f})")

# Action plan (tie to weaknesses)
st.markdown("#### Action Plan for Highest ROI Improvements")

actions = []
weak_keys = [k for k, _ in bot3]

if "capture_s" in weak_keys:
    if isinstance(cap, (int, float)) and cap < 0.60:
        actions.append(
            "Capture is under 60%: increase available nights, reduce blocked dates, revisit minimum-stay rules, and tune pricing to close the gap.")
    else:
        actions.append(
            "Improve revenue capture: align calendar availability and pricing so actual revenue approaches predicted potential.")

if "amenity_s" in weak_keys:
    if has_ac is False:
        actions.append(
            "Amenity upgrade: add **A/C first** (largest experience uplift), then consider hot tub/pool/waterfront/views depending on market.")
    else:
        actions.append(
            "Amenity upgrade: add/upgrade high-impact amenities (hot tub, pool, waterfront/beach access, premium views) to justify higher ADR.")

if "review_s" in weak_keys:
    # your senior insight: few reviews + good rating is not bad -> so don’t punish
    actions.append("Review strategy: keep rating high, increase review volume gently (post-stay nudges, faster issue resolution). Low reviews with high rating is fine — focus on consistency.")

if "occ_gap_s" in weak_keys:
    actions.append(
        "Occupancy boost: enable Instant Book (if possible), improve photos/title, and adjust seasonal pricing to increase booked nights.")

if "adr_gap_s" in weak_keys:
    actions.append(
        "ADR boost: reposition listing around premium features, tighten discounting, and highlight experience amenities to command higher nightly rates.")

if "trust_s" in weak_keys:
    actions.append(
        "Trust boost: improve response time and booking friction; aim for Superhost/Guest Favorite signals to increase conversion.")

# Always add a smart “investor upside” message if capture is decent but revenue potential is large
rp = perf.get("revenue_potential") if perf else None
rev = perf.get("revenue") if perf else None
if isinstance(rp, (int, float)) and isinstance(rev, (int, float)) and rp > 0:
    cap2 = rev / rp
    if cap2 >= 0.60 and cap2 < 0.95:
        actions.append(
            "This listing already captures ≥60% of its potential — it’s a good ‘invest now, optimize to grow’ candidate if upgrades are manageable.")

# De-dupe, show top 3-5
seen = set()
final_actions = []
for a in actions:
    if a not in seen:
        seen.add(a)
        final_actions.append(a)

for a in final_actions[:5]:
    st.write("✅ " + a)
