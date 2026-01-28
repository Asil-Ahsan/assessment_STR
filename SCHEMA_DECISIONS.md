# Schema Decisions

## Goal
Investor-facing insights with low-noise, queryable structure:
- Fast filtering (market, bedrooms, tier, amenities)
- Repeatable scoring updates and explainability
- Minimal joins required for API performance

## Tables
### `markets`
One row per market: id + display fields.

### `properties_core`
Stable identity + location + core metadata.
- `property_id` (text PK)
- `market_id` (FK)
- title/city/state/zipcode + lat/lon + URLs

### `performance`
Aggregated performance metrics used for scoring.
- revenue, revenue_potential, adr, occupancy
- review summary: total_reviews, property_rating/stars
- price_tier, available_nights, cleaning_fee

### `property_features`
Mostly boolean amenities + capacity/size.
- bedrooms/bathrooms/accommodates/minimum_stay
- amenities + flags (A/C, hot tub, waterfront, views, etc.)
- trust flags (superhost/instant_book/guest_favorite)

### `property_scores`
Computed scores + explanation payload.
- investment_score (0–100)
- score_breakdown (jsonb)
- computed_at (timestamptz)

## Why split this way?
- Avoid repeated columns from raw CSVs
- Keep boolean-heavy fields isolated and filterable
- Recompute scoring without rewriting core/performance/features

## Indexing (recommended)
- `properties_core(market_id)`
- `property_features(bedrooms)`
- `performance(price_tier)`
- `property_scores(investment_score desc)`, `property_scores(market_id)`
