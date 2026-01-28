# Scoring Logic

## Intuition
Score = “How investable is this listing?”
We combine:
1) Performance vs similar listings (same market + bedrooms)
2) Headroom / upside (revenue vs potential capture)
3) Guest experience signal (rating + review count handled safely)
4) Amenities package strength
5) Trust/friction flags (superhost, instant book, guest favorite)

## Buckets
A *bucket* = (market_id, bedrooms)
We normalize within buckets so a 3BR is compared to other 3BRs in the same market.

## Normalization
Within each bucket:
- Winsorize (1%/99%) to reduce outlier dominance
- Percentile rank → [0,1]

## Components
- rev_gap_s, adr_gap_s, occ_gap_s: gaps vs bucket averages
- capture_s: smooth boost around 60% capture, then normalize
- review_s: Bayesian-smoothed rating + confidence floor, then normalize
- amenity_s: weighted amenity sum scaled by size, then normalize
- trust_s: small trust signal, then normalize

## Final score
Weighted sum in [0,1], then curved mapping:
`investment_score = (10 + 90*(score01^0.65)) * ac_penalty`
Clipped to [0,100].

## Explainability
`property_scores.score_breakdown` stores raw gaps, normalized factors, and weights.
