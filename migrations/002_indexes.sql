create index if not exists idx_properties_core_market_id
  on public.properties_core(market_id);

create index if not exists idx_properties_core_market_property
  on public.properties_core(market_id, property_id);

create index if not exists idx_property_features_bedrooms
  on public.property_features(bedrooms);

create index if not exists idx_property_features_bedrooms_property
  on public.property_features(bedrooms, property_id);

create index if not exists idx_performance_revenue_desc
  on public.performance(revenue desc);

create index if not exists idx_performance_property_id
  on public.performance(property_id);

create index if not exists idx_property_scores_score_desc
  on public.property_scores(investment_score desc);

create index if not exists idx_property_scores_market_score_desc
  on public.property_scores(market_id, investment_score desc);
