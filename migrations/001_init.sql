create extension if not exists pgcrypto;

create table if not exists public.markets (
  market_id uuid primary key default gen_random_uuid(),
  market_name text not null unique,
  city text,
  state text,
  created_at timestamptz not null default now()
);

create table if not exists public.properties_core (
  property_id text primary key,
  market_id uuid not null references public.markets(market_id) on delete cascade,

  title text,
  host_id text,

  airbnb_listing_url text,
  vrbo_listing_url text,
  airbnb_host_url text,

  city text,
  state text,
  zipcode text,

  latitude double precision,
  longitude double precision,

  description text,
  amenities_text text,

  raw jsonb,

  ingested_at timestamptz not null default now()
);

create table if not exists public.performance (
  property_id text primary key references public.properties_core(property_id) on delete cascade,

  revenue numeric,
  revenue_potential numeric,
  adr numeric,
  occupancy numeric,
  cleaning_fee numeric,
  available_nights integer,

  price_tier text,

  total_reviews integer,
  property_rating numeric,
  stars numeric,

  updated_at timestamptz not null default now()
);

create table if not exists public.property_features (
  property_id text primary key references public.properties_core(property_id) on delete cascade,

  bedrooms integer,
  accommodates integer,
  bathrooms numeric,
  minimum_stay integer,

  -- Boolean flags (58-column clean set)
  has_aircon boolean,
  has_gym boolean,
  has_hottub boolean,
  has_kitchen boolean,
  has_parking boolean,
  has_pets_allowed boolean,
  has_pool boolean,
  has_beach_access boolean,
  has_lake_access boolean,
  has_outdoor_dining_area boolean,
  has_outdoor_furniture boolean,
  has_waterfront boolean,
  instant_book boolean,
  superhost boolean,
  system_arcade_machine boolean,
  system_bowling boolean,
  system_chess boolean,
  system_crib boolean,
  system_firepit boolean,
  system_golf boolean,
  system_grill boolean,
  system_jacuzzi boolean,
  system_movie boolean,
  system_pack_n_play boolean,
  system_play_slide boolean,
  system_pool boolean,
  system_pool_table boolean,
  system_view_mountain boolean,
  system_view_ocean boolean,
  is_guest_favorite boolean,
  is_super_host boolean,
  updated_at timestamptz not null default now()
);

create table if not exists public.property_scores (
  property_id text primary key references public.properties_core(property_id) on delete cascade,
  market_id uuid references public.markets(market_id) on delete set null,

  investment_score numeric,
  score_breakdown jsonb,
  computed_at timestamptz not null default now()
);
