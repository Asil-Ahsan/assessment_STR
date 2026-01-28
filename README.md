# STR Search Take-Home (Data Engineer)

This repo contains:
- CSV cleaning + ingestion into Supabase
- A scoring pipeline that computes an **investment_score** per property and upserts into `property_scores`
- A FastAPI service that serves investor-friendly endpoints
- (Bonus) A Streamlit dashboard that calls the API

## Quickstart (local)

### 1) Create env + install deps
**Option A — venv**
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Mac/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2) Configure environment
Copy:
```bash
cp .env.example .env
```
Fill in:
- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY`

### 3) Data
Place the 3 source CSVs under `data/` (or keep them where your pipeline expects).
Cleaned CSVs used for ingestion are under `data/clean/`.

### 4) Run pipeline
From repo root:

**Clean CSVs (if needed)**
```bash
python -m src.pipeline.clean_csvs
```

**Ingest cleaned CSVs into Supabase**
```bash
python -m src.pipeline.ingest_clean
```

**Compute scores (REST) + upsert into Supabase**
```bash
python -m src.scoring.score_rest
```

### 5) Run API
Open:

https://strsearch.onrender.com/
### 6) Run Dashboard (bonus)
```bash
https://assessment-str.onrender.com
```
## Deliverables
- SQL optimization query + EXPLAIN ANALYZE: `sql/sql_optimization.sql`
- Schema decisions: `SCHEMA_DECISIONS.md`
- Scoring logic: `SCORING_LOGIC.md`

### Dashboard
https://assessment-str.onrender.com
https://strsearch.onrender.com/
### VIDEO LINK
https://youtu.be/DFjJSuy5LzM
