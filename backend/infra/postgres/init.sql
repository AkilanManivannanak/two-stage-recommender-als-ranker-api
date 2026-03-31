-- Netflix-Inspired RecSys — PostgreSQL Schema
-- =============================================
-- Postgres is the SOURCE OF TRUTH for:
--   users, items, interactions, features,
--   experiments, audit logs, API state
-- Redis  = online hot features (rebuilt from here on restart)
-- MinIO  = artifacts, embeddings, Parquet exports
-- Qdrant = vector index (rebuilt from MinIO on restart)

-- ── Extensions ───────────────────────────────────────────────────────
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- trigram similarity for text search

-- ── Users ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS users (
    user_id         SERIAL PRIMARY KEY,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    last_active_at  TIMESTAMPTZ,
    -- Aggregated features (updated by feature pipeline, served via Redis)
    total_ratings   INT     DEFAULT 0,
    avg_rating      FLOAT   DEFAULT 3.5,
    primary_genres  TEXT[], -- top 3 inferred genre preferences
    tenure_days     INT     DEFAULT 0,
    is_cold_start   BOOLEAN DEFAULT TRUE
);

-- ── Items / Catalog ───────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS items (
    item_id         SERIAL PRIMARY KEY,
    tmdb_id         INT,
    title           TEXT NOT NULL,
    primary_genre   TEXT,
    genres          TEXT[],
    year            INT,
    maturity_rating TEXT,
    avg_rating      FLOAT   DEFAULT 3.5,
    vote_count      INT     DEFAULT 0,
    popularity      FLOAT   DEFAULT 1.0,
    runtime_min     INT,
    description     TEXT,
    poster_url      TEXT,
    backdrop_url    TEXT,
    -- LLM enrichment fields (written by llm_enrichment_worker)
    themes          TEXT[],
    moods           TEXT[],
    semantic_tags   TEXT[],
    spoiler_summary TEXT,
    pacing          TEXT,
    -- Artwork audit (written by vlm_audit_worker)
    artwork_trust_score     FLOAT   DEFAULT 1.0,
    artwork_flagged         BOOLEAN DEFAULT FALSE,
    artwork_audit_note      TEXT,
    -- Embedding reference (actual vectors in Qdrant + MinIO)
    embedding_version       TEXT,
    is_cold_start           BOOLEAN DEFAULT TRUE,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

-- ── Interactions (raw event log) ──────────────────────────────────────
CREATE TABLE IF NOT EXISTS interactions (
    id          BIGSERIAL PRIMARY KEY,
    user_id     INT REFERENCES users(user_id),
    item_id     INT REFERENCES items(item_id),
    event       TEXT NOT NULL,  -- play, pause, like, dislike, search, impression
    duration_s  FLOAT,          -- how long they watched
    position_s  FLOAT,          -- where they stopped
    context     JSONB,          -- device, session_id, page position, etc.
    created_at  TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_interactions_user ON interactions(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_interactions_item ON interactions(item_id, created_at DESC);

-- ── Feature registry (what features exist and where they are stored) ──
CREATE TABLE IF NOT EXISTS feature_registry (
    feature_name    TEXT PRIMARY KEY,
    feature_type    TEXT,       -- user, item, user_item
    store           TEXT,       -- redis, postgres, minio, qdrant
    ttl_seconds     INT,
    description     TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
INSERT INTO feature_registry VALUES
    ('user_recent_genres',    'user',      'redis',    300,  'Genres active in last 5min session'),
    ('user_avg_rating',       'user',      'postgres', 86400,'User historical avg rating'),
    ('user_interaction_count','user',      'postgres', 86400,'Total interaction count'),
    ('item_trending_score',   'item',      'redis',    60,   'Rolling 5min trending score'),
    ('item_avg_rating',       'item',      'postgres', 86400,'Item average rating'),
    ('item_embedding',        'item',      'qdrant',   NULL, 'Fused multimodal embedding'),
    ('session_intent',        'user',      'redis',    1800, 'Current session intent category')
ON CONFLICT DO NOTHING;

-- ── User features (daily aggregates — source of truth for Redis warm-up) 
CREATE TABLE IF NOT EXISTS user_features_daily (
    user_id         INT REFERENCES users(user_id),
    feature_date    DATE DEFAULT CURRENT_DATE,
    total_plays     INT     DEFAULT 0,
    total_likes     INT     DEFAULT 0,
    total_abandons  INT     DEFAULT 0,
    genre_counts    JSONB,  -- {"Crime": 12, "Drama": 8, ...}
    avg_watch_s     FLOAT,
    cold_start_flag BOOLEAN DEFAULT TRUE,
    PRIMARY KEY (user_id, feature_date)
);

-- ── Item features (daily aggregates) ──────────────────────────────────
CREATE TABLE IF NOT EXISTS item_features_daily (
    item_id         INT REFERENCES items(item_id),
    feature_date    DATE DEFAULT CURRENT_DATE,
    play_count      INT     DEFAULT 0,
    like_count      INT     DEFAULT 0,
    impression_count INT    DEFAULT 0,
    avg_watch_pct   FLOAT,  -- avg % of content watched
    PRIMARY KEY (item_id, feature_date)
);

-- ── Experiment tracking ───────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS experiments (
    id              SERIAL PRIMARY KEY,
    name            TEXT NOT NULL,
    description     TEXT,
    status          TEXT DEFAULT 'draft', -- draft, running, completed, archived
    config          JSONB,    -- experiment configuration
    metrics_before  JSONB,
    metrics_after   JSONB,
    agent_summary   TEXT,     -- generated by agentic_ops
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    completed_at    TIMESTAMPTZ
);

-- ── Model versions ─────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS model_versions (
    id              SERIAL PRIMARY KEY,
    version_hash    TEXT NOT NULL UNIQUE,
    run_id          TEXT,
    metrics         JSONB,
    artifact_path   TEXT,   -- path in MinIO
    status          TEXT DEFAULT 'candidate', -- candidate, shadow, production, retired
    agent_triage    JSONB,  -- agent recommendation
    policy_result   JSONB,  -- policy gate result
    deployed_at     TIMESTAMPTZ,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- ── Feedback log ───────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS feedback_log (
    id          BIGSERIAL PRIMARY KEY,
    user_id     INT,
    item_id     INT,
    event       TEXT,
    context     JSONB,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_feedback_user ON feedback_log(user_id, created_at DESC);

-- ── Audit log (artwork, policy, agent decisions) ───────────────────────
CREATE TABLE IF NOT EXISTS audit_log (
    id          BIGSERIAL PRIMARY KEY,
    entity_type TEXT,   -- item, model, experiment
    entity_id   TEXT,
    action      TEXT,
    result      JSONB,
    requires_human_review BOOLEAN DEFAULT TRUE,
    reviewed_by TEXT,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- ── Recommendation log (sampled — not every request) ──────────────────
CREATE TABLE IF NOT EXISTS recommendation_log (
    id          BIGSERIAL PRIMARY KEY,
    user_id     INT,
    items       INT[],     -- top-10 item IDs shown
    row_title   TEXT,
    session_intent TEXT,
    latency_ms  FLOAT,
    model_version TEXT,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_rec_log_user ON recommendation_log(user_id, created_at DESC);
