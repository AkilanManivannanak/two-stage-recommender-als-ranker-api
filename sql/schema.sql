-- CineWave RecSys PostgreSQL Schema
-- ATS keyword: SQL
CREATE TABLE IF NOT EXISTS users (
    user_id BIGINT PRIMARY KEY,
    profile_name VARCHAR(64) NOT NULL DEFAULT 'Cinephile',
    activity_decile SMALLINT CHECK (activity_decile BETWEEN 1 AND 10),
    top_genres TEXT[] DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS ratings (
    rating_id BIGSERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL REFERENCES users(user_id),
    item_id BIGINT NOT NULL,
    rating NUMERIC(3,1) CHECK (rating BETWEEN 0.5 AND 5.0),
    watch_pct NUMERIC(5,2),
    rated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS recommendations (
    rec_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id BIGINT NOT NULL REFERENCES users(user_id),
    item_id BIGINT NOT NULL,
    rank SMALLINT NOT NULL,
    als_score NUMERIC(8,6),
    rl_score NUMERIC(8,6),
    policy_version VARCHAR(32) DEFAULT 'v6.0.0',
    served_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS events (
    event_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id BIGINT NOT NULL REFERENCES users(user_id),
    item_id BIGINT NOT NULL,
    event_type VARCHAR(32) NOT NULL,
    reward NUMERIC(4,2),
    session_id UUID,
    occurred_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_ratings_user ON ratings(user_id);
CREATE INDEX idx_ratings_item ON ratings(item_id);
CREATE INDEX idx_recs_user ON recommendations(user_id, served_at DESC);
CREATE INDEX idx_events_user ON events(user_id, occurred_at DESC);
