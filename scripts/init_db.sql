CREATE TABLE IF NOT EXISTS reviewed_comments (
    id SERIAL PRIMARY KEY,
    comment_text TEXT NOT NULL,
    original_predictions JSONB,
    reviewed_labels JSONB,
    moderator_id VARCHAR(100),
    reviewed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW(),
    source VARCHAR(50) DEFAULT 'api',
    model_version VARCHAR(50),
    status VARCHAR(20) DEFAULT 'pending'
);

CREATE INDEX IF NOT EXISTS idx_reviewed_comments_status 
    ON reviewed_comments(status);

CREATE INDEX IF NOT EXISTS idx_reviewed_comments_reviewed_at 
    ON reviewed_comments(reviewed_at);

CREATE INDEX IF NOT EXISTS idx_reviewed_comments_source 
    ON reviewed_comments(source);

CREATE INDEX IF NOT EXISTS idx_reviewed_comments_created_at 
    ON reviewed_comments(created_at);

CREATE DATABASE IF NOT EXISTS mlflow;

GRANT ALL PRIVILEGES ON DATABASE reviews TO dbadmin;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO dbadmin;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO dbadmin;

COMMENT ON TABLE reviewed_comments IS 'Stores comments flagged for manual review and their moderator labels';
COMMENT ON COLUMN reviewed_comments.original_predictions IS 'Model predictions at time of flagging (JSONB)';
COMMENT ON COLUMN reviewed_comments.reviewed_labels IS 'Labels assigned by moderator (JSONB)';
COMMENT ON COLUMN reviewed_comments.status IS 'Status: pending, reviewed, skipped';
