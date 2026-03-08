-- Migration 001: RBAC tables for existing databases
-- Run this manually if you already have a running docling_rag database.
-- New installations get these tables from docker/init.sql automatically.

BEGIN;

CREATE TABLE IF NOT EXISTS groups (
    id          SERIAL PRIMARY KEY,
    name        TEXT UNIQUE NOT NULL,
    description TEXT DEFAULT '',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS users (
    id                   SERIAL PRIMARY KEY,
    username             TEXT UNIQUE NOT NULL,
    password_hash        TEXT,
    display_name         TEXT DEFAULT '',
    email                TEXT DEFAULT '',
    is_admin             BOOLEAN NOT NULL DEFAULT FALSE,
    is_active            BOOLEAN NOT NULL DEFAULT TRUE,
    must_change_password BOOLEAN NOT NULL DEFAULT FALSE,
    auth_type            TEXT NOT NULL DEFAULT 'local'
        CHECK (auth_type IN ('local', 'ldap')),
    created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS user_groups (
    user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    group_id   INTEGER NOT NULL REFERENCES groups(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (user_id, group_id)
);

CREATE TABLE IF NOT EXISTS path_permissions (
    id         SERIAL PRIMARY KEY,
    path       TEXT NOT NULL,
    group_id   INTEGER NOT NULL REFERENCES groups(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (path, group_id)
);

CREATE TABLE IF NOT EXISTS document_permissions (
    doc_id     TEXT NOT NULL,
    group_id   INTEGER NOT NULL REFERENCES groups(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (doc_id, group_id)
);

CREATE INDEX IF NOT EXISTS idx_user_groups_user ON user_groups (user_id);
CREATE INDEX IF NOT EXISTS idx_user_groups_group ON user_groups (group_id);
CREATE INDEX IF NOT EXISTS idx_path_permissions_path ON path_permissions (path);
CREATE INDEX IF NOT EXISTS idx_path_permissions_group ON path_permissions (group_id);
CREATE INDEX IF NOT EXISTS idx_document_permissions_doc ON document_permissions (doc_id);
CREATE INDEX IF NOT EXISTS idx_document_permissions_group ON document_permissions (group_id);

-- Default admin user (password: p@ssw0rd)
INSERT INTO users (username, password_hash, display_name, is_admin, auth_type)
VALUES ('admin', '$2b$12$3J2nk5UcD5Jb5.6eRVeo0eJudwcjThFv1MQ4ajFMT6PykIgofezgK', 'Administrator', TRUE, 'local')
ON CONFLICT (username) DO NOTHING;

COMMIT;
