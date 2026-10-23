-- Shared login rate-limit state for independent API processes.
-- Keys and reservation tickets are application-side SHA-256 digests; raw
-- addresses, usernames, cookies, and password material are never stored here.

CREATE TABLE login_rate_limit_failures (
    failure_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    key_hash TEXT NOT NULL CHECK (key_hash ~ '^[0-9a-f]{64}$'),
    failed_at DOUBLE PRECISION NOT NULL
);
CREATE INDEX login_rate_limit_failures_key_time_idx
    ON login_rate_limit_failures(key_hash, failed_at);
CREATE INDEX login_rate_limit_failures_time_idx
    ON login_rate_limit_failures(failed_at);

CREATE TABLE login_rate_limit_attempts (
    key_hash TEXT NOT NULL CHECK (key_hash ~ '^[0-9a-f]{64}$'),
    ticket_hash TEXT NOT NULL CHECK (ticket_hash ~ '^[0-9a-f]{64}$'),
    reserved_at DOUBLE PRECISION NOT NULL,
    expires_at DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (key_hash, ticket_hash),
    CHECK (expires_at > reserved_at)
);
CREATE INDEX login_rate_limit_attempts_key_expiry_idx
    ON login_rate_limit_attempts(key_hash, expires_at);
CREATE INDEX login_rate_limit_attempts_expiry_idx
    ON login_rate_limit_attempts(expires_at);
