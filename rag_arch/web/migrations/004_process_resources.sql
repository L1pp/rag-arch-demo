-- Persist worker-owned operating-system resources and their task use history.
-- A worker's PID alone is not proof that child Office/PDF processes stopped.

CREATE TABLE process_resources (
    resource_id TEXT PRIMARY KEY CHECK (char_length(resource_id) BETWEEN 1 AND 128),
    worker_instance TEXT NOT NULL
        REFERENCES worker_instances(worker_instance) ON DELETE RESTRICT,
    kind TEXT NOT NULL CHECK (char_length(kind) BETWEEN 1 AND 64),
    scope TEXT NOT NULL CHECK (scope IN ('worker', 'task')),
    task_id TEXT REFERENCES tasks(task_id) ON DELETE RESTRICT,
    owner_path TEXT,
    owner_descriptor_path TEXT CHECK (owner_descriptor_path IS NULL OR char_length(owner_descriptor_path) BETWEEN 1 AND 4096),
    owner_nonce TEXT CHECK (owner_nonce IS NULL OR char_length(owner_nonce) BETWEEN 1 AND 128),
    state TEXT NOT NULL DEFAULT 'starting'
        CHECK (state IN ('starting', 'active', 'cleanup_pending', 'exited', 'unknown')),
    host_id TEXT,
    pid BIGINT CHECK (pid IS NULL OR pid > 0),
    process_start_token TEXT,
    executable TEXT,
    error_code TEXT,
    created_at DOUBLE PRECISION NOT NULL,
    updated_at DOUBLE PRECISION NOT NULL,
    exited_at DOUBLE PRECISION,
    CHECK (
        (scope = 'task' AND task_id IS NOT NULL)
        OR scope = 'worker'
    ),
    CHECK (
        (host_id IS NULL AND pid IS NULL AND process_start_token IS NULL)
        OR (host_id IS NOT NULL AND pid IS NOT NULL AND process_start_token IS NOT NULL)
    ),
    CHECK (
        (owner_descriptor_path IS NULL AND owner_nonce IS NULL)
        OR (owner_descriptor_path IS NOT NULL AND owner_nonce IS NOT NULL)
    )
);

CREATE INDEX process_resources_worker_state_idx
    ON process_resources(worker_instance, state, created_at);
CREATE INDEX process_resources_task_state_idx
    ON process_resources(task_id, state)
    WHERE scope = 'task';

CREATE TABLE process_resource_uses (
    use_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    resource_id TEXT NOT NULL REFERENCES process_resources(resource_id) ON DELETE RESTRICT,
    worker_instance TEXT NOT NULL
        REFERENCES worker_instances(worker_instance) ON DELETE RESTRICT,
    task_id TEXT NOT NULL REFERENCES tasks(task_id) ON DELETE RESTRICT,
    attempt_token TEXT NOT NULL CHECK (char_length(attempt_token) BETWEEN 1 AND 128),
    acquired_at DOUBLE PRECISION NOT NULL,
    released_at DOUBLE PRECISION,
    UNIQUE (resource_id, task_id, attempt_token),
    CHECK (released_at IS NULL OR released_at >= acquired_at)
);

CREATE INDEX process_resource_uses_task_active_idx
    ON process_resource_uses(task_id, released_at, resource_id);
CREATE INDEX process_resource_uses_worker_idx
    ON process_resource_uses(worker_instance, released_at, resource_id);

CREATE TABLE process_resource_identity_audit (
    identity_audit_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    worker_instance TEXT NOT NULL
        REFERENCES worker_instances(worker_instance) ON DELETE RESTRICT,
    resource_id TEXT NOT NULL
        REFERENCES process_resources(resource_id) ON DELETE RESTRICT,
    owner_descriptor_path TEXT NOT NULL,
    owner_nonce TEXT NOT NULL,
    probe_state TEXT NOT NULL CHECK (probe_state IN ('running', 'exited', 'unknown')),
    evidence JSONB NOT NULL CHECK (jsonb_typeof(evidence) = 'object'),
    created_at DOUBLE PRECISION NOT NULL,
    UNIQUE (resource_id)
);

CREATE INDEX process_resource_identity_audit_worker_created_idx
    ON process_resource_identity_audit(worker_instance, created_at DESC);

CREATE TABLE task_resolution_audit (
    audit_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    worker_instance TEXT NOT NULL
        REFERENCES worker_instances(worker_instance) ON DELETE RESTRICT,
    task_id TEXT NOT NULL REFERENCES tasks(task_id) ON DELETE RESTRICT,
    attempt_token TEXT NOT NULL CHECK (char_length(attempt_token) BETWEEN 1 AND 128),
    terminal_status TEXT NOT NULL CHECK (terminal_status IN ('failed', 'cancelled')),
    proof JSONB NOT NULL CHECK (jsonb_typeof(proof) = 'object'),
    created_at DOUBLE PRECISION NOT NULL,
    UNIQUE (task_id, attempt_token)
);

CREATE INDEX task_resolution_audit_worker_created_idx
    ON task_resolution_audit(worker_instance, created_at DESC);

CREATE FUNCTION reject_append_only_audit_mutation()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    RAISE EXCEPTION 'audit table is append-only';
END;
$$;

CREATE TRIGGER task_resolution_audit_append_only
    BEFORE UPDATE OR DELETE ON task_resolution_audit
    FOR EACH ROW EXECUTE FUNCTION reject_append_only_audit_mutation();

CREATE TRIGGER process_resource_identity_audit_append_only
    BEFORE UPDATE OR DELETE ON process_resource_identity_audit
    FOR EACH ROW EXECUTE FUNCTION reject_append_only_audit_mutation();
