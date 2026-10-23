-- Durable worker identities and bounded lease metadata for the task queue.
-- Task state and event history remain in the schema introduced by 001.

CREATE TABLE worker_instances (
    worker_instance TEXT PRIMARY KEY CHECK (char_length(worker_instance) BETWEEN 1 AND 128),
    host_id TEXT NOT NULL CHECK (char_length(host_id) BETWEEN 1 AND 255),
    pid BIGINT NOT NULL CHECK (pid > 0),
    process_start_token TEXT NOT NULL CHECK (char_length(process_start_token) BETWEEN 1 AND 255),
    state TEXT NOT NULL DEFAULT 'active'
        CHECK (state IN ('active', 'stale', 'exited', 'stopped')),
    heartbeat_at DOUBLE PRECISION NOT NULL,
    lease_seconds DOUBLE PRECISION NOT NULL DEFAULT 15 CHECK (lease_seconds > 0),
    registered_at DOUBLE PRECISION NOT NULL,
    stale_at DOUBLE PRECISION,
    exited_at DOUBLE PRECISION,
    stopped_at DOUBLE PRECISION,
    UNIQUE (host_id, pid, process_start_token)
);

ALTER TABLE tasks
    ADD CONSTRAINT tasks_worker_instance_fk
    FOREIGN KEY (worker_instance)
    REFERENCES worker_instances(worker_instance);

CREATE INDEX worker_instances_heartbeat_idx
    ON worker_instances(state, heartbeat_at);
CREATE INDEX tasks_worker_active_idx
    ON tasks(worker_instance, status)
    WHERE status IN ('running', 'cancelling', 'recovering');
CREATE INDEX tasks_owner_history_idx
    ON tasks(user_id, assistant_id, generation, created_at)
    WHERE status = 'completed';
