-- TriStore bootstrap for AGE/Cypher + minimal relational tables

-- Enable AGE extension (requires superuser privileges and AGE installed)
-- CREATE EXTENSION IF NOT EXISTS age;
-- LOAD 'age';

-- Create graph and set path
-- Note: If using AGE 1.x, use: SELECT create_graph('hob');
CREATE GRAPH hob;
SET graph_path = hob;

-- Minimal outbox table for events (optional)
CREATE TABLE IF NOT EXISTS hobnob_outbox (
  id uuid PRIMARY KEY,
  instance_id text NOT NULL,
  kind text NOT NULL,
  payload jsonb NOT NULL,
  created_at timestamptz DEFAULT now(),
  delivered_at timestamptz
);

-- Minimal idempotency table (no policy logic here)
CREATE TABLE IF NOT EXISTS hobnob_idempotency (
  key text PRIMARY KEY,
  seen_at timestamptz DEFAULT now()
);

