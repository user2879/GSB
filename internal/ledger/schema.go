package ledger

const ledgerFormatVersion = 1

const schemaSQL = `
CREATE TABLE IF NOT EXISTS ledger_meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY,
  format_version INTEGER NOT NULL,
  spec_version INTEGER NOT NULL,
  policy_version INTEGER NOT NULL,
  spec_digest TEXT NOT NULL,
  policy_digest TEXT NOT NULL,
  manifest_json TEXT NOT NULL,
  manifest_digest TEXT NOT NULL,
  options_json TEXT NOT NULL,
  options_digest TEXT NOT NULL,
  semantic_digest TEXT NOT NULL,
  stream_digest TEXT NOT NULL,
  change_set_json TEXT NOT NULL,
  status TEXT NOT NULL,
  is_current INTEGER NOT NULL,
  stale INTEGER NOT NULL,
  created_ms INTEGER NOT NULL,
  updated_ms INTEGER NOT NULL,
  completed_windows INTEGER NOT NULL DEFAULT 0,
  total_windows INTEGER NOT NULL DEFAULT 0,
  progress_window_start_ms INTEGER,
  ledger_content_digest TEXT NOT NULL,
  report_json TEXT,
  report_digest TEXT,
  error_json TEXT
);

CREATE TABLE IF NOT EXISTS scheduler_segments (
  run_id TEXT NOT NULL,
  segment_id TEXT NOT NULL,
  started_wall_ms INTEGER NOT NULL,
  clock_basis TEXT NOT NULL,
  PRIMARY KEY (run_id, segment_id),
  FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS window_tasks (
  run_id TEXT NOT NULL,
  window_start_ms INTEGER NOT NULL,
  window_end_ms INTEGER NOT NULL,
  ordinal INTEGER NOT NULL,
  status TEXT NOT NULL,
  attempts INTEGER NOT NULL DEFAULT 0,
  recovered_from_unknown INTEGER NOT NULL DEFAULT 0,
  segment_id TEXT,
  queue_start_mono_ms INTEGER,
  start_mono_ms INTEGER,
  end_mono_ms INTEGER,
  queue_duration_ms INTEGER,
  duration_ms INTEGER,
  global_quota INTEGER NOT NULL,
  key_quota_json TEXT NOT NULL,
  result_digest TEXT,
  error_json TEXT,
  PRIMARY KEY (run_id, window_start_ms),
  FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS event_decisions (
  run_id TEXT NOT NULL,
  window_start_ms INTEGER NOT NULL,
  source_id TEXT NOT NULL,
  event_seq INTEGER NOT NULL,
  event_id TEXT NOT NULL,
  timestamp_ms INTEGER NOT NULL,
  client_key TEXT NOT NULL,
  weight INTEGER NOT NULL,
  strategy_id TEXT NOT NULL,
  policy_version INTEGER NOT NULL,
  bucket INTEGER NOT NULL,
  threshold INTEGER NOT NULL,
  sampled INTEGER NOT NULL,
  decision_digest TEXT NOT NULL,
  PRIMARY KEY (run_id, window_start_ms, source_id, event_seq),
  FOREIGN KEY (run_id, window_start_ms) REFERENCES window_tasks(run_id, window_start_ms) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS window_results (
  run_id TEXT NOT NULL,
  window_start_ms INTEGER NOT NULL,
  window_end_ms INTEGER NOT NULL,
  total_event_count INTEGER NOT NULL,
  sampled_event_count INTEGER NOT NULL,
  unseen_event_count INTEGER NOT NULL,
  sample_set_digest TEXT NOT NULL,
  decision_set_digest TEXT NOT NULL,
  aggregation_digest TEXT NOT NULL,
  ranking_digest TEXT NOT NULL,
  estimate_applicable INTEGER NOT NULL,
  top_k_guaranteed INTEGER NOT NULL,
  result_json TEXT NOT NULL,
  result_digest TEXT NOT NULL,
  PRIMARY KEY (run_id, window_start_ms),
  FOREIGN KEY (run_id, window_start_ms) REFERENCES window_tasks(run_id, window_start_ms) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS key_aggregates (
  run_id TEXT NOT NULL,
  window_start_ms INTEGER NOT NULL,
  client_key TEXT NOT NULL,
  source_ids_json TEXT NOT NULL,
  sample_count INTEGER NOT NULL,
  sample_weight INTEGER NOT NULL,
  estimated_count_milli INTEGER NOT NULL,
  estimator_lower_milli INTEGER NOT NULL,
  estimator_upper_milli INTEGER NOT NULL,
  observed_count_lower INTEGER NOT NULL,
  observed_count_upper INTEGER NOT NULL,
  rank INTEGER NOT NULL,
  PRIMARY KEY (run_id, window_start_ms, client_key),
  FOREIGN KEY (run_id, window_start_ms) REFERENCES window_tasks(run_id, window_start_ms) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_window_tasks_status ON window_tasks(run_id, status, ordinal);
CREATE INDEX IF NOT EXISTS idx_event_decisions_order ON event_decisions(run_id, window_start_ms, timestamp_ms, source_id, event_seq);
CREATE INDEX IF NOT EXISTS idx_runs_current ON runs(is_current, stale);
`
