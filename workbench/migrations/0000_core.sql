CREATE TABLE IF NOT EXISTS project(id TEXT PRIMARY KEY,name TEXT NOT NULL,revision INTEGER NOT NULL,
created_at TEXT NOT NULL,updated_at TEXT NOT NULL,classes TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS assets(id TEXT PRIMARY KEY,name TEXT NOT NULL,width INTEGER NOT NULL,
height INTEGER NOT NULL,sha256 TEXT NOT NULL UNIQUE,image_file TEXT NOT NULL,
batch_id TEXT NOT NULL,split TEXT NOT NULL,source TEXT NOT NULL,
revision INTEGER NOT NULL,review_state TEXT NOT NULL,shapes TEXT NOT NULL,
created_at TEXT NOT NULL,updated_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS history(id INTEGER PRIMARY KEY,asset_id TEXT NOT NULL,
revision INTEGER NOT NULL,action TEXT NOT NULL,created_at TEXT NOT NULL,data TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS history_asset_id ON history(asset_id);
CREATE TABLE IF NOT EXISTS exports(id TEXT PRIMARY KEY,created_at TEXT NOT NULL,data TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS dataset_versions(
id TEXT PRIMARY KEY NOT NULL CHECK(id GLOB 'D[0-9]*'),
project_revision INTEGER NOT NULL,
created_at TEXT NOT NULL,
manifest_sha256 TEXT NOT NULL CHECK(length(manifest_sha256) = 64),
asset_count INTEGER NOT NULL CHECK(asset_count >= 0),
relative_path TEXT NOT NULL UNIQUE,
source_dataset_id TEXT,
FOREIGN KEY(source_dataset_id) REFERENCES dataset_versions(id)
);
CREATE TABLE IF NOT EXISTS training_runs(
id TEXT PRIMARY KEY NOT NULL CHECK(id GLOB 'R[0-9]*'),
dataset_version_id TEXT NOT NULL,
proposed_model_id TEXT,
engine TEXT NOT NULL,
status TEXT NOT NULL CHECK(status IN ('queued','preparing','running','stopping','completed','failed','stopped')),
created_at REAL NOT NULL,
updated_at REAL NOT NULL,
relative_path TEXT NOT NULL UNIQUE,
FOREIGN KEY(dataset_version_id) REFERENCES dataset_versions(id)
);
CREATE TABLE IF NOT EXISTS model_versions(
id TEXT PRIMARY KEY NOT NULL CHECK(id GLOB 'M[0-9]*'),
run_id TEXT NOT NULL UNIQUE,
dataset_version_id TEXT NOT NULL,
engine TEXT NOT NULL,
created_at REAL NOT NULL,
relative_path TEXT NOT NULL UNIQUE,
FOREIGN KEY(run_id) REFERENCES training_runs(id),
FOREIGN KEY(dataset_version_id) REFERENCES dataset_versions(id)
);
CREATE TABLE IF NOT EXISTS model_exports(
id TEXT PRIMARY KEY NOT NULL CHECK(id GLOB 'E[0-9]*'),
model_version_id TEXT NOT NULL,
run_id TEXT NOT NULL,
dataset_version_id TEXT NOT NULL,
created_at TEXT NOT NULL,
sha256 TEXT NOT NULL CHECK(length(sha256) = 64),
byte_count INTEGER NOT NULL CHECK(byte_count >= 0),
relative_path TEXT NOT NULL UNIQUE,
FOREIGN KEY(model_version_id) REFERENCES model_versions(id),
FOREIGN KEY(run_id) REFERENCES training_runs(id),
FOREIGN KEY(dataset_version_id) REFERENCES dataset_versions(id)
);
CREATE TABLE IF NOT EXISTS prediction_candidates(
id TEXT PRIMARY KEY NOT NULL CHECK(length(id) = 32),
model_version_id TEXT NOT NULL,
run_id TEXT NOT NULL,
created_at TEXT NOT NULL,
candidate_count INTEGER NOT NULL CHECK(candidate_count >= 0),
accepted_count INTEGER NOT NULL CHECK(accepted_count >= 0),
relative_path TEXT NOT NULL UNIQUE,
FOREIGN KEY(model_version_id) REFERENCES model_versions(id),
FOREIGN KEY(run_id) REFERENCES training_runs(id)
);
CREATE INDEX IF NOT EXISTS idx_training_runs_dataset ON training_runs(dataset_version_id);
CREATE INDEX IF NOT EXISTS idx_models_dataset ON model_versions(dataset_version_id);
CREATE INDEX IF NOT EXISTS idx_exports_model ON model_exports(model_version_id);
CREATE INDEX IF NOT EXISTS idx_predictions_model ON prediction_candidates(model_version_id);
