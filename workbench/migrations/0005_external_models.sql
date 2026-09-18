-- External weights have no local training run or dataset lineage.
CREATE TABLE model_versions_v5(
id TEXT PRIMARY KEY NOT NULL CHECK(id GLOB 'M[0-9]*'),
run_id TEXT UNIQUE,
dataset_version_id TEXT,
engine TEXT NOT NULL,
created_at REAL NOT NULL,
relative_path TEXT NOT NULL UNIQUE,
FOREIGN KEY(run_id) REFERENCES training_runs(id),
FOREIGN KEY(dataset_version_id) REFERENCES dataset_versions(id)
);
CREATE TABLE model_exports_v5(
id TEXT PRIMARY KEY NOT NULL CHECK(id GLOB 'E[0-9]*'),
model_version_id TEXT NOT NULL,
run_id TEXT,
dataset_version_id TEXT,
created_at TEXT NOT NULL,
sha256 TEXT NOT NULL CHECK(length(sha256) = 64),
byte_count INTEGER NOT NULL CHECK(byte_count >= 0),
relative_path TEXT NOT NULL UNIQUE,
FOREIGN KEY(model_version_id) REFERENCES model_versions_v5(id),
FOREIGN KEY(run_id) REFERENCES training_runs(id),
FOREIGN KEY(dataset_version_id) REFERENCES dataset_versions(id)
);
CREATE TABLE prediction_candidates_v5(
id TEXT PRIMARY KEY NOT NULL CHECK(length(id) = 32),
model_version_id TEXT NOT NULL,
run_id TEXT,
created_at TEXT NOT NULL,
candidate_count INTEGER NOT NULL CHECK(candidate_count >= 0),
accepted_count INTEGER NOT NULL CHECK(accepted_count >= 0),
relative_path TEXT NOT NULL UNIQUE,
FOREIGN KEY(model_version_id) REFERENCES model_versions_v5(id),
FOREIGN KEY(run_id) REFERENCES training_runs(id)
);
INSERT INTO model_versions_v5 SELECT * FROM model_versions;
INSERT INTO model_exports_v5 SELECT * FROM model_exports;
INSERT INTO prediction_candidates_v5 SELECT * FROM prediction_candidates;
DROP TABLE prediction_candidates;
DROP TABLE model_exports;
DROP TABLE model_versions;
ALTER TABLE model_versions_v5 RENAME TO model_versions;
ALTER TABLE model_exports_v5 RENAME TO model_exports;
ALTER TABLE prediction_candidates_v5 RENAME TO prediction_candidates;
CREATE INDEX idx_models_dataset ON model_versions(dataset_version_id);
CREATE INDEX idx_exports_model ON model_exports(model_version_id);
CREATE INDEX idx_predictions_model ON prediction_candidates(model_version_id);
