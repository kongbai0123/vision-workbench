CREATE TABLE workflow_drafts (
scope TEXT PRIMARY KEY CHECK(scope = 'preparation-training'),
schema_version INTEGER NOT NULL CHECK(schema_version = 1),
revision INTEGER NOT NULL CHECK(revision > 0),
updated_at TEXT NOT NULL,
payload TEXT NOT NULL CHECK(json_valid(payload))
);
CREATE TABLE workflow_draft_history (
scope TEXT NOT NULL,
revision INTEGER NOT NULL,
updated_at TEXT NOT NULL,
payload TEXT NOT NULL CHECK(json_valid(payload)),
PRIMARY KEY(scope, revision),
FOREIGN KEY(scope) REFERENCES workflow_drafts(scope)
);
