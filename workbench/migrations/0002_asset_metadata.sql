CREATE TABLE asset_review(asset_id TEXT PRIMARY KEY, data TEXT NOT NULL);
CREATE TABLE asset_quality(asset_id TEXT PRIMARY KEY, data TEXT NOT NULL);
CREATE TABLE annotation_revisions(asset_id TEXT PRIMARY KEY, revision INTEGER NOT NULL);
INSERT INTO asset_review SELECT id,json_extract(source,'$.review') FROM assets WHERE json_type(source,'$.review')='object';
INSERT INTO asset_quality SELECT id,json_extract(source,'$.quality') FROM assets WHERE json_type(source,'$.quality')='object';
UPDATE assets SET source=json_remove(source,'$.review','$.quality');
INSERT INTO annotation_revisions SELECT id,revision FROM assets;
CREATE TRIGGER annotation_revision_insert AFTER INSERT ON assets BEGIN INSERT OR IGNORE INTO annotation_revisions VALUES(NEW.id,1); END;
CREATE TRIGGER annotation_revision_update AFTER UPDATE OF shapes ON assets WHEN OLD.shapes<>NEW.shapes BEGIN UPDATE annotation_revisions SET revision=revision+1 WHERE asset_id=NEW.id; END;
