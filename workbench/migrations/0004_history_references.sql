ALTER TABLE history ADD COLUMN annotation_hash TEXT;
UPDATE history SET annotation_hash=json_extract(data,'$.annotation_hash');
CREATE INDEX history_annotation_hash ON history(annotation_hash);
DELETE FROM asset_summaries WHERE asset_id NOT IN (SELECT id FROM assets UNION SELECT id FROM review_trash);
DELETE FROM annotation_revisions WHERE asset_id NOT IN (SELECT id FROM assets UNION SELECT id FROM review_trash);
DELETE FROM asset_review WHERE asset_id NOT IN (SELECT id FROM assets UNION SELECT id FROM review_trash);
DELETE FROM asset_quality WHERE asset_id NOT IN (SELECT id FROM assets UNION SELECT id FROM review_trash);
DELETE FROM annotation_blobs WHERE NOT EXISTS (SELECT 1 FROM history WHERE history.annotation_hash=annotation_blobs.hash);
