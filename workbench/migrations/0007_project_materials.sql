CREATE TABLE project_material_files(
    id TEXT PRIMARY KEY NOT NULL CHECK(length(id) = 32),
    kind TEXT NOT NULL CHECK(kind IN ('video','annotation')),
    relative_path TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    sha256 TEXT NOT NULL CHECK(length(sha256) = 64),
    byte_count INTEGER NOT NULL CHECK(byte_count >= 0),
    metadata TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX project_material_files_kind_hash ON project_material_files(kind,sha256);

CREATE TABLE project_annotation_links(
    annotation_id TEXT NOT NULL REFERENCES project_material_files(id),
    image_id TEXT NOT NULL CHECK(length(image_id) = 32),
    annotation_revision INTEGER NOT NULL CHECK(annotation_revision >= 1),
    source_metadata TEXT NOT NULL,
    PRIMARY KEY(annotation_id,image_id,annotation_revision)
);
CREATE INDEX project_annotation_links_image ON project_annotation_links(image_id);

CREATE TABLE project_image_video_links(
    image_id TEXT NOT NULL CHECK(length(image_id) = 32),
    video_id TEXT NOT NULL REFERENCES project_material_files(id),
    frame_index INTEGER NOT NULL CHECK(frame_index >= 0),
    timestamp_seconds REAL NOT NULL CHECK(timestamp_seconds >= 0),
    timestamp_method TEXT NOT NULL,
    PRIMARY KEY(image_id,video_id,frame_index)
);
CREATE INDEX project_image_video_links_video ON project_image_video_links(video_id);

CREATE TRIGGER project_material_links_remove_image AFTER DELETE ON assets
WHEN NOT EXISTS (SELECT 1 FROM review_trash WHERE id=OLD.id)
BEGIN
    DELETE FROM project_annotation_links WHERE image_id=OLD.id;
    DELETE FROM project_image_video_links WHERE image_id=OLD.id;
END;

CREATE TRIGGER project_material_links_remove_trashed_image AFTER DELETE ON review_trash
WHEN NOT EXISTS (SELECT 1 FROM assets WHERE id=OLD.id)
BEGIN
    DELETE FROM project_annotation_links WHERE image_id=OLD.id;
    DELETE FROM project_image_video_links WHERE image_id=OLD.id;
END;

CREATE VIEW project_materials(
    id,kind,relative_path,name,sha256,byte_count,metadata,created_at,state,width,height,batch_id,review_state
) AS
SELECT id,'image','images/' || image_file,name,sha256,NULL,source,created_at,'active',width,height,batch_id,review_state
FROM assets
UNION ALL
SELECT id,'image','images/' || json_extract(data,'$.image_file'),json_extract(data,'$.name'),
       json_extract(data,'$.sha256'),NULL,json_extract(data,'$.source'),json_extract(data,'$.created_at'),
       'trashed',json_extract(data,'$.width'),json_extract(data,'$.height'),
       json_extract(data,'$.batch_id'),json_extract(data,'$.review_state')
FROM review_trash
UNION ALL
SELECT id,kind,relative_path,name,sha256,byte_count,metadata,created_at,'active',NULL,NULL,NULL,NULL
FROM project_material_files;
