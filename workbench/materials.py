"""Project-owned video, annotation source, and image catalog."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from hashlib import file_digest, sha256 as hash_bytes
import json
import math
from pathlib import Path, PurePosixPath
import re
import stat
from sys import float_info

from .store import ConflictError, dump, identifier, timestamp
from .project_layout import logical_reference, project_area, project_reference


VIDEO_SUFFIXES = frozenset({
    '.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v',
    '.mpg', '.mpeg', '.wmv', '.mts', '.m2ts', '.3gp',
})
MATERIAL_KINDS = frozenset({'image', 'video', 'annotation'})
_DIGEST = re.compile(r'[0-9a-f]{64}')
_FILE_COLUMNS = 'id,kind,relative_path,name,sha256,byte_count,metadata,created_at'


def _owned_relative(value, area):
    if not isinstance(value, str):
        raise ValueError('素材路徑必須是專案相對路徑')
    parts = value.split('/')
    if (len(parts) != 2 or parts[0] != area or parts[1] in {'', '.', '..'}
            or parts[1] != parts[1].rstrip(' .') or re.search(r'[\\:<>|?*"\x00-\x1f]', parts[1])):
        raise ValueError('素材路徑超出專案儲存區')
    return value


def _valid_digest(value):
    return isinstance(value, str) and _DIGEST.fullmatch(value) is not None


def _record_file(db, project_id, *, kind, relative_path, name, sha256, byte_count, metadata, created_at):
    if not _valid_digest(sha256) or type(byte_count) is not int or byte_count < 0:
        raise ValueError('素材雜湊或檔案大小無效')
    material_id = hash_bytes(f'{identifier(project_id)}:{kind}:{relative_path}'.encode()).hexdigest()[:32]
    db.execute(
        f'INSERT INTO project_material_files ({_FILE_COLUMNS}) VALUES (?,?,?,?,?,?,?,?) '
        'ON CONFLICT(relative_path) DO NOTHING',
        (material_id, kind, relative_path, name, sha256, byte_count, dump(metadata), created_at),
    )
    row = db.execute(
        'SELECT id,kind,sha256,byte_count FROM project_material_files WHERE relative_path=?',
        (relative_path,),
    ).fetchone()
    if tuple(row) != (material_id, kind, sha256, byte_count):
        raise ConflictError('素材路徑已有不同的原始檔案，請檢查專案資料')
    return material_id


def record_annotation(db, project_id, *, relative_path, sha256, byte_count, name, format,
                      image_id, annotation_revision, metadata=None):
    """Register an already-published annotation source in the image transaction."""
    relative_path = _owned_relative(logical_reference(relative_path), 'imported-annotations')
    if type(annotation_revision) is not int or annotation_revision < 1:
        raise ValueError('匯入標註的修訂版本無效')
    material_id = _record_file(
        db, project_id, kind='annotation', relative_path=relative_path, name=name, sha256=sha256,
        byte_count=byte_count, metadata={**(metadata or {}), 'format': format}, created_at=timestamp(),
    )
    source = {**metadata, 'name': name, 'format': format} if metadata else {}
    db.execute(
        'INSERT INTO project_annotation_links VALUES (?,?,?,?) ON CONFLICT DO NOTHING',
        (material_id, identifier(image_id), annotation_revision, dump(source)),
    )
    return material_id


def _sample_time(source):
    index = source.get('frame_index')
    seconds = source.get('timestamp_seconds')
    method = source.get('timestamp_method') or source.get('timestamp_source')
    if type(index) is not int or not 0 <= index <= 2**63 - 1:
        return None
    fps = source.get('fps')
    fps_seconds = index / fps if type(fps) in (int, float) and 0 < fps <= float_info.max else None
    if type(seconds) not in (int, float) or not 0 <= seconds <= float_info.max:
        seconds, method = fps_seconds, 'frame_fps'
    if seconds is None or not math.isfinite(seconds):
        return None
    if not isinstance(method, str) or not method:
        method = ('frame_fps' if fps_seconds is not None and math.isclose(seconds, fps_seconds, rel_tol=0, abs_tol=1e-9)
                  else 'legacy_unspecified')
    return index, seconds, method


def record_video_frame(db, image_id, source):
    """Save one derivation even if identical frame bytes already had an image ID."""
    video_id = identifier(source.get('video_id'))
    video = db.execute(
        "SELECT sha256 FROM project_material_files WHERE id=? AND kind='video'", (video_id,),
    ).fetchone()
    if video is None or source.get('video_sha256', video[0]) != video[0]:
        raise ConflictError('影片與素材清單中的原始檔案不一致')
    sample = _sample_time(source)
    if sample is None:
        raise ValueError('影片取樣需要有效的畫格編號及時間')
    db.execute(
        'INSERT INTO project_image_video_links VALUES (?,?,?,?,?) ON CONFLICT DO NOTHING',
        (identifier(image_id), video_id, *sample),
    )


def _ordinary_file(path, mode):
    info = path.lstat()
    if (not mode(info.st_mode) or stat.S_ISLNK(info.st_mode) or
            getattr(info, 'st_file_attributes', 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT):
        raise ValueError(f'素材必須位於專案的實際儲存區：{path.name}')
    return info


def _recording_relative(project_root, path):
    location = Path(path)
    if location.suffix.lower() not in VIDEO_SUFFIXES:
        raise ValueError('影片必須位於專案的錄影資料夾')
    if location.is_absolute():
        location = location.relative_to(project_root)
    return _owned_relative(logical_reference(location.as_posix()), 'recordings')


def _file_signature(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def _prepare_video(project_root, project_id, recording, verified_sha256=None):
    if recording.get('project_id', project_id) != project_id:
        raise ValueError('影片所屬專案無效')
    relative = _recording_relative(project_root, recording['path'])
    path = project_reference(project_root, relative)
    provided = project_root / recording['path']
    if provided.is_file() and provided.resolve() != path:
        raise ConflictError('影片在新舊儲存區有同名檔案，請先處理重複路徑')
    _ordinary_file(path.parent, stat.S_ISDIR)
    before = _ordinary_file(path, stat.S_ISREG)
    checksum = verified_sha256
    if checksum is None:
        with path.open('rb') as source:
            checksum = file_digest(source, 'sha256').hexdigest()
    signature = _file_signature(before)
    if _file_signature(_ordinary_file(path, stat.S_ISREG)) != signature:
        raise ConflictError('影片在登錄期間變更，請重新匯入')
    if not _valid_digest(checksum) or recording.get('sha256', checksum) != checksum:
        raise ConflictError('影片與來源紀錄的 SHA-256 不一致')
    metadata = {key: value for key, value in recording.items()
                if key not in {'id', 'video_id', 'path', 'relative_path', 'sha256', 'byte_count'}}
    name = recording.get('display_name')
    return signature, dict(
        kind='video', relative_path=relative, name=name if isinstance(name, str) and name else path.name,
        sha256=checksum, byte_count=before.st_size, metadata=metadata,
        created_at=datetime.fromtimestamp(before.st_mtime, timezone.utc).isoformat(),
    )


def _as_recording(project_root, row):
    material_id, _, relative, _, checksum, byte_count, raw, created_at = row
    try:
        path = project_reference(project_root, _owned_relative(relative, 'recordings'))
        _ordinary_file(path.parent, stat.S_ISDIR)
        info = _ordinary_file(path, stat.S_ISREG)
    except (OSError, ValueError):
        return None
    if info.st_size != byte_count:
        return None
    return {**json.loads(raw), 'id': material_id, 'video_id': material_id, 'path': str(path),
            'relative_path': path.relative_to(project_root).as_posix(), 'name': path.name, 'sha256': checksum,
            'byte_count': byte_count, 'created_at': created_at}


def _video_from_source(project_id, source, by_id, by_path, by_digest):
    own_source = True
    while isinstance(source, dict):
        own_source &= source.get('project_id', project_id) == project_id
        checksum = source.get('video_sha256')
        if not _valid_digest(checksum):
            checksum = None
        video_id = source.get('video_id')
        stored_path = source.get('video')
        if isinstance(stored_path, str):
            portable = PurePosixPath(stored_path.replace('\\', '/'))
            parents = portable.parent.parts
            base = None
            if parents[-2:] == ('inputs', 'videos'):
                base = parents[:-2]
            elif parents[-1:] == ('recordings',):
                base = parents[:-1]
            if base is not None:
                folder = base[-1] if base else ''
                if not base or checksum or folder == project_id or folder.endswith('__' + project_id[:8]):
                    stored_path = f'recordings/{portable.name}'
        for record in (by_id.get(video_id) if isinstance(video_id, str) else None,
                       by_path.get(stored_path) if isinstance(stored_path, str) else None):
            if record is not None and (record[2] == checksum if checksum else own_source):
                return record, source
        matches = by_digest.get(checksum, ())
        if len(matches) == 1:
            return matches[0], source
        source = source.get('original')
    return None


def _video_index(db):
    by_id, by_path, by_digest = {}, {}, defaultdict(list)
    for row in db.execute("SELECT id, relative_path, sha256 FROM project_material_files WHERE kind='video'"):
        material_id, relative_path, checksum = row
        by_id[material_id] = row
        by_path[relative_path] = row
        by_digest[checksum].append(row)
    return by_id, by_path, by_digest


def _unlinked_video_sources(db):
    candidates = db.execute(
        "SELECT id,metadata FROM project_materials WHERE kind='image' "
        "AND metadata LIKE '%\"video%' AND NOT EXISTS "
        "(SELECT 1 FROM project_image_video_links WHERE image_id=project_materials.id)"
    )
    for image_id, encoded in candidates:
        try:
            metadata = json.loads(encoded)
        except (TypeError, ValueError):
            continue
        yield image_id, metadata


def reconcile_recordings(db, project_root, project_id):
    """Import durable legacy sidecars and attach source links without editing images."""
    project_root = Path(project_root).resolve()
    imported = rejected = linked = 0
    try:
        folder = project_area(project_root, 'recordings')
    except (OSError, ValueError):
        return {'registered': 0, 'rejected': 1, 'linked': 0}
    folders = [folder]
    legacy = project_root / 'recordings'
    if legacy != folder and (legacy.exists() or legacy.is_symlink()):
        folders.append(legacy)
    _, known, _ = _video_index(db)
    known_stems = {PurePosixPath(path).stem for path in known}
    for directory in folders:
        if not directory.exists() and not directory.is_symlink():
            continue
        try:
            _ordinary_file(directory, stat.S_ISDIR)
        except (OSError, ValueError):
            rejected += 1
            continue
        for sidecar in sorted(directory.glob('recording_*.json')):
            if sidecar.stem in known_stems:
                continue
            try:
                _ordinary_file(sidecar, stat.S_ISREG)
                metadata = json.loads(sidecar.read_text(encoding='utf-8'))
                if not isinstance(metadata, dict) or metadata.get('project_id') != project_id:
                    raise ValueError('影片紀錄所屬專案無效')
                extension = metadata.get('extension', '.mp4')
                if not isinstance(extension, str) or extension not in VIDEO_SUFFIXES:
                    raise ValueError('影片格式無效')
                location = sidecar.with_suffix(extension)
                _, prepared = _prepare_video(project_root, project_id, {**metadata, 'path': str(location)})
                _record_file(db, project_id, **prepared)
            except (OSError, ValueError):
                rejected += 1
                continue
            known_stems.add(sidecar.stem)
            imported += 1
    indexes = _video_index(db)
    if indexes[0]:
        for image_id, metadata in _unlinked_video_sources(db):
            result = _video_from_source(project_id, metadata, *indexes)
            if result is None:
                continue
            (video_id, _, checksum), source = result
            if _sample_time(source) is None:
                continue
            record_video_frame(db, image_id, {**source, 'video_id': video_id, 'video_sha256': checksum})
            linked += 1
    return {'registered': imported, 'rejected': rejected, 'linked': linked}


class MaterialCatalog:
    def __init__(self, store):
        self.store = store

    def register_video(self, project_id, recording, *, verified_sha256=None):
        root = self.store.directory(project_id).resolve()
        signature, data = _prepare_video(root, project_id, recording, verified_sha256=verified_sha256)
        with self.store.connection(project_id, write=True) as db:
            path = project_reference(root, data['relative_path'])
            if _file_signature(_ordinary_file(path, stat.S_ISREG)) != signature:
                raise ConflictError('影片在登錄期間變更，請重新匯入')
            material_id = _record_file(db, project_id, **data)
            row = db.execute(f'SELECT {_FILE_COLUMNS} FROM project_material_files WHERE id=?', (material_id,)).fetchone()
        recording = _as_recording(root, row)
        if recording is None:
            raise ConflictError('登錄後找不到原始影片，請檢查專案的錄影資料夾')
        return recording

    def find_recording(self, project_id, *, path=None, sha256=None, video_id=None):
        root = self.store.directory(project_id).resolve()
        terms = ["kind='video'"]
        values = []
        if path is not None:
            try:
                relative = _recording_relative(root, path)
            except (OSError, ValueError):
                return None
            terms.append('relative_path=?')
            values.append(relative)
        if sha256 is not None:
            terms.append('sha256=?')
            values.append(sha256)
        if video_id is not None:
            terms.append('id=?')
            values.append(identifier(video_id))
        if not values:
            raise ValueError('請指定影片路徑、雜湊或素材識別碼')
        where = ' AND '.join(terms)
        with self.store.connection(project_id) as db:
            for row in db.execute(
                f'SELECT {_FILE_COLUMNS} FROM project_material_files WHERE {where} ORDER BY created_at,id', values,
            ):
                if recording := _as_recording(root, row):
                    return recording
        return None

    def list_recordings(self, project_id):
        root = self.store.directory(project_id).resolve()
        with self.store.connection(project_id) as db:
            rows = db.execute(
                f"SELECT {_FILE_COLUMNS} FROM project_material_files WHERE kind='video' ORDER BY created_at DESC,id",
            )
            return [recording for row in rows if (recording := _as_recording(root, row))]

    def reconcile_recordings(self, project_id):
        root = self.store.directory(project_id)
        with self.store.connection(project_id, write=True) as db:
            return reconcile_recordings(db, root, project_id)

    def recover_shared_recordings(self, project_id, directory):
        """Copy old shared videos only when historical project frames prove their identity."""
        directory = Path(directory)
        try:
            _ordinary_file(directory.parent, stat.S_ISDIR)
            _ordinary_file(directory, stat.S_ISDIR)
        except FileNotFoundError:
            return {'recovered': 0, 'rejected': 0}
        except (OSError, ValueError):
            return {'recovered': 0, 'rejected': 1}

        candidates = defaultdict(set)
        with self.store.connection(project_id) as db:
            for _, source in _unlinked_video_sources(db):
                while isinstance(source, dict):
                    path, checksum = source.get('video'), source.get('video_sha256')
                    if source.get('kind') == 'video' and isinstance(path, str) and _valid_digest(checksum):
                        video = Path(path)
                        if (video.is_absolute() and video.parent == directory
                                and video.suffix.lower() in VIDEO_SUFFIXES and _sample_time(source) is not None):
                            candidates[video].add(checksum)
                    source = source.get('original')
        if not candidates:
            return {'recovered': 0, 'rejected': 0}

        try:
            from .acquisition import AcquisitionError, import_project_video
        except ImportError:
            return {'recovered': 0, 'rejected': len(candidates)}
        recovered = rejected = 0
        for video, checksums in candidates.items():
            if len(checksums) != 1:
                rejected += 1
                continue
            checksum, = checksums
            if self.find_recording(project_id, sha256=checksum):
                continue
            try:
                _ordinary_file(video, stat.S_ISREG)
                folder = project_area(self.store.directory(project_id), 'recordings')
                import_project_video(str(video), folder, project_id, lambda *_: None,
                                     catalog=self, expected_sha256=checksum)
            except (AcquisitionError, OSError, ValueError):
                rejected += 1
                continue
            recovered += 1
        return {'recovered': recovered, 'rejected': rejected}

    def list_materials(self, project_id, kind=None):
        if kind is not None and (not isinstance(kind, str) or kind not in MATERIAL_KINDS):
            raise ValueError('素材類型無效')
        root = self.store.directory(project_id).resolve()
        with self.store.connection(project_id) as db:
            return _material_rows(db, root, kind)


def _material_rows(db, root, kind):
    video_links, annotation_links = defaultdict(list), defaultdict(list)
    if kind in (None, 'image'):
        for row in db.execute(
            'SELECT image_id, video_id, relative_path, sha256, frame_index, timestamp_seconds, timestamp_method '
            'FROM project_image_video_links JOIN project_material_files ON id=video_id '
            'ORDER BY image_id,video_id,frame_index'
        ):
            link = dict(row)
            link['relative_path'] = project_reference(root, link['relative_path']).relative_to(root).as_posix()
            video_links[link.pop('image_id')].append(link)
    if kind in (None, 'annotation'):
        for row in db.execute('SELECT * FROM project_annotation_links ORDER BY image_id,annotation_revision'):
            link = dict(row)
            source = json.loads(link.pop('source_metadata'))
            if source:
                link['source'] = source
            annotation_links[link.pop('annotation_id')].append(link)
    where = ' WHERE kind = ?' if kind else ''
    rows = db.execute(
        'SELECT * FROM project_materials' + where + ' ORDER BY created_at DESC,id', (kind,) if kind else (),
    )
    materials = []
    for row in rows:
        material = dict(row)
        material['relative_path'] = project_reference(root, material['relative_path']).relative_to(root).as_posix()
        metadata = json.loads(material['metadata'])
        fields = {key: material.pop(key) for key in ('width', 'height', 'batch_id', 'review_state')}
        material_id = material['id']
        if material['kind'] == 'image':
            metadata = {**fields, 'source': metadata, 'video_sources': video_links[material_id]}
        if material['kind'] == 'annotation':
            metadata['linked_images'] = annotation_links[material_id]
        material['metadata'] = metadata
        materials.append(material)
    return materials
