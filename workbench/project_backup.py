"""Portable, checksummed copies of a project's database and owned files."""
from __future__ import annotations

from contextlib import closing
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import stat
from tempfile import TemporaryDirectory
import time
import unicodedata
from uuid import uuid4
import zipfile

from .file_lock import exclusive_file_lock
from .migrations import migrate
from .project_layout import AREA_PATHS, logical_reference, migrate_project_layout, project_area, project_relative
from .store import ConflictError, clean_class_names, dump, identifier, timestamp


FORMAT = 'vision-workbench-project-backup'
MANIFEST = 'backup-manifest.json'
DATABASE = 'project.sqlite3'
ARCHIVE_SUFFIX = '.vwb-project.zip'
ACTIVE_RUN_STATES = {'queued', 'preparing', 'running', 'stopping'}
FIRST_BACKUP_DATABASE_VERSION = 6
MATERIAL_CATALOG_VERSION = 7
CHUNK_SIZE = 1024 * 1024
MANIFEST_LIMIT = 32 * CHUNK_SIZE
INVALID_NAME = re.compile(r'[<>:"\\|?*\x00-\x1f]')
DEVICE_NAME = re.compile(r'(?i)^(CON|PRN|AUX|NUL|COM[1-9¹²³]|LPT[1-9¹²³])$')
SHA256 = re.compile(r'[0-9a-f]{64}')


def _relative(value):
    if not isinstance(value, str) or not value:
        raise ValueError('備份檔案路徑無效')
    for piece in value.split('/'):
        if (not piece or piece in {'.', '..'} or piece != piece.rstrip(' .')
                or INVALID_NAME.search(piece) or DEVICE_NAME.fullmatch(piece.split('.', 1)[0])):
            raise ValueError(f'備份含有無法跨主機使用的路徑：{value}')
    return value


def _basename(value):
    if '/' in _relative(value):
        raise ValueError(f'備份檔名無效：{value}')
    return value


def _assert_unique_paths(files, directories):
    paths = {}
    for full_path in (*files, *directories):
        parts = _relative(full_path).split('/')
        for end in range(1, len(parts) + 1):
            path = '/'.join(parts[:end])
            key = unicodedata.normalize('NFC', path).casefold()
            if paths.setdefault(key, path) != path:
                raise ValueError(f'備份含有大小寫或名稱表示重複的路徑：{path}')
            if path in files and (end < len(parts) or path in directories):
                raise ValueError(f'備份路徑同時指向檔案及資料夾：{path}')


def _unstable(path):
    parts = path.split('/')
    name = parts[-1]
    scratch = AREA_PATHS['scratch']
    return (parts[0] == 'scratch' or path == scratch or path.startswith(scratch + '/')
            or any(piece.startswith('.') for piece in parts)
            or name.endswith(('.tmp', '.partial', '.lock'))
            or (len(parts) == 1 and (name in {f'{DATABASE}-wal', f'{DATABASE}-shm', f'{DATABASE}-journal'}
                                    or '.before-migration-' in name)))


def _plain_stat(path):
    value = path.lstat()
    if (stat.S_ISLNK(value.st_mode)
            or getattr(value, 'st_file_attributes', 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT):
        raise ValueError(f'備份不支援連結或重新剖析路徑：{path.name}')
    return value


def _signature(value):
    return value.st_size, value.st_mtime_ns, value.st_ino, value.st_dev


def _collect(root):
    files, directories = {}, set()

    def visit(folder):
        for item in folder.iterdir():
            relative = item.relative_to(root).as_posix()
            if relative == DATABASE or _unstable(relative):
                continue
            value = _plain_stat(item)
            if stat.S_ISDIR(value.st_mode):
                directories.add(relative)
                visit(item)
            elif stat.S_ISREG(value.st_mode):
                files[relative] = _signature(value)
            else:
                raise ValueError(f'備份含有非一般檔案：{relative}')

    visit(root)
    _assert_unique_paths({**files, DATABASE: None}, directories)
    return files, directories


def _read_only(path):
    return sqlite3.connect(f'{path.resolve().as_uri()}?mode=ro', uri=True)


def _schema(db):
    return [(kind, name, table, ' '.join(sql.split()))
            for kind, name, table, sql in db.execute(
                'SELECT type, name, tbl_name, sql FROM sqlite_schema '
                'WHERE sql IS NOT NULL ORDER BY type, name')]


def _project_row(db):
    rows = db.execute('SELECT id, name, revision, classes FROM project').fetchall()
    if len(rows) != 1:
        raise ValueError('備份資料庫必須只有一個專案')
    pid, name, revision, class_text = rows[0]
    classes = json.loads(class_text)
    if (not isinstance(name, str) or not name.strip() or len(name) > 100
            or type(revision) is not int or revision < 1
            or not isinstance(classes, list) or clean_class_names(classes) != classes):
        raise ValueError('備份專案名稱、修訂或類別順序無效')
    return dict(project_id=identifier(pid), project_name=name, project_revision=revision, classes=classes)


def _require_file(files, path, digest=None):
    item = files.get(_relative(path))
    if item is None:
        raise ValueError(f'備份缺少專案檔案：{path}')
    if digest is not None and item['sha256'] != digest:
        raise ValueError(f'專案檔案的 SHA-256 與原紀錄不符：{path}')
    return item


def _require_project_file(root, files, path, digest=None):
    return _require_file(files, project_relative(root, _relative(path)).as_posix(), digest)


def _required_digest(value):
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        raise ValueError('專案檔案的原始 SHA-256 紀錄無效')
    return value


def _validate_database(root, path, identity, files):
    with closing(_read_only(path)) as db:
        db.execute('PRAGMA trusted_schema=OFF')
        versions = db.execute('SELECT version FROM schema_migrations ORDER BY version').fetchall()
        version = versions[-1][0] if versions else None
        if type(version) is not int or version < FIRST_BACKUP_DATABASE_VERSION:
            raise ValueError('備份資料庫結構與此版本不相容，請使用相同版本或相容版本的工作台')
        with closing(sqlite3.connect(':memory:')) as template:
            migrate(template, through=version)
            if (_schema(db) != _schema(template)
                    or versions != template.execute('SELECT version FROM schema_migrations ORDER BY version').fetchall()):
                raise ValueError('備份資料庫結構與此版本不相容，請使用相同版本或相容版本的工作台')
        if db.execute('PRAGMA integrity_check').fetchone() != ('ok',) or db.execute('PRAGMA foreign_key_check').fetchone():
            raise ValueError('備份資料庫完整性或關聯檢查失敗')
        if any(identity.get(key) != value for key, value in _project_row(db).items()):
            raise ValueError('備份清單與資料庫的專案內容不一致')
        for image, digest in db.execute('SELECT image_file, sha256 FROM assets'):
            _require_project_file(root, files, f'images/{_basename(image)}', _required_digest(digest))
        asset_columns = {row[1] for row in db.execute('PRAGMA table_info(assets)')}
        for (record,) in db.execute('SELECT data FROM review_trash'):
            asset = json.loads(record)
            if not isinstance(asset, dict) or set(asset) != asset_columns:
                raise ValueError('備份的垃圾桶紀錄格式無效')
            _require_project_file(root, files, f"images/{_basename(asset['image_file'])}", _required_digest(asset['sha256']))
        for table in ('dataset_versions', 'training_runs', 'model_versions', 'model_exports', 'prediction_candidates'):
            for (relative,) in db.execute(f'SELECT relative_path FROM {table}'):
                _require_project_file(root, files, relative)
        if version >= MATERIAL_CATALOG_VERSION:
            _validate_material_registry(root, db, files)
    return version


def _validate_material_registry(root, db, files):
    areas = {'video': 'recordings', 'annotation': 'imported-annotations'}
    for kind, relative, digest, byte_count in db.execute(
        'SELECT kind, relative_path, sha256, byte_count FROM project_material_files'
    ):
        actual = project_relative(root, _relative(relative))
        area = areas.get(kind)
        allowed = (Path(area), project_area(root, area).relative_to(root)) if area else ()
        if actual.parent not in allowed:
            raise ValueError(f'素材類型與專案路徑不符：{relative}')
        registered = _require_file(files, actual.as_posix(), _required_digest(digest))
        if registered['bytes'] != byte_count:
            raise ValueError(f'素材清單的檔案大小與備份不符：{relative}')
    invalid_link = db.execute("""
        WITH material_links(image_id,material_id,expected_kind) AS (
            SELECT image_id,video_id,'video' FROM project_image_video_links
            UNION ALL
            SELECT image_id,annotation_id,'annotation' FROM project_annotation_links
        )
        SELECT 1 FROM material_links AS link
        JOIN project_material_files AS material ON material.id=link.material_id
        WHERE material.kind<>link.expected_kind OR (
            NOT EXISTS (SELECT 1 FROM assets WHERE id=link.image_id)
            AND NOT EXISTS (SELECT 1 FROM review_trash WHERE id=link.image_id)
        )
        LIMIT 1
    """).fetchone()
    if invalid_link:
        raise ValueError('備份素材關聯指向不符類型的檔案或不存在的影像')


def _upgrade_restored_materials(root, manifest, files):
    from .materials import reconcile_recordings

    database = root / DATABASE
    with closing(sqlite3.connect(database)) as db:
        migrate(db, backup_existing=False)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA foreign_keys=ON')
        with db:
            reconcile_recordings(db, root, manifest['project_id'])
    _validate_database(root, database, manifest, files)


def _json_record(path):
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError(f'專案紀錄格式無效：{path.name}')
    return value


def _validate_file_records(root, identity, files, progress):
    """Check important references without starting a model or media decoder."""
    for name in files:
        parts = logical_reference(name).split('/')
        parent = Path(name).parent.as_posix()
        if len(parts) == 3 and re.fullmatch(r'D[0-9]+', parts[1]) and parts[0] == 'datasets' and parts[2] == 'manifest.json':
            progress('驗證固定資料版本及圖片')
            data = _json_record(root / name)
            digest = data.pop('manifest_sha256', None)
            if digest != sha256(dump(data).encode('utf-8')).hexdigest() or data.get('dataset_version_id') != parts[1]:
                raise ValueError(f'資料版本清單的內容或 SHA-256 不符：{parts[1]}')
            for asset in data['assets']:
                _require_file(files, f"{parent}/{_relative(asset['image_file'])}", _required_digest(asset['sha256']))
        elif len(parts) == 3 and parts[0] == 'runs' and re.fullmatch(r'R[0-9]+', parts[1]) and parts[2] == 'run.json':
            from .run_events import TERMINAL, read_state
            data = read_state(root / name)
            terminal = f'{parent}/control/terminal.json'
            if data.get('status') in ACTIVE_RUN_STATES and terminal in files:
                override = _json_record(root / terminal)
                if override.get('status') in TERMINAL:
                    data.update(override)
            if data.get('status') in ACTIVE_RUN_STATES:
                raise ConflictError('專案有尚未結束的訓練，請完成後再備份')
            _require_project_file(root, files, f"datasets/{_basename(data['dataset_version_id'])}/manifest.json")
        elif len(parts) == 3 and parts[0] == 'models' and parts[2] == 'model.json':
            data = _json_record(root / name)
            if checkpoint := data.get('checkpoint'):
                source = data.get('source') or {}
                _require_file(files, f'{parent}/{_basename(checkpoint)}', source.get('sha256'))
        elif len(parts) == 3 and parts[0] == 'model-exports' and parts[2] == 'record.json':
            data = _json_record(root / name)
            archive = _basename(str(data['path']).replace('\\', '/').rsplit('/', 1)[-1])
            _require_file(files, f'{parent}/{archive}', _required_digest(data['sha256']))
        elif len(parts) == 2 and parts[0] == 'recordings' and parts[1].startswith('recording_') and parts[1].endswith('.json'):
            data = _json_record(root / name)
            extension = data.get('extension', '.mp4')
            if (data.get('project_id') != identity['project_id'] or not isinstance(extension, str)
                    or not re.fullmatch(r'\.[a-z0-9]{1,8}', extension)):
                raise ValueError(f'錄影來源或格式無效：{name}')
            _require_file(files, name[:-5] + extension, data.get('sha256'))


def _stream_contents(source, destination, size, checkpoint):
    digest, copied = sha256(), 0
    while chunk := source.read(CHUNK_SIZE):
        copied += len(chunk)
        if copied > size:
            raise ValueError('讀到的檔案內容大於備份紀錄')
        destination.write(chunk)
        digest.update(chunk)
        checkpoint(copied)
    if copied != size:
        raise ValueError('讀到的檔案大小與備份紀錄不符')
    return {'bytes': copied, 'sha256': digest.hexdigest()}


def _copy_with_progress(source, dest, size, progress, message, start, total):
    last_update = time.monotonic()

    def checkpoint(copied):
        nonlocal last_update
        now = time.monotonic()
        if now - last_update >= .2 or copied == size:
            progress(message, 5 + 85 * (start + copied) / max(total, 1))
            last_update = now

    return _stream_contents(source, dest, size, checkpoint)


def create_project_backup(store, project_id, output_dir, progress=lambda *_: None):
    root = store.directory(project_id)
    database = root / DATABASE
    if not database.is_file():
        raise FileNotFoundError('找不到要備份的專案')
    _plain_stat(database)
    output = Path(output_dir).resolve()
    if output.is_relative_to(store.root):
        raise ValueError('完整專案備份需儲存在專案目錄以外')
    output.mkdir(parents=True, exist_ok=True)
    target = output / f'project-{project_id[:8]}-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{uuid4().hex[:8]}{ARCHIVE_SUFFIX}'

    with (exclusive_file_lock(root / '.training-admission.lock'),
          exclusive_file_lock(root / '.dataset-publish.lock'),
          exclusive_file_lock(root / '.images.lock'),
          TemporaryDirectory(prefix='.backup-', dir=output) as staging,
          closing(_read_only(database)) as source):
        progress('建立包含已提交修訂的資料庫快照', 0)
        data_version = source.execute('PRAGMA data_version').fetchone()[0]
        snapshot = Path(staging) / DATABASE
        with closing(sqlite3.connect(snapshot)) as db:
            source.backup(db, pages=1024, progress=lambda *_: progress('建立包含已提交修訂的資料庫快照'))
            db.execute('PRAGMA journal_mode=DELETE')
            identity = _project_row(db)
        if identity['project_id'] != project_id:
            raise ConflictError('備份來源的專案識別碼已變更')
        signatures, directories = _collect(root)
        snapshot_size = snapshot.stat().st_size
        total = snapshot_size + sum(value[0] for value in signatures.values())
        archive = Path(staging) / target.name
        entries = {}
        copied = 0
        with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as bundle:
            inputs = [(DATABASE, snapshot, snapshot_size)]
            inputs.extend((name, root / name, signature[0]) for name, signature in sorted(signatures.items()))
            for name, path, size in inputs:
                with path.open('rb') as file, bundle.open('project/' + name, 'w', force_zip64=True) as dest:
                    if name in signatures and _signature(os.fstat(file.fileno())) != signatures[name]:
                        raise ConflictError(f'檔案在備份期間已變更：{name}')
                    entries[name] = _copy_with_progress(file, dest, size, progress, '備份專案檔案', copied, total)
                copied += size
            _validate_database(root, snapshot, identity, entries)
            _validate_file_records(root, identity, entries, progress)
            if (data_version != source.execute('PRAGMA data_version').fetchone()[0]
                    or _collect(root) != (signatures, directories)):
                raise ConflictError('備份期間專案有變動，請完成工作後再試')
            manifest = {**identity, 'format': FORMAT, 'schema_version': 1, 'created_at': timestamp(),
                        'project_folder': root.name, 'files': entries, 'directories': sorted(directories)}
            encoded = dump(manifest).encode('utf-8')
            if len(encoded) > MANIFEST_LIMIT:
                raise ValueError('專案的完整備份索引過大')
            bundle.writestr(MANIFEST, encoded)
        with archive.open('r+b') as file:
            os.fsync(file.fileno())
        progress('發布已驗證的專案備份', 98)
        if target.exists():
            raise ConflictError('備份檔案已存在，請再試一次')
        archive.rename(target)
    return {**identity, 'path': str(target), 'files': len(entries), 'bytes': target.stat().st_size}


def _archive_index(bundle):
    members = bundle.infolist()
    by_name = {item.filename: item for item in members}
    manifest_item = by_name.get(MANIFEST)
    if (len(by_name) != len(members) or manifest_item is None or manifest_item.file_size > MANIFEST_LIMIT
            or any(item.is_dir() or item.flag_bits & 1 or stat.S_IFMT(item.external_attr >> 16) not in {0, stat.S_IFREG}
                   or item.filename != item.orig_filename for item in members)):
        raise ValueError('專案備份封裝索引無效')
    manifest = json.loads(bundle.read(manifest_item))
    if not isinstance(manifest, dict) or manifest.get('format') != FORMAT or manifest.get('schema_version') != 1:
        raise ValueError('這不是目前支援的 Vision Workbench 完整專案備份')
    files, directories = manifest.get('files'), manifest.get('directories')
    if (not isinstance(files, dict) or DATABASE not in files or not isinstance(directories, list)
            or any(not isinstance(path, str) for path in directories) or len(set(directories)) != len(directories)
            or set(by_name) != {MANIFEST, *('project/' + name for name in files)}):
        raise ValueError('專案備份的檔案清單與 ZIP 不一致')
    directories = set(directories)
    _assert_unique_paths(files, directories)
    if any(_unstable(path) for path in (*files, *directories)):
        raise ValueError('備份含有不支援的暫存路徑')
    for name, entry in files.items():
        if (not isinstance(entry, dict) or type(entry.get('bytes')) is not int
                or entry['bytes'] != by_name['project/' + name].file_size
                or not SHA256.fullmatch(str(entry.get('sha256', '')))):
            raise ValueError(f'備份檔案的大小或 SHA-256 紀錄無效：{name}')
    return manifest, files, directories


def _portable_export_path(root, value, project_folder, paths):
    if not isinstance(value, str) or not value:
        return None
    parts = value.replace('\\', '/').split('/')
    starts = [0, *(index + 1 for index, part in enumerate(parts[:-1])
                   if part.casefold() == project_folder.casefold())]
    for start in reversed(starts):
        try:
            relative = logical_reference(_relative('/'.join(parts[start:])))
            if relative != 'exports' and not relative.startswith('exports/'):
                continue
            if project_relative(root, relative).as_posix() in paths:
                return relative
        except ValueError:
            continue
    return None


def _restore_local_metadata(root, manifest, files, directories):
    source_folder = _basename(manifest['project_folder'])
    paths = set(files) | directories
    with closing(sqlite3.connect(root / DATABASE)) as db:
        db.execute('PRAGMA trusted_schema=OFF')
        for export_id, raw in db.execute('SELECT id, data FROM exports').fetchall():
            record = json.loads(raw)
            for key in ('path', 'zip_path'):
                value = record.get(key)
                if not value:
                    continue
                relative = _portable_export_path(root, value, source_folder, paths)
                if relative:
                    record[key] = relative
                else:
                    record.setdefault('original_paths', {})[key] = record.pop(key)
            if dump(record) != raw:
                db.execute('UPDATE exports SET data=? WHERE id=?', (dump(record), export_id))
        db.commit()
        db.execute('PRAGMA journal_mode=WAL')

    for name in files:
        parts = logical_reference(name).split('/')
        if len(parts) != 3 or parts[0] != 'model-exports' or parts[2] != 'record.json':
            continue
        record_path = root / name
        record = _json_record(record_path)
        archive = _basename(str(record['path']).replace('\\', '/').rsplit('/', 1)[-1])
        record['path'] = archive
        record.pop('directory', None)
        record_path.write_text(dump(record), encoding='utf-8')

    # CVAT task IDs belong to a server, and may refer to unrelated jobs there.
    cvat = root / 'integrations' / 'cvat'
    old_links = [path for path in (cvat / 'project-link.json', cvat / 'sync.json') if path.is_file()]
    if old_links:
        provenance = cvat / 'source-host' / uuid4().hex
        provenance.mkdir(parents=True)
        for link in old_links:
            link.rename(provenance / link.name)


def _relocated_index(root, files, directories):
    def location(name):
        return name if (root / name).exists() else project_relative(root, name).as_posix()

    relocated = {}
    for name, entry in files.items():
        actual = location(name)
        if actual in relocated or _plain_stat(root / actual).st_size != entry['bytes']:
            raise ValueError(f'專案資料架構遷移後的檔案與備份不符：{actual}')
        relocated[actual] = entry
    relocated_dirs = {location(name) for name in directories}
    _assert_unique_paths(relocated, relocated_dirs)
    return relocated, relocated_dirs


def _assert_available(store, pid, target):
    conflicts = {pid.casefold(), target.name.casefold()}
    if any(item.name.casefold() in conflicts or item.name.casefold().endswith('__' + pid[:8])
           for item in store.root.iterdir()):
        raise ConflictError('目的主機已存在相同專案識別碼或資料夾，還原未覆寫既有專案')


def restore_project_backup(store, archive_path, progress=lambda *_: None):
    path = Path(archive_path)
    if not path.is_file():
        raise FileNotFoundError('找不到完整專案備份檔')
    try:
        with zipfile.ZipFile(path) as bundle:
            manifest, files, directories = _archive_index(bundle)
            pid = identifier(manifest.get('project_id'))
            name = store._name(manifest.get('project_name'))
            target = store.root / store._folder_name(name, pid)
            _assert_available(store, pid, target)
            total = sum(entry['bytes'] for entry in files.values())
            if total > shutil.disk_usage(store.root).free:
                raise ValueError('目的主機磁碟空間不足，無法解開完整專案備份')
            # The project scanner sees no database until the final rename.
            with TemporaryDirectory(prefix='.project-restore-', dir=store.root) as staging:
                root = Path(staging) / 'project'
                root.mkdir()
                for directory in sorted(directories):
                    (root / directory).mkdir(parents=True, exist_ok=True)
                copied = 0
                progress('檢查備份清單並準備還原', 0)
                for filename, entry in files.items():
                    destination = root / filename
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with bundle.open('project/' + filename) as source, destination.open('xb') as output:
                        stored = _copy_with_progress(source, output, entry['bytes'], progress,
                                                     '還原並驗證專案檔案', copied, total)
                    if stored != entry:
                        raise ValueError(f'備份檔案的 SHA-256 驗證失敗：{filename}')
                    copied += entry['bytes']
                version = _validate_database(root, root / DATABASE, manifest, files)
                _validate_file_records(root, manifest, files, progress)
                layout_changes = {'moved': [], 'conflicts': []}
                migrate_project_layout(root, layout_changes)
                files, directories = _relocated_index(root, files, directories)
                if version < MATERIAL_CATALOG_VERSION:
                    progress('升級舊版素材清單', 95)
                    _upgrade_restored_materials(root, manifest, files)
                else:
                    _validate_database(root, root / DATABASE, manifest, files)
                _restore_local_metadata(root, manifest, files, directories)
                project_area(root, 'images').mkdir(parents=True, exist_ok=True)
                with exclusive_file_lock(store.root / '.project-restore.lock'):
                    _assert_available(store, pid, target)
                    progress('發布已完整驗證的專案', 98)
                    root.rename(target)
    except (zipfile.BadZipFile, KeyError, sqlite3.DatabaseError, TypeError, UnicodeError) as error:
        raise ValueError('完整專案備份格式損壞或資料紀錄無效') from error
    return {'project_id': pid, 'project_name': name, 'project_revision': manifest['project_revision'],
            'classes': manifest['classes'], 'files': len(files), 'bytes': total}
