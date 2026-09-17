"""Ordered, transactional migrations executed when a store opens, not on reads."""
from pathlib import Path
import hashlib
import json
import sqlite3
from contextlib import closing
from .file_lock import exclusive_file_lock


def migrate(db):
    database = db.execute('PRAGMA database_list').fetchone()[2]
    if database:
        with exclusive_file_lock(Path(database).with_suffix('.migration.lock')):
            return _migrate(db)
    return _migrate(db)


def _migrate(db):
    paths = sorted(Path(__file__).with_name('migrations').glob('*.sql'))
    # SQLite's backup API includes committed WAL content; copying the file does not.
    installed = set()
    if db.execute("SELECT 1 FROM sqlite_master WHERE name='schema_migrations'").fetchone():
        installed = {row[0] for row in db.execute('SELECT version FROM schema_migrations')}
    pending = [path for path in paths if int(path.name.split('_')[0]) not in installed]
    database = db.execute('PRAGMA database_list').fetchone()[2]
    if pending and database and db.execute("SELECT 1 FROM sqlite_master WHERE name='project'").fetchone():
        backup = Path(database).with_suffix(f'.before-migration-{pending[-1].name.split("_")[0]}.backup')
        if not backup.exists():
            temporary = backup.with_suffix('.partial')
            with closing(sqlite3.connect(temporary)) as target:
                db.backup(target)
            temporary.replace(backup)
    db.execute('CREATE TABLE IF NOT EXISTS schema_migrations(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)')
    db.commit()
    for path in paths:
        version = int(path.name.split('_')[0])
        db.execute('BEGIN IMMEDIATE')
        try:
            if db.execute('SELECT 1 FROM schema_migrations WHERE version=?', (version,)).fetchone():
                db.commit()
                continue
            prior = None
            if version == 2:
                from .independence import confirmation, fingerprint, valid, preserve
                candidate = confirmation(db)
                if valid(candidate, fingerprint(db)):
                    prior = candidate
            statement = ''
            for line in path.read_text(encoding='utf-8').splitlines(True):
                statement += line
                if sqlite3.complete_statement(statement):
                    db.execute(statement)
                    statement = ''
            if statement.strip():
                raise ValueError(f'Incomplete migration: {path.name}')
            if version == 2 and prior is not None:
                preserve(db, prior, fingerprint(db), migration='0002')
            if version == 1:
                for hid, raw in db.execute('SELECT id,data FROM history').fetchall():
                    data = json.loads(raw)
                    if 'shapes' in data:
                        shapes = json.dumps(data.pop('shapes'), ensure_ascii=False, separators=(',', ':'))
                        key = hashlib.sha256(shapes.encode()).hexdigest()
                        db.execute('INSERT OR IGNORE INTO annotation_blobs VALUES(?,?)', (key, shapes))
                        data['annotation_hash'] = key
                        db.execute('UPDATE history SET data=? WHERE id=?', (json.dumps(data, ensure_ascii=False), hid))
            db.execute('INSERT INTO schema_migrations(version) VALUES(?)', (version,))
            db.commit()
        except BaseException:
            db.rollback()
            raise
