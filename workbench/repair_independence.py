"""Offline, dry-run-first repair using an original pre-0002 backup as evidence.

Usage: python -m workbench.repair_independence --database PATH --backup PATH
Add --apply --offline only after closing every writer. Never starts ProjectStore.
"""
import argparse
from contextlib import closing
import json
from pathlib import Path
import shutil
import sqlite3
import tempfile
import uuid
from .file_lock import exclusive_file_lock
from .independence import repair


def copy_database(source, destination):
    """Copy a quiescent DB and WAL, never open the original or copy its SHM."""
    paths = [Path(source), Path(str(source) + '-wal')]
    def signature():
        return [(p.stat().st_size, p.stat().st_mtime_ns) if p.exists() else None for p in paths]
    before = signature()
    for index, path in enumerate(paths):
        if path.exists():
            shutil.copy2(path, destination if index == 0 else Path(str(destination) + '-wal'))
    if before != signature():
        raise ValueError('資料庫正在變動，請關閉所有寫入程序後重試')


def repair_files(database, backup, *, apply=False, offline=False):
    database, backup = Path(database), Path(backup)
    if database.is_symlink() or backup.is_symlink():
        raise ValueError('資料庫與備份不可為符號連結')
    database, backup = database.resolve(), backup.resolve()
    if database == backup:
        raise ValueError('目前資料庫與備份必須是不同的普通檔案')
    if apply and not offline:
        raise ValueError('套用修復前必須關閉所有寫入程序，並明確指定 --offline')
    lock_path = database.parents[2] / 'desktop.lock' if len(database.parents) > 2 else None
    if lock_path and lock_path.exists():
        from .process_identity import alive
        pid = lock_path.read_text(encoding='utf-8').splitlines()[0]
        if alive(pid) is not False:
            raise ValueError('桌面程式仍在執行，請先關閉；未開啟任何資料庫')
    with tempfile.TemporaryDirectory(prefix='independence-repair-') as temporary:
        current_copy, backup_copy = Path(temporary) / 'current.db', Path(temporary) / 'backup.db'
        copy_database(database, current_copy)
        copy_database(backup, backup_copy)
        with closing(sqlite3.connect(current_copy)) as current, closing(sqlite3.connect(backup_copy)) as evidence:
            for db in (current, evidence):
                if db.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
                    raise ValueError('資料庫完整性檢查失敗')
            if current.execute('SELECT id FROM project').fetchone() != evidence.execute('SELECT id FROM project').fetchone():
                raise ValueError('備份不屬於同一個專案')
            result = repair(current, evidence)
            if not apply or not result['eligible']:
                return result
            with exclusive_file_lock(database.with_suffix('.migration.lock')):
                with closing(sqlite3.connect(database)) as target:
                    safety = database.with_name(database.name + f'.before-independence-repair-{uuid.uuid4().hex}.backup')
                    with closing(sqlite3.connect(safety)) as saved:
                        target.backup(saved)
                    target.execute('BEGIN IMMEDIATE')
                    try:
                        result = repair(target, evidence, apply=True)
                        target.commit()
                    except BaseException:
                        target.rollback()
                        raise
                    result['backup'] = str(safety)
                    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', required=True)
    parser.add_argument('--backup', required=True)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()
    print(json.dumps(repair_files(args.database, args.backup, apply=args.apply, offline=args.offline), ensure_ascii=False))
