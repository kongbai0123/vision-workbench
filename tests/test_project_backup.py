from hashlib import sha256
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest
import zipfile

from workbench.project_backup import MANIFEST, create_project_backup, restore_project_backup
from workbench.project_layout import project_area
from workbench.store import ConflictError, ProjectStore, dump


class ProjectBackupTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = ProjectStore(self.root / 'source' / 'projects')
        self.destination = ProjectStore(self.root / 'destination' / 'projects')
        self.project = self.source.create_project('握把外觀', classes=['grip_done', 'grip'])
        self.project_id = self.project['id']

    def backup(self):
        return Path(create_project_backup(self.source, self.project_id, self.root / 'backups')['path'])

    def rewrite(self, archive, change):
        with zipfile.ZipFile(archive) as bundle:
            members = {name: bundle.read(name) for name in bundle.namelist()}
        manifest = json.loads(members.pop(MANIFEST))
        change(members, manifest)
        path = self.root / 'modified.vwb-project.zip'
        with zipfile.ZipFile(path, 'w') as bundle:
            for name, content in members.items():
                bundle.writestr(name, content)
            bundle.writestr(MANIFEST, json.dumps(manifest, ensure_ascii=False))
        return path

    def test_round_trip_preserves_committed_wal_and_no_duplicate_overwrite(self):
        # Keep a SQLite reader in a transaction across the write and backup so
        # the most recent commit remains in the live source's WAL.
        source = self.source.directory(self.project_id) / 'project.sqlite3'
        reader = sqlite3.connect(source)
        try:
            reader.execute('BEGIN')
            reader.execute('SELECT classes FROM project').fetchone()
            self.source.update_project(self.project_id, classes=['grip_done', 'mid_done', 'grip'])
            self.assertTrue(source.with_name(source.name + '-wal').exists())
            archive = self.backup()
        finally:
            reader.close()
        with zipfile.ZipFile(archive) as bundle:
            self.assertEqual(set(bundle.namelist()), {MANIFEST, 'project/project.sqlite3'})
        restored = restore_project_backup(self.destination, archive)
        self.assertEqual(restored['project_id'], self.project_id)
        self.assertEqual(self.destination.get_project(self.project_id)['classes'], ['grip_done', 'mid_done', 'grip'])
        self.assertTrue(self.destination.database_integrity(self.project_id)['ok'])
        with self.assertRaises(ConflictError):
            restore_project_backup(self.destination, archive)
        self.assertEqual(len(self.destination.list_projects()), 1)

    def test_corrupted_payload_is_not_published(self):
        archive = self.backup()

        def corrupt(members, _manifest):
            payload = members['project/project.sqlite3']
            members['project/project.sqlite3'] = bytes([payload[0] ^ 1]) + payload[1:]

        with self.assertRaisesRegex(ValueError, 'SHA-256'):
            restore_project_backup(self.destination, self.rewrite(archive, corrupt))
        self.assertEqual(list(self.destination.root.iterdir()), [])

    def test_material_file_must_match_database_hash_even_with_valid_archive_hash(self):
        payload = b'{"shapes":[]}'
        relative = 'inputs/annotations/source.json'
        annotation = self.source.directory(self.project_id) / relative
        annotation.parent.mkdir(parents=True, exist_ok=True)
        annotation.write_bytes(payload)
        with self.source.connection(self.project_id, write=True) as db:
            db.execute(
                'INSERT INTO project_material_files VALUES (?,?,?,?,?,?,?,?)',
                ('a' * 32, 'annotation', 'imported-annotations/source.json', annotation.name, sha256(payload).hexdigest(),
                 len(payload), '{}', '2026-11-26T00:00:00+00:00'),
            )
        archive = self.backup()

        def corrupt_material(members, manifest):
            different_bytes = b'{"shapes":{}}'
            members['project/' + relative] = different_bytes
            manifest['files'][relative] = {
                'bytes': len(different_bytes), 'sha256': sha256(different_bytes).hexdigest(),
            }

        with self.assertRaisesRegex(ValueError, 'SHA-256'):
            restore_project_backup(self.destination, self.rewrite(archive, corrupt_material))
        self.assertEqual(list(self.destination.root.iterdir()), [])

    def test_rejects_traversal_aliases_and_case_collisions(self):
        archive = self.backup()
        for files in (('../escaped.txt',), ('images/NUL.png',), ('images/new.jpg:secret',),
                      ('Images/one.jpg', 'images/two.jpg'), ('images.',)):
            with self.subTest(files=files):
                def add_invalid(members, manifest):
                    for name in files:
                        members['project/' + name] = b'x'
                        manifest['files'][name] = {'bytes': 1, 'sha256': sha256(b'x').hexdigest()}

                with self.assertRaises(ValueError):
                    restore_project_backup(self.destination, self.rewrite(archive, add_invalid))
                self.assertEqual(list(self.destination.root.iterdir()), [])
                self.assertFalse((self.destination.root / 'escaped.txt').exists())

    def test_rejects_modified_sqlite_schema_even_with_matching_checksum(self):
        archive = self.backup()

        def mutate_schema(members, manifest):
            database = self.root / 'changed.sqlite3'
            database.write_bytes(members['project/project.sqlite3'])
            connection = sqlite3.connect(database)
            try:
                connection.execute('CREATE TRIGGER altered_schema AFTER UPDATE ON project BEGIN DELETE FROM history; END')
                connection.commit()
            finally:
                connection.close()
            payload = database.read_bytes()
            members['project/project.sqlite3'] = payload
            manifest['files']['project.sqlite3'] = {'bytes': len(payload), 'sha256': sha256(payload).hexdigest()}

        with self.assertRaisesRegex(ValueError, '資料庫結構'):
            restore_project_backup(self.destination, self.rewrite(archive, mutate_schema))
        self.assertEqual(list(self.destination.root.iterdir()), [])

    def test_changes_to_source_during_backup_do_not_publish(self):
        project_folder = self.source.directory(self.project_id)
        (project_folder / 'notes.txt').write_text('before', encoding='utf-8')
        changed = False

        def mutate_on_copy(message, *_):
            nonlocal changed
            if message == '備份專案檔案' and not changed:
                (project_folder / 'notes.txt').write_text('changed', encoding='utf-8')
                changed = True

        with self.assertRaises(ConflictError):
            create_project_backup(self.source, self.project_id, self.root / 'backups', progress=mutate_on_copy)
        self.assertEqual(list((self.root / 'backups').iterdir()), [])

    def test_preserves_stable_video_recovery_fragment(self):
        # Backups taken while a project still uses a legacy directory retain
        # the fragment and place it in the destination's canonical video area.
        source = self.source.directory(self.project_id) / 'recordings' / 'recording_capture.part.mp4'
        source.parent.mkdir()
        payload = b'unfinished capture available for file recovery'
        source.write_bytes(payload)

        restore_project_backup(self.destination, self.backup())

        restored = self.destination.directory(self.project_id) / 'inputs' / 'videos' / source.name
        self.assertEqual(restored.read_bytes(), payload)

    def test_backup_excludes_canonical_work_scratch(self):
        root = self.source.directory(self.project_id)
        scratch = project_area(root, 'scratch') / 'current' / 'capture.png'
        scratch.parent.mkdir(parents=True)
        scratch.write_bytes(b'temporary capture bytes')
        archive = self.backup()
        with zipfile.ZipFile(archive) as bundle:
            manifest = json.loads(bundle.read(MANIFEST))
            self.assertFalse(any(name.startswith('work/scratch') for name in manifest['files']))
            self.assertFalse(any(name.startswith('work/scratch') for name in manifest['directories']))
        restore_project_backup(self.destination, archive)
        restored = self.destination.directory(self.project_id)
        self.assertFalse((restored / scratch.relative_to(root)).exists())

    def test_external_export_paths_become_source_history(self):
        external = self.root / 'external-exports'
        external.mkdir()
        bundle = external / 'output.zip'
        bundle.write_bytes(b'external export')
        original = self.source.record_export(self.project_id, {
            'path': str(external), 'zip_path': str(bundle), 'version': 'outside',
        })

        restore_project_backup(self.destination, self.backup())

        history, = self.destination.get_project(self.project_id)['exports']
        self.assertEqual(history['id'], original['id'])
        self.assertEqual(history['original_paths'], {'path': str(external), 'zip_path': str(bundle)})
        self.assertNotIn('path', history)
        self.assertNotIn('zip_path', history)

    def test_legacy_project_outputs_are_rebased_after_physical_migration(self):
        root = self.source.directory(self.project_id)
        dataset_output = root / 'exports' / 'ชุด-甲'
        dataset_output.mkdir(parents=True)
        dataset_zip = root / 'exports' / 'ชุด-甲.zip'
        dataset_zip.write_bytes(b'dataset output bytes')
        model_output = root / 'model-exports' / 'E001'
        model_output.mkdir(parents=True)
        model_zip = model_output / 'result.zip'
        model_zip.write_bytes(b'model output bytes')
        (model_output / 'record.json').write_text(dump({
            'path': str(model_zip), 'directory': str(model_output),
            'sha256': sha256(model_zip.read_bytes()).hexdigest(),
        }), encoding='utf-8')
        self.source.record_export(self.project_id, {
            'path': str(dataset_output), 'zip_path': str(dataset_zip), 'version': 'legacy',
        })

        restore_project_backup(self.destination, self.backup())

        restored = self.destination.directory(self.project_id)
        history, = self.destination.get_project(self.project_id)['exports']
        expected_zip = restored / 'outputs' / 'datasets' / dataset_zip.name
        self.assertEqual(Path(history['zip_path']), expected_zip)
        self.assertEqual(expected_zip.read_bytes(), dataset_zip.read_bytes())
        restored_model = restored / 'outputs' / 'models' / model_output.name
        record = json.loads((restored_model / 'record.json').read_text(encoding='utf-8'))
        self.assertEqual(record['path'], model_zip.name)
        self.assertEqual((restored_model / record['path']).read_bytes(), model_zip.read_bytes())

    def test_reconciled_failed_run_is_preserved_and_active_run_is_rejected(self):
        project_folder = self.source.directory(self.project_id)
        dataset_dir = project_folder / 'datasets' / 'D001'
        dataset_dir.mkdir(parents=True)
        manifest = {'dataset_version_id': 'D001', 'assets': []}
        manifest['manifest_sha256'] = sha256(dump(manifest).encode('utf-8')).hexdigest()
        (dataset_dir / 'manifest.json').write_text(dump(manifest), encoding='utf-8')
        run = project_folder / 'runs' / 'R001'
        (run / 'control').mkdir(parents=True)
        (run / 'run.json').write_text(dump({'dataset_version_id': 'D001', 'status': 'running'}), encoding='utf-8')
        with self.assertRaisesRegex(ConflictError, '訓練'):
            self.backup()
        terminal = run / 'control' / 'terminal.json'
        terminal.write_text(dump({'status': 'failed'}), encoding='utf-8')

        restore_project_backup(self.destination, self.backup())
        restored = project_area(self.destination.directory(self.project_id), 'runs') / 'R001' / 'control' / terminal.name
        self.assertEqual(json.loads(restored.read_text(encoding='utf-8'))['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
