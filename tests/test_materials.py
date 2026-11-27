from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from workbench.materials import MaterialCatalog, record_annotation, record_video_frame
from workbench.project_layout import project_area, project_reference
from workbench.store import ConflictError, ProjectStore


class MaterialCatalogTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="workbench-materials-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.store = ProjectStore(self.root / 'projects')
        self.project_id = self.store.create_project('素材測試')['id']
        self.project = self.store.directory(self.project_id)
        self.materials = MaterialCatalog(self.store)

    def make_video(self, name='recording_test.mp4', video_bytes=b'video source bytes', *, legacy=False):
        folder = self.project / 'recordings' if legacy else project_area(self.project, 'recordings')
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(video_bytes)
        return path

    def make_image(self, source=None, color=(93, 21, 172)):
        image = self.root / 'photo.png'
        Image.new('RGB', (8, 8), color).save(image)
        image_id, = self.store.add_assets(self.project_id, [{
            'path': str(image), 'name': image.name, 'source': source or {'kind': 'camera'},
        }])['asset_ids']
        return image_id

    def test_registered_recording_does_not_need_a_sidecar_to_read_or_resolve(self):
        path = self.make_video()
        digest = sha256(path.read_bytes()).hexdigest()
        recording = self.materials.register_video(self.project_id, {'path': str(path), 'fps': 30, 'frames': 90})
        self.assertEqual(recording['sha256'], digest)
        self.assertEqual(recording['relative_path'], 'inputs/videos/recording_test.mp4')
        self.assertEqual(recording['frames'], 90)
        self.assertEqual(recording['id'], recording['video_id'])

        with patch('workbench.materials.file_digest', side_effect=AssertionError('Catalog lookup re-read bytes')):
            self.assertEqual(self.materials.list_recordings(self.project_id), [recording])
            self.assertEqual(self.materials.find_recording(self.project_id, path=path), recording)
            self.assertEqual(self.materials.find_recording(self.project_id, sha256=digest), recording)
            self.assertEqual(self.materials.find_recording(self.project_id, video_id=recording['id']), recording)
        self.store.update_project(self.project_id, name='重新命名')
        moved = self.materials.find_recording(self.project_id, video_id=recording['id'])
        self.assertEqual(moved['relative_path'], recording['relative_path'])
        self.assertEqual(Path(moved['path']), self.store.directory(self.project_id) / moved['relative_path'])

    def test_legacy_sidecar_and_frame_were_backfilled_without_rewriting_source(self):
        path = self.make_video(legacy=True)
        digest = sha256(path.read_bytes()).hexdigest()
        path.with_suffix('.json').write_text(json.dumps({
            'project_id': self.project_id, 'fps': 24, 'frames': 48, 'display_name': '舊版錄影.mp4',
        }), encoding='utf-8')
        source = {'kind': 'legacy-import', 'original': {
            'kind': 'video', 'video': r'D:\old-host\recordings\former-name.mp4',
            'video_sha256': digest, 'frame_index': 12, 'timestamp_seconds': .5, 'fps': 24,
        }}
        image_id = self.make_image(source)
        before = self.store.get_asset(self.project_id, image_id)

        self.assertEqual(self.materials.reconcile_recordings(self.project_id),
                         {'registered': 1, 'rejected': 0, 'linked': 1})
        video, = self.materials.list_recordings(self.project_id)
        image, = self.materials.list_materials(self.project_id, 'image')
        link, = image['metadata']['video_sources']
        self.assertEqual(link['video_id'], video['id'])
        self.assertEqual(link['frame_index'], 12)
        self.assertEqual(link['timestamp_seconds'], .5)
        self.assertEqual(link['timestamp_method'], 'frame_fps')
        self.assertEqual(before, self.store.get_asset(self.project_id, image_id))
        path.parent.rename(self.project / 'inputs' / 'videos')
        relocated, = self.materials.list_recordings(self.project_id)
        self.assertEqual(relocated['id'], video['id'])
        self.assertEqual(relocated['relative_path'], 'inputs/videos/recording_test.mp4')
        self.assertEqual(self.materials.find_recording(self.project_id, path=path), relocated)
        video_source, = self.materials.list_materials(self.project_id, 'image')[0]['metadata']['video_sources']
        self.assertEqual(video_source['relative_path'], relocated['relative_path'])
        with patch('workbench.materials.file_digest', side_effect=AssertionError('A known video was rehashed')):
            self.assertEqual(self.materials.reconcile_recordings(self.project_id),
                             {'registered': 0, 'rejected': 0, 'linked': 0})

    def test_only_verified_shared_videos_are_copied_and_linked_on_first_project_use(self):
        import cv2
        import numpy as np
        from workbench.acquisition import _opencv_path
        from workbench.server import WorkbenchService

        service = WorkbenchService(self.root)
        self.addCleanup(service.close)
        directory = service.incoming / 'recordings'
        directory.mkdir()
        video = directory / 'recording_verified.mp4'
        writer = cv2.VideoWriter(_opencv_path(video), cv2.VideoWriter_fourcc(*'mp4v'), 10, (48, 32))
        self.assertTrue(writer.isOpened())
        try:
            for index in range(6):
                writer.write(np.full((32, 48, 3), index * 25, np.uint8))
        finally:
            writer.release()
        original = video.read_bytes()
        digest = sha256(original).hexdigest()
        source = {'kind': 'video', 'video': str(video), 'video_sha256': digest,
                  'frame_index': 4, 'timestamp_seconds': .4, 'fps': 10}
        image_id = self.make_image(source)
        wrong = directory / 'recording_changed.mp4'
        wrong.write_bytes(original)
        self.make_image({**source, 'video': str(wrong), 'video_sha256': 'f' * 64}, color=(89, 21, 172))
        orphan = directory / 'recording_orphan.mp4'
        orphan.write_bytes(original)
        other_id = self.store.create_project('另一個專案')['id']

        service.prepare_materials(other_id)
        self.assertEqual(self.materials.list_recordings(other_id), [])
        with self.assertLogs(level='WARNING') as messages:
            service.prepare_materials(self.project_id)
        self.assertIn('1 unverified shared recordings', messages.output[0])
        recording, = self.materials.list_recordings(self.project_id)
        self.assertEqual(recording['display_name'], video.name)
        self.assertEqual(recording['sha256'], digest)
        self.assertTrue(recording['relative_path'].startswith('inputs/videos/'))
        self.assertEqual(Path(recording['path']).read_bytes(), original)
        image = next(row for row in self.materials.list_materials(self.project_id, 'image') if row['id'] == image_id)
        link, = image['metadata']['video_sources']
        self.assertEqual((link['video_id'], link['frame_index'], link['timestamp_seconds']), (recording['id'], 4, .4))
        self.assertEqual(self.store.get_asset(self.project_id, image_id)['source'], source)
        self.assertEqual({path.name: path.read_bytes() for path in directory.iterdir()},
                         {path.name: original for path in (video, wrong, orphan)})
        service.prepare_materials(self.project_id)
        self.assertEqual(self.materials.list_recordings(self.project_id), [recording])

    def test_links_cover_deduplicated_frames_and_keep_trash_history(self):
        image_id = self.make_image()
        path = self.make_video()
        video = self.materials.register_video(self.project_id, {'path': str(path)})
        data = b'{"shapes":[]}'
        checksum = sha256(data).hexdigest()
        annotation_path = project_area(self.project, 'imported-annotations') / f'{checksum}.json'
        annotation_path.parent.mkdir(parents=True)
        annotation_path.write_bytes(data)
        with self.store.connection(self.project_id, write=True) as db:
            annotation_id = record_annotation(
                db, self.project_id, relative_path=annotation_path.relative_to(self.project).as_posix(),
                sha256=checksum, byte_count=len(data), name='image.json', format='labelme',
                image_id=image_id, annotation_revision=1,
            )
            for frame_index, seconds in [(12, .5), (24, 1.)]:
                record_video_frame(db, image_id, {
                    'video_id': video['id'], 'video_sha256': video['sha256'], 'frame_index': frame_index,
                    'timestamp_seconds': seconds, 'timestamp_method': 'decoder_position',
                })
        before = self.store.get_asset(self.project_id, image_id)
        self.store.trash_assets(self.project_id, [image_id], {image_id: before['revision']})
        rows = {item['kind']: item for item in self.materials.list_materials(self.project_id)}
        self.assertEqual(set(rows), {'image', 'annotation', 'video'})
        self.assertEqual(rows['image']['state'], 'trashed')
        self.assertTrue(rows['image']['relative_path'].startswith('inputs/images/'))
        self.assertTrue(rows['annotation']['relative_path'].startswith('inputs/annotations/'))
        self.assertEqual([link['timestamp_seconds'] for link in rows['image']['metadata']['video_sources']], [.5, 1.])
        self.assertEqual(rows['annotation']['id'], annotation_id)
        self.assertEqual(rows['annotation']['metadata']['linked_images'], [
            {'image_id': image_id, 'annotation_revision': 1},
        ])

        self.store.trash_assets(self.project_id, [image_id], {image_id: before['revision']}, restore=True)
        restored = self.store.get_asset(self.project_id, image_id)
        self.assertEqual(self.materials.list_materials(self.project_id, 'image')[0]['state'], 'active')
        self.store.delete_asset(self.project_id, image_id, restored['revision'])
        self.assertEqual(self.materials.list_materials(self.project_id, 'image'), [])
        self.assertEqual(self.materials.list_materials(self.project_id, 'annotation')[0]['metadata']['linked_images'], [])
        self.assertTrue(annotation_path.is_file())

    def test_shared_annotation_content_keeps_each_image_import_source(self):
        image_ids = [self.make_image(), self.make_image(color=(12, 173, 22))]
        data = b'0 0.3 0.2 0.1 0.1'
        checksum = sha256(data).hexdigest()
        relative_path = f'imported-annotations/{checksum}.txt'
        path = project_reference(self.project, relative_path)
        path.parent.mkdir(parents=True)
        path.write_bytes(data)
        source_paths = dict(zip(image_ids, ('first/a.txt', 'second/b.txt')))
        with self.store.connection(self.project_id, write=True) as db:
            for image_id, source_path in source_paths.items():
                record_annotation(db, self.project_id, relative_path=relative_path, sha256=checksum,
                                  byte_count=len(data), name=Path(source_path).name, format='yolo',
                                  image_id=image_id, annotation_revision=1,
                                  metadata={'original_path': source_path})

        material, = self.materials.list_materials(self.project_id, 'annotation')
        self.assertEqual({link['image_id']: link['source']['original_path']
                          for link in material['metadata']['linked_images']}, source_paths)

    def test_reconcile_ignores_conflicting_sidecar_checksum_and_foreign_project(self):
        first = self.make_video('recording_first.mp4')
        second = self.make_video('recording_second.mp4')
        first.with_suffix('.json').write_text(json.dumps({
            'project_id': self.project_id, 'sha256': 'a' * 64,
        }), encoding='utf-8')
        second.with_suffix('.json').write_text(json.dumps({
            'project_id': 'b' * 32,
        }), encoding='utf-8')
        self.assertEqual(self.materials.reconcile_recordings(self.project_id),
                         {'registered': 0, 'rejected': 2, 'linked': 0})
        self.assertEqual(self.materials.list_recordings(self.project_id), [])
        with self.assertRaises(ConflictError):
            self.materials.register_video(self.project_id, {'path': str(first), 'sha256': 'a' * 64})

    def test_reconcile_reads_unmigrated_clips_in_mixed_layout(self):
        current = self.make_video('recording_current.mp4')
        older = self.make_video('recording_older.mp4', b'older bytes', legacy=True)
        for path in (current, older):
            path.with_suffix('.json').write_text(json.dumps({'project_id': self.project_id}), encoding='utf-8')

        self.assertEqual(self.materials.reconcile_recordings(self.project_id)['registered'], 2)
        paths = {row['relative_path'] for row in self.materials.list_recordings(self.project_id)}
        self.assertEqual(paths, {'inputs/videos/recording_current.mp4', 'recordings/recording_older.mp4'})

    def test_merged_provenance_does_not_bind_another_projects_video_by_name(self):
        path = self.make_video()
        self.materials.register_video(self.project_id, {'path': str(path)})
        self.make_image({'project_id': 'b' * 32, 'original': {
            'kind': 'video', 'video': 'recordings/recording_test.mp4',
            'frame_index': 12, 'timestamp_seconds': .5,
        }})

        self.assertEqual(self.materials.reconcile_recordings(self.project_id)['linked'], 0)
        image, = self.materials.list_materials(self.project_id, 'image')
        self.assertEqual(image['metadata']['video_sources'], [])


if __name__ == '__main__':
    unittest.main()
