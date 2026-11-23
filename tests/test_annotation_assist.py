"""Exercise the native API and persistence with an in-process offline worker fixture."""
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
import time
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from PIL import Image

from workbench.annotation_assist import AnnotationAssist, GEMMA_SCHEMA, RESULT_SCHEMA
from workbench.server import WorkbenchService


def rectangle(shape_id, label, x):
    return {'id': shape_id, 'type': 'rectangle', 'label': label, 'x': x, 'y': 10,
            'width': 20, 'height': 25, 'hidden': False}


class AnnotationAssistTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.service = WorkbenchService(self.root / 'data').start()
        self.addCleanup(self.service.close)
        release = Event()
        self.started, self.release = Event(), release
        self.addCleanup(release.set)
        self.project = self.service.store.create_project('原生標註')
        self.pid = self.project['id']
        self.service.store.manage_classes(self.pid, ['hammer', 'tool'], {}, self.project['revision'])
        path = self.root / 'test.png'
        Image.new('RGB', (105, 71)).save(path)
        self.aid = self.service.store.add_assets(self.pid, [{'path': path, 'shapes': []}])['asset_ids'][0]
        self.base = f'/api/projects/{self.pid}'
        self.batch = f'{self.base}/annotation-assist'
        self.asset = f'{self.base}/assets/{self.aid}'
        self.classes = [{'label': 'hammer', 'prompt': 'hammer head and handle', 'description': 'tight box'}]

    def call(self, path, payload=None, method=None):
        headers = {'Content-Type': 'application/json', 'X-Workbench': '1', 'Origin': self.service.url,
                   'X-Workbench-Token': self.service.api_token}
        request = Request(self.service.url + path,
                          data=json.dumps(payload).encode() if payload is not None else None,
                          method=method or ('POST' if payload is not None else 'GET'), headers=headers)
        with urlopen(request, timeout=5) as response:
            return json.loads(response.read())

    def wait(self, job):
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            result = self.call(f"/api/jobs/{job['id']}")
            if result['state'] in {'succeeded', 'failed', 'cancelled'}:
                return result
            time.sleep(.025)
        self.fail('annotation assist job did not finish')

    def fake_worker(self, _root, input_path, output, progress, *, tampered=False):
        manifest = json.loads(input_path.read_bytes())
        frame, = manifest['frames']
        path = input_path.parent / frame['image']
        self.assertEqual(sha256(path.read_bytes()).hexdigest(), frame['image_sha256'])
        self.assertEqual(manifest['config']['classes'][0]['prompt'], self.classes[0]['prompt'])
        self.assertFalse((input_path.parent / 'project.sqlite3').exists())
        self.started.set()
        while not self.release.wait(.025):
            progress('等待測試事件', 20)
        progress('覆核一張影像', 76, phase='qwen')
        candidates = [
            {'id': 'match', 'label': 'hammer', 'xyxy': [10, 10, 30, 35], 'score': .88, 'verdict': 'accepted'},
            {'id': 'elsewhere', 'label': 'hammer', 'xyxy': [60, 10, 80, 35], 'score': .38, 'verdict': 'unresolved',
             'reason_code': 'uncertain_review', 'votes': [{'winner': 'A'}, {'winner': 'D'}]},
        ]
        output.write_text(json.dumps({
            'schema': RESULT_SCHEMA, 'job_id': manifest['job_id'], 'status': 'complete',
            'manifest_sha256': 'bad' if tampered else sha256(input_path.read_bytes()).hexdigest(),
            'frames': [{'id': frame['id'], 'image_sha256': frame['image_sha256'], 'status': 'review',
                        'candidates': candidates, 'boxes': [{'candidate_id': 'match'}],
                        'exceptions': [{'code': 'uncertain_review', 'reason': '兩次結果不一致', 'candidate_id': 'elsewhere'}]}],
        }), encoding='utf-8')
        return {'fixture': True}

    def launch(self, worker):
        self.service.annotation_assist.runner = worker
        return self.call(self.batch, {'asset_ids': [self.aid], 'classes': self.classes})

    def test_one_run_preserves_current_manual_shapes_and_reopens_edited_review(self):
        original = self.call(self.asset)
        job = self.launch(self.fake_worker)
        self.assertTrue(self.started.wait(5))
        run_path = f"{self.batch}/{job['run_id']}"
        self.assertEqual(self.call(run_path)['frames'][0]['base_revision'], original['revision'])
        saved = self.call(self.asset, {'shapes': [rectangle('user-box', 'tool', 10)],
                                       'revision': original['revision']}, method='PUT')
        self.release.set()
        self.assertEqual(self.wait(job)['state'], 'succeeded')
        batch = self.call(run_path)
        self.assertEqual(batch['status'], 'complete')
        frame, = batch['frames']
        self.assertTrue(frame['stale'])
        self.assertEqual(frame['revision'], saved['revision'])
        self.assertEqual(frame['candidates'][0]['shape']['width'], 20)
        self.assertEqual(frame['candidates'][1]['votes'][1]['winner'], 'D')
        self.assertEqual(self.call(self.asset)['shapes'], saved['shapes'])

        with self.assertRaises(HTTPError) as failure:
            self.call(run_path + '/review', {'asset_id': self.aid, 'revision': original['revision']})
        self.assertEqual(failure.exception.code, 409)

        verified = self.call(run_path + '/review', {'asset_id': self.aid, 'revision': saved['revision'],
                             'candidate_ids': ['match', 'elsewhere'], 'decisions': [{'candidate_id': 'match', 'reason': '以人工類別為準'}]})
        self.assertEqual(verified['review']['adopted_candidate_ids'], ['elsewhere'])
        self.assertEqual(verified['review']['skipped_candidate_ids'], ['match'])
        self.assertEqual(verified['summary']['pending_review_frames'], 0)
        self.assertEqual(verified['asset']['shapes'][0], saved['shapes'][0])
        self.assertEqual(verified['asset']['review_state'], 'pending')
        self.assertEqual(verified['asset']['shapes'][1]['metadata']['annotation_assist']['candidate_id'], 'elsewhere')
        self.assertEqual(verified['asset']['shapes'][1]['metadata']['source'], 'manual')
        self.assertTrue(verified['asset']['shapes'][1]['metadata']['annotation_assist']['human_confirmed'])

        self.call(f'{self.base}/review', {'asset_ids': [self.aid], 'state': 'approved',
                                         'revisions': {self.aid: verified['asset']['revision']}})
        self.assertEqual(self.call(run_path)['frames'][0]['review']['status'], 'resolved')
        revised = self.call(self.asset)
        revised['shapes'][1]['x'] += 5
        self.call(self.asset, {'shapes': revised['shapes'], 'revision': revised['revision']}, method='PUT')
        restored = AnnotationAssist(self.service.data_root, self.service.store, self.service.jobs).detail(self.pid, job['run_id'])
        self.assertEqual(restored['frames'][0]['review']['status'], 'outdated')
        self.assertEqual(restored['summary']['pending_review_frames'], 1)
        self.assertEqual(restored['frames'][0]['candidates'][1]['xyxy'], [60, 10, 80, 35])

    def test_invalid_worker_output_and_cancel_never_publish_candidates(self):
        job = self.launch(self.fake_worker)
        self.assertTrue(self.started.wait(5))
        self.call(f"/api/jobs/{job['id']}/cancel", {})
        self.assertEqual(self.wait(job)['state'], 'cancelled')
        self.assertEqual(self.call(f"{self.batch}/{job['run_id']}")['summary']['candidates'], 0)
        self.release.set()
        second = self.launch(lambda *args: self.fake_worker(*args, tampered=True))
        self.assertEqual(self.wait(second)['state'], 'failed')
        self.assertEqual(self.call(f"{self.batch}/{second['run_id']}")['summary']['candidates'], 0)
        self.assertEqual(self.call(self.asset)['shapes'], [])

    def test_batch_cannot_select_foreign_assets_or_unknown_classes(self):
        with self.assertRaises(HTTPError) as failure:
            self.call(self.batch, {'asset_ids': [self.aid], 'classes': [{'label': 'unknown'}]})
        self.assertEqual(failure.exception.code, 400)
        with self.assertRaises(HTTPError) as failure:
            self.call(self.batch, {'asset_ids': ['a' * 32], 'classes': self.classes})
        self.assertEqual(failure.exception.code, 404)
        self.assertEqual(self.call(self.batch)['runs'], [])

    def test_sam_evidence_and_gemma_comparison_do_not_change_primary_candidates(self):
        self.release.set()

        def optional_worker(root, manifest, output, progress):
            proof = self.fake_worker(root, manifest, output, progress)
            settings = json.loads(manifest.read_bytes())['config']['settings']
            self.assertTrue(settings['sam_extent'] and settings['gemma_observation'])
            primary = json.loads(output.read_bytes())
            frame, = primary['frames']
            frame['candidates'][0]['sam'] = {'status': 'mismatch', 'mask_box_xyxy': [10, 10, 24, 29], 'iou': .532}
            output.write_text(json.dumps(primary), encoding='utf-8')
            output.with_name('gemma-observation.json').write_text(json.dumps({
                'schema': GEMMA_SCHEMA, 'observation_only': True, 'job_id': primary['job_id'],
                'manifest_sha256': primary['manifest_sha256'], 'inference_sha256': sha256(output.read_bytes()).hexdigest(),
                'health': {'status': 'failed'}, 'rows': [{
                    'asset_id': frame['id'], 'image_sha256': frame['image_sha256'], 'candidate_id': 'match', 'label': 'hammer',
                    'xyxy': [10, 10, 30, 35], 'verdict': 'rejected', 'qwen_comparison': 'disagreement',
                }],
            }), encoding='utf-8')
            return proof

        self.service.annotation_assist.runner = optional_worker
        job = self.call(self.batch, {'asset_ids': [self.aid], 'classes': self.classes,
                                    'model_checks': {'sam': True, 'gemma': True}})
        self.assertEqual(self.wait(job)['state'], 'succeeded')
        batch = self.call(f"{self.batch}/{job['run_id']}")
        frame, = batch['frames']
        self.assertEqual(frame['accepted_candidate_ids'], ['match'])
        self.assertEqual([item['code'] for item in frame['exceptions']], ['uncertain_review'])
        self.assertEqual(frame['candidates'][0]['xyxy'], [10, 10, 30, 35])
        self.assertEqual(frame['candidates'][0]['sam']['mask_box_xyxy'], [10, 10, 24, 29])
        self.assertEqual(frame['candidates'][0]['gemma']['qwen_comparison'], 'disagreement')
        self.assertEqual(batch['gemma']['health']['status'], 'failed')
        self.assertEqual(self.call(self.asset)['shapes'], [])
