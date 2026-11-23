"""Inference contract tests with deterministic stand-ins for the local models."""

from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from PIL import Image

from workbench.annotation_assist_runtime import PROFILE, WorkerProgress, _validated_proof
from workbench.annotation_assist_worker import DEFAULTS, RESULT_SCHEMA, run_inference, select_vote


class GroundingStub:
    def __init__(self, directory):
        pass

    def propose(self, image, prompt, *, candidate_limit):
        if image.size != (80, 54):
            # An empty local search should retain the unresolved full-image exception.
            return []
        return [((3, 5, 28, 37), 0.7), ((3, 5, 28, 37), 0.6), ((30, 10, 52, 24), 0.05)]


class JudgeStub:
    def __init__(self, directory, classes, settings):
        self.settings = settings

    def scan(self, image):
        return {'status': 'parsed', 'targets': [{'label': 'tool', 'xyxy': [3, 5, 28, 37]},
                                              {'label': 'tool', 'xyxy': [66, 40, 75, 52]}]}

    def review(self, image, candidates):
        for row in candidates:
            row.update(select_vote([([0.92, 0.05, 0.02, 0.01], 0.86)] * 2, self.settings))


class AnnotationAssistWorkerTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.manifest = self.root / 'manifest.json'

    def build_job(self, *, hash_mismatch=False, retry_regions=1):
        images = self.root / 'images'
        images.mkdir()
        image = images / 'example.jpg'
        exif = Image.Exif()
        exif[274] = 6
        Image.new('RGB', (80, 54), 'white').save(image, exif=exif)
        self.manifest.write_text(json.dumps({
            'job_id': 'batch-stub',
            'config': {'classes': [{'id': 'c1', 'label': 'tool', 'prompt': 'hammer', 'description': 'Visible tool'}],
                       'settings': {'retry_regions': retry_regions}},
            'frames': [{'id': 'picture', 'image': 'images/example.jpg', 'width': 80, 'height': 54,
                        'image_sha256': '0' * 64 if hash_mismatch else sha256(image.read_bytes()).hexdigest()}],
        }), encoding='utf-8')
        (self.root / 'model.safetensors').write_bytes(b'test')

    def infer(self):
        return run_inference(self.manifest, self.root, self.root, self.root / 'output.json',
                             engine_factory=GroundingStub, judge_factory=JudgeStub)

    def test_raw_pixel_proposal_dedup_vote_and_retry_exception(self):
        self.build_job()
        result = self.infer()
        self.assertEqual(result['schema'], RESULT_SCHEMA)
        frame, = result['frames']
        accepted, low_score = frame['candidates']
        self.assertEqual(accepted['xyxy'], [3, 5, 28, 37])
        self.assertEqual(accepted['verdict'], 'accepted')
        self.assertEqual(low_score['reason_code'], 'low_score_proposal')
        self.assertEqual(frame['boxes'][0]['candidate_id'], accepted['id'])
        self.assertEqual(frame['status'], 'review')
        self.assertEqual([ex['code'] for ex in frame['exceptions']], ['unmatched_scene_target', 'local_retry'])
        self.assertEqual(len(frame['retries']), 1)
        self.assertEqual(frame['confidence']['accepted_count'], 1)

    def test_tampered_image_and_excessive_retry_budget_fail_closed(self):
        self.build_job(hash_mismatch=True)
        with self.assertRaisesRegex(ValueError, '已變更'):
            self.infer()
        self.assertFalse((self.root / 'output.json').exists())
        manifest = json.loads(self.manifest.read_bytes())
        manifest['config']['settings']['retry_regions'] = 3
        self.manifest.write_text(json.dumps(manifest), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, '0 至 2'):
            self.infer()

    def test_requires_agreement_and_allowed_token_mass(self):
        accept = ([0.8, 0.1, 0.05, 0.05], 0.9)
        disagree = ([0.1, 0.05, 0.8, 0.05], 0.9)
        weak_mass = (accept[0], 0.1)
        self.assertEqual(select_vote([accept, disagree], DEFAULTS)['verdict'], 'unresolved')
        self.assertEqual(select_vote([accept, weak_mass], DEFAULTS)['verdict'], 'unresolved')

    def test_proof_is_bound_to_manifest_and_requires_os_isolation(self):
        self.manifest.write_bytes(b'original manifest')
        path = self.root / 'isolation.json'
        proof = {'schema': 'vision_workbench.annotation_assist_isolation.v1', 'profile': PROFILE,
                 'manifest_sha256': sha256(self.manifest.read_bytes()).hexdigest(), 'appcontainer_token': True,
                 'os_network_capabilities': 0, 'network_denied': True, 'host_files_denied': True, 'cuda': True}
        path.write_text(json.dumps(proof), encoding='utf-8')
        self.assertEqual(_validated_proof(path, self.manifest), proof)
        self.manifest.write_bytes(b'tampered')
        with self.assertRaisesRegex(RuntimeError, 'AppContainer'):
            _validated_proof(path, self.manifest)

    def test_incremental_progress_ignores_partial_line(self):
        worker = WorkerProgress(self.root)
        worker.path = self.root / 'worker.log'
        worker.path.write_bytes(b'diagnostic\n{"event":"progress","stage":"dino","completed":1')
        self.assertEqual(worker.read()[2], 'initializing')
        with worker.path.open('ab') as stream:
            stream.write(b',"total":2}\n')
        self.assertEqual(worker.read(), ('MM 尋找候選框 1／2 張', 28, 'dino'))


if __name__ == '__main__':
    unittest.main()
