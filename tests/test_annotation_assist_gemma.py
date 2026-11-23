import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from PIL import Image

from workbench import annotation_assist_gemma as gemma
from workbench.annotation_assist_worker import RESULT_SCHEMA, file_hash


class AnnotationAssistGemmaTests(unittest.TestCase):
    def test_round_robin_supports_all_verdicts_and_keeps_original_boxes(self):
        classes = [{"id": "hammer", "label": "hammer", "prompt": "hammer", "description": ""}]
        manifest = {"config": {"classes": classes}, "frames": [
            {"id": name, "width": 100, "height": 100, "image_sha256": name} for name in ("one", "two")
        ]}
        candidates = [
            {"id": "adopted", "label": "hammer", "xyxy": [1, 2, 11, 12], "score": .5, "verdict": "accepted"},
            {"id": "uncertain", "label": "hammer", "xyxy": [30, 40, 40, 50], "score": .4, "verdict": "unresolved"},
            {"id": "threshold", "label": "hammer", "xyxy": [3, 4, 8, 9], "score": .88, "verdict": "rejected"},
            {"id": "qwen-seen", "label": "hammer", "xyxy": [50, 50, 58, 58], "score": .55,
             "verdict": "rejected", "votes": [{"winner": "C"}]},
            {"id": "invalid", "label": "hammer", "xyxy": [-1, -2, 100, 100], "score": .99, "verdict": "accepted"},
        ]
        inference = {"method": {"classes": classes}, "frames": [
            {"id": name, "image_sha256": name, "candidates": candidates} for name in ("one", "two")
        ]}
        with patch.object(gemma, "MAX_CASES", 4):
            cases, eligible = gemma.collect_cases(manifest, inference)
        self.assertEqual(eligible, 8)
        self.assertEqual([(frame["id"], candidate["id"]) for frame, candidate, _ in cases],
                         [("one", "adopted"), ("one", "uncertain"), ("one", "qwen-seen"), ("two", "adopted")])
        self.assertEqual(cases[2][1]["xyxy"], [50, 50, 58, 58])
        inference["frames"][0]["image_sha256"] = "changed"
        with self.assertRaisesRegex(ValueError, "雜湊"):
            gemma.collect_cases(manifest, inference)

    def test_wrong_acceptance_fails_even_if_positive_is_accepted(self):
        checks = [
            {"kind": "blank", "expected": "rejected", "verdict": "accepted"},
            {"kind": "wrong_location", "expected": "rejected", "verdict": "rejected"},
            {"kind": "wrong_class", "expected": "rejected", "verdict": "unresolved"},
            {"kind": "positive", "expected": "accepted", "verdict": "accepted"},
        ]
        self.assertEqual(gemma.health_status(checks)["status"], "failed")
        checks[0]["verdict"] = "rejected"
        self.assertEqual(gemma.health_status(checks)["status"], "inconclusive")
        checks[2]["verdict"] = "rejected"
        self.assertEqual(gemma.health_status(checks)["status"], "passed")

    def test_sidecar_hash_binds_empty_batch_without_publishing_any_new_boxes(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            image = folder / "asset.png"
            Image.new("RGB", (8, 6)).save(image)
            image_hash = file_hash(image)
            classes = [{"id": 0, "label": "part", "prompt": "part", "description": ""}]
            manifest = folder / "manifest.json"
            manifest.write_text(json.dumps({"job_id": "01", "config": {"classes": classes}, "frames": [
                {"id": "image-A", "image": image.name, "image_sha256": image_hash, "width": 8, "height": 6},
            ]}), encoding="utf-8")
            inference = folder / "inference.json"
            inference.write_text(json.dumps({"schema": RESULT_SCHEMA, "status": "complete", "job_id": "01",
                                             "manifest_sha256": file_hash(manifest), "method": {"classes": classes},
                                             "frames": [{"id": "image-A", "image_sha256": image_hash, "candidates": []}]}),
                                 encoding="utf-8")
            original = inference.read_bytes()
            output = folder / "observation.json"
            sidecar = gemma.observe(manifest, inference, folder, output)
            self.assertEqual(sidecar["manifest_sha256"], file_hash(manifest))
            self.assertEqual(sidecar["inference_sha256"], file_hash(inference))
            self.assertTrue(sidecar["observation_only"])
            self.assertEqual(sidecar["rows"], [])
            self.assertNotIn("frames", sidecar)
            self.assertEqual(sidecar["health"]["status"], "not_run")
            self.assertEqual(inference.read_bytes(), original)
            manifest.write_bytes(manifest.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "清單"):
                gemma.observe(manifest, inference, folder, folder / "unexpected.json")


if __name__ == "__main__":
    unittest.main()
