"""SAM boundary evidence and review routing without requiring a GPU."""

from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from PIL import Image

from workbench.annotation_assist_sam import SAM_FILES, MAX_BOXES_PER_FRAME, check_sam_frame, verify_sam_files


class SamStub:
    def __init__(self, bounds, score, *, failure=None):
        self.bounds, self.score = bounds, score
        self.failure = failure
        self.calls = []

    def measure(self, image, prompt, image_hash):
        self.calls.append((image.size, prompt, image_hash))
        if self.failure:
            raise self.failure
        return self.bounds, self.score


def candidate(candidate_id, verdict="accepted", score=0.90):
    return {"id": candidate_id, "label": "tool", "xyxy": [10, 10, 30, 30], "score": score, "verdict": verdict}


class AnnotationAssistSamTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (80, 60))
        self.frame = {"image_sha256": "0" * 64, "status": "ready", "exceptions": [],
                      "candidates": [candidate("ready-1")],
                      "boxes": [{"candidate_id": "ready-1", "xyxy": [10, 10, 30, 30]}]}

    def test_larger_extent_difference_routes_to_review_without_replacing_original_box(self):
        model = SamStub([10, 10, 20, 20], 0.9)
        original_boxes = deepcopy(self.frame["boxes"])
        check_sam_frame(model, self.image, self.frame)
        row, = self.frame["candidates"]
        self.assertEqual(row["sam"]["status"], "mismatch")
        self.assertEqual(row["sam"]["iou"], 0.25)
        self.assertEqual(row["sam"]["mask_box_xyxy"], [10, 10, 20, 20])
        self.assertEqual(row["xyxy"], [10, 10, 30, 30])
        self.assertEqual(row["verdict"], "accepted")
        self.assertEqual(self.frame["boxes"], original_boxes)
        self.assertEqual(self.frame["status"], "review")
        exception, = self.frame["exceptions"]
        self.assertEqual((exception["code"], exception["candidate_id"]), ("sam_boundary_disagreement", "ready-1"))

    def test_iou_at_threshold_stays_aligned_while_low_quality_mask_is_unavailable(self):
        check_sam_frame(SamStub([10, 10, 25, 30], 0.83), self.image, self.frame)
        self.assertEqual(self.frame["candidates"][0]["sam"]["status"], "aligned")
        self.assertEqual(self.frame["exceptions"], [])
        check_sam_frame(SamStub([10, 10, 25, 30], 0.20), self.image, self.frame)
        self.assertEqual(self.frame["candidates"][0]["sam"]["status"], "unavailable")
        self.assertEqual(self.frame["exceptions"][0]["code"], "sam_boundary_unavailable")

    def test_budget_prefers_accepted_and_leaves_rejected_proposals_untouched(self):
        unresolved = candidate("unresolved", verdict="unresolved")
        rejected = candidate("rejected", verdict="rejected")
        self.frame["candidates"] = [unresolved, rejected] + [candidate(str(i)) for i in range(MAX_BOXES_PER_FRAME)]
        model = SamStub([10, 10, 30, 30], 0.9)
        check_sam_frame(model, self.image, self.frame)
        self.assertEqual(len(model.calls), MAX_BOXES_PER_FRAME)
        self.assertEqual(unresolved["sam"]["status"], "skipped_budget")
        self.assertEqual(rejected["sam"]["status"], "not_eligible")
        self.assertEqual(rejected["verdict"], "rejected")

    def test_report_image_model_errors_without_producing_geometry(self):
        check_sam_frame(SamStub(None, None, failure=RuntimeError("CUDA error")), self.image, self.frame)
        evidence = self.frame["candidates"][0]["sam"]
        self.assertEqual(evidence["status"], "unavailable")
        self.assertIsNone(evidence["mask_box_xyxy"])
        self.assertIn("CUDA error", evidence["diagnostic"])

    def test_pin_rejects_equal_size_tampering(self):
        with TemporaryDirectory() as directory:
            model = Path(directory) / "config.json"
            model.write_bytes(b"one")
            with patch.dict(SAM_FILES, {model.name: (3, sha256(b"one").hexdigest())}, clear=True):
                self.assertEqual(verify_sam_files(directory), Path(directory).resolve())
                model.write_bytes(b"two")
                with self.assertRaises(ValueError):
                    verify_sam_files(directory)
