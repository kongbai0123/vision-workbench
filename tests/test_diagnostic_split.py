import hashlib
import json
from pathlib import Path
import tempfile
import threading
import unittest

from workbench.diagnostic_split import diagnostic_temporal_split
from workbench.training import TrainingWorkspace


def assets():
    return [{"asset_id": f"{label}-{i}", "name": f"sample_{i:06d}.png", "batch_id": label,
             "shapes": [{"label": label}], "split": "val", "sha256": f"{label}-{i}"}
            for label in ("a", "b") for i in range(14)]


class DiagnosticSplitTests(unittest.TestCase):
    def test_temporal_blocks_exclude_boundaries_and_cover_every_class(self):
        original = assets()
        before = json.dumps(original)
        plan = diagnostic_temporal_split(original)
        self.assertEqual(json.dumps(original), before)
        self.assertEqual(plan["coverage"]["image_counts"], {"train": 16, "val": 4, "test": 4})
        self.assertFalse(plan["data_quality"]["independent_sources"])
        self.assertEqual(plan["data_quality"]["purpose"], "diagnostic")
        self.assertEqual({x["asset_id"] for x in plan["data_quality"]["excluded_assets"]}, {"a-8", "a-11", "b-8", "b-11"})
        self.assertEqual(plan["assignments"]["a-9"], "val")
        self.assertEqual(plan["assignments"]["a-12"], "test")

    def test_duplicate_images_across_blocks_are_rejected(self):
        rows = assets()
        rows[12]["sha256"] = rows[0]["sha256"]
        with self.assertRaisesRegex(ValueError, "完全相同"):
            diagnostic_temporal_split(rows)

    def test_insufficient_or_ambiguous_sources_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "沒有圖片"):
            diagnostic_temporal_split([])
        with self.assertRaisesRegex(ValueError, "不足"):
            diagnostic_temporal_split(assets()[:7])
        rows = assets()
        rows[0]["shapes"].append({"label": "other"})
        with self.assertRaisesRegex(ValueError, "單一類別"):
            diagnostic_temporal_split(rows)

    def test_new_version_preserves_source_and_records_exclusions(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = TrainingWorkspace.__new__(TrainingWorkspace)
            workspace.datasets = Path(directory)
            workspace.lock = threading.RLock()
            source = workspace.datasets / "pid" / "D004"
            (source / "images").mkdir(parents=True)
            rows = assets()
            for row in rows:
                raw = row["asset_id"].encode()
                filename = f"{row['asset_id']}.png"
                (source / "images" / filename).write_bytes(raw)
                row.update(image_file=f"images/{filename}", sha256=hashlib.sha256(raw).hexdigest())
            manifest = {"dataset_version_id": "D004", "project_revision": 1, "classes": ["a", "b"],
                        "manifest_sha256": "original", "assets": rows}
            path = source / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            before = path.read_bytes()
            result = workspace.create_diagnostic_dataset_version("pid", "D004")
            self.assertEqual(result["id"], "D005")
            self.assertEqual(path.read_bytes(), before)
            created = json.loads((source.parent / "D005" / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(len(created["assets"]), 24)
            self.assertEqual(created["data_quality"]["source_dataset_version_id"], "D004")
            self.assertEqual(len(created["data_quality"]["excluded_assets"]), 4)
            self.assertNotEqual(created["manifest_sha256"], "original")


if __name__ == "__main__":
    unittest.main()
