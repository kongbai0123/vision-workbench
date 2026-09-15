"""Project container migration, SQL lineage, and immutable image reuse."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

from PIL import Image

from workbench.store import ProjectStore
from workbench.training import TrainingWorkspace


class ProjectStorageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.data = self.root / "data"
        self.store = ProjectStore(self.data / "projects")
        self.pid = self.store.create_project("portable")["id"]

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def write_json(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def test_legacy_artifacts_migrate_and_sql_catalog_preserves_lineage(self):
        dataset = {"dataset_version_id": "D001", "project_revision": 1,
                   "created_at": "2026-09-15T00:00:00+00:00", "manifest_sha256": "a" * 64, "assets": []}
        run = {"run_id": "R001", "dataset_version_id": "D001", "model_version_id": "M001",
               "engine": "pixel_prototype_v1", "status": "completed", "created_at": 1.0, "updated_at": 2.0}
        model = {"model_version_id": "M001", "run_id": "R001", "dataset_version_id": "D001",
                 "engine": "pixel_prototype_v1", "created_at": 3.0}
        export = {"export_id": "E001", "model_version_id": "M001", "run_id": "R001",
                  "dataset_version_id": "D001", "created_at": "2026-09-15T00:00:00+00:00",
                  "sha256": "d" * 64, "bytes": 4}
        candidate_id = "1" * 32
        prediction = {"candidate_id": candidate_id, "project_id": self.pid, "model_version_id": "M001",
                      "run_id": "R001", "created_at": "2026-09-15T00:00:00+00:00",
                      "assets": [{"status": "candidate"}, {"status": "accepted"}]}
        self.write_json(self.data / "datasets" / self.pid / "D001" / "manifest.json", dataset)
        self.write_json(self.data / "runs" / self.pid / "R001" / "run.json", run)
        self.write_json(self.data / "models" / self.pid / "M001" / "model.json", model)
        self.write_json(self.data / "model-exports" / self.pid / "E001" / "record.json", export)
        self.write_json(self.data / "predictions" / f"{candidate_id}.json", prediction)
        self.write_json(self.data / "labelme" / self.pid / "recovery.json", {})
        self.write_json(self.data / "cvat" / "project-links.json", {self.pid: {"task_id": 7, "project_id": 8}})
        self.write_json(self.data / "cvat" / f"sync-{self.pid}.json", {"remote": "hash"})
        self.write_json(self.data / "cvat" / "annotation-backups" / "task-7-before.json", {})

        workspace = TrainingWorkspace(self.data, self.store, python_executable=sys.executable)
        project_root = self.store.directory(self.pid)
        for relative in ("datasets/D001/manifest.json", "runs/R001/run.json", "models/M001/model.json",
                         "model-exports/E001/record.json", f"predictions/{candidate_id}.json",
                         "integrations/labelme/recovery.json", "integrations/cvat/project-link.json",
                         "integrations/cvat/sync.json", "integrations/cvat/annotation-backups/task-7-before.json"):
            self.assertTrue((project_root / relative).is_file(), relative)
        self.assertFalse((self.data / "datasets" / self.pid).exists())
        self.assertNotIn(self.pid, json.loads((self.data / "cvat" / "project-links.json").read_text()))

        catalog = self.store.artifact_catalog(self.pid)
        self.assertEqual([row["id"] for row in catalog["dataset_versions"]], ["D001"])
        self.assertEqual(catalog["training_runs"][0]["dataset_version_id"], "D001")
        self.assertEqual(catalog["model_versions"][0]["run_id"], "R001")
        self.assertEqual(catalog["prediction_candidates"][0]["candidate_count"], 1)
        self.assertTrue(self.store.database_integrity(self.pid)["ok"])

        self.store.update_project(self.pid, name="renamed")
        self.assertTrue((self.store.directory(self.pid) / "models" / "M001" / "model.json").is_file())
        workspace.close()

    def test_new_dataset_reuses_project_image_and_is_sql_indexed(self):
        self.store.update_project(self.pid, classes=["part"])
        records = []
        for index, split in enumerate(("train", "val", "test")):
            path = self.root / f"{index}.png"
            Image.new("RGB", (8, 8), (index * 40, 20, 30)).save(path)
            records.append({"path": str(path), "review_state": "approved", "split": split,
                            "batch_id": f"batch-{index}",
                            "shapes": [{"type": "rectangle", "label": "part", "x": 1, "y": 1,
                                        "width": 4, "height": 4}]})
        self.store.add_assets(self.pid, records)
        workspace = TrainingWorkspace(self.data, self.store, python_executable=sys.executable)
        created = workspace.create_dataset_version(self.pid)
        manifest_path = workspace.datasets_dir(self.pid) / created["id"] / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        source = Path(self.store.snapshot(self.pid)["assets"][0]["image_path"])
        copy = manifest_path.parent / manifest["assets"][0]["image_file"]
        self.assertEqual(source.read_bytes(), copy.read_bytes())
        if os.name == "nt":
            self.assertTrue(os.path.samefile(source, copy))
        catalog = self.store.artifact_catalog(self.pid)
        self.assertEqual(catalog["dataset_versions"][0]["asset_count"], 3)
        self.assertEqual(catalog["dataset_versions"][0]["relative_path"], "datasets/D001/manifest.json")
        self.assertTrue(self.store.database_integrity(self.pid)["ok"])
        workspace.close()


if __name__ == "__main__":
    unittest.main()

