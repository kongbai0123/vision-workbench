"""Project container migration, SQL lineage, and immutable image reuse."""
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest

from PIL import Image

from workbench.project_layout import project_area, project_reference
from workbench.project_storage import ProjectStorage
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
        for relative in ("artifacts/datasets/D001/manifest.json", "artifacts/runs/R001/run.json",
                         "artifacts/models/M001/model.json", "outputs/models/E001/record.json",
                         f"artifacts/predictions/{candidate_id}.json",
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
        self.assertTrue((self.store.directory(self.pid) / "artifacts/models/M001/model.json").is_file())
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
        self.assertEqual(catalog["dataset_versions"][0]["relative_path"], "artifacts/datasets/D001/manifest.json")
        self.assertTrue(self.store.database_integrity(self.pid)["ok"])
        workspace.close()

    def test_migration_preserves_verified_dataset_copy_if_project_image_is_corrupt(self):
        project_root = self.store.directory(self.pid)
        good_image = b"verified dataset image bytes"
        digest = sha256(good_image).hexdigest()
        project_image = project_area(project_root, "images") / f"{digest}.png"
        project_image.write_bytes(b"corrupt source image bytes")
        dataset = project_area(project_root, "datasets") / "D001"
        self.write_json(dataset / "manifest.json", {
            "assets": [{"image_file": "images/retained.png", "sha256": digest}],
        })
        dataset_image = dataset / "images" / "retained.png"
        dataset_image.parent.mkdir()
        dataset_image.write_bytes(good_image)

        report, = ProjectStorage(self.data, self.store).migrate()

        self.assertEqual(report["deduplicated_files"], 0)
        self.assertIn({"path": str(project_image), "reason": "source_sha256_mismatch"}, report["conflicts"])
        self.assertEqual(dataset_image.read_bytes(), good_image)
        self.assertFalse(os.path.samefile(project_image, dataset_image))

    def test_default_dataset_export_is_owned_by_project_and_survives_rename(self):
        from workbench.server import WorkbenchService

        service = WorkbenchService(self.data)
        try:
            self.store.update_project(self.pid, classes=["part"])
            image = self.root / "approved.png"
            Image.new("RGB", (16, 16), (100, 20, 30)).save(image)
            self.store.add_assets(self.pid, [{"path": str(image), "review_state": "approved", "split": "train",
                "shapes": [{"type": "rectangle", "label": "part", "x": 1, "y": 1, "width": 6, "height": 6}]}])

            def export(payload):
                job = service.pipeline_job(self.pid, "export", payload)
                for _ in range(150):
                    current = service.jobs.get(job["id"])
                    if current["state"] in {"succeeded", "failed"}:
                        return current
                    time.sleep(.02)
                self.fail("Export job did not finish")

            result = export({"format": "native", "version": "project-v1"})
            self.assertEqual(result["state"], "succeeded", result)
            archive = Path(result["result"]["zip_path"])
            self.assertEqual(archive.parent, self.store.directory(self.pid) / "outputs/datasets")
            self.assertTrue(archive.is_file())

            other = self.store.create_project("other")["id"]
            rejected = export({"output_dir": str(self.store.directory(other) / "outputs/datasets")})
            self.assertEqual(rejected["state"], "failed")
            self.assertFalse((self.store.directory(other) / "outputs/datasets").exists())
            self.store.update_project(self.pid, name="portable-renamed")
            current = self.store.get_project(self.pid)["exports"][0]
            self.assertEqual(Path(current["zip_path"]).parent, self.store.directory(self.pid) / "outputs/datasets")
            self.assertTrue(Path(current["zip_path"]).is_file())
        finally:
            service.close()

    def test_migration_moves_flat_project_areas_and_preserves_existing_references(self):
        root = self.store.directory(self.pid)
        image = self.root / "import.png"
        Image.new("RGB", (16, 16), (50, 60, 70)).save(image)
        imported = self.store.add_assets(self.pid, [{"path": str(image)}])
        original = Path(self.store.get_asset(self.pid, imported["asset_ids"][0], internal=True)["image_path"])
        original.parent.rename(root / "images")

        legacy_files = {
            "recordings/clip.mp4": b"recording bytes",
            "imported-annotations/source.json": b"original annotation",
            "datasets/D001/manifest.json": b'{"assets":[],"source":{"video":"recordings/clip.mp4"}}',
            "runs/R001/metrics.jsonl": b"run metrics",
            "models/M001/checkpoint.pt": b"weights",
            "predictions/candidate.json": b"prediction",
            "exports/v1.zip": b"dataset output",
            "model-exports/E001/model.zip": b"model output",
            "scratch/recording-notes": b"work file",
        }
        for relative, payload in legacy_files.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)

        storage = ProjectStorage(self.data, self.store)
        report, = storage.migrate()
        self.assertEqual(report["conflicts"], [])
        self.assertEqual(len(report["moved"]), 10)
        for relative, payload in legacy_files.items():
            self.assertFalse((root / relative).exists())
            self.assertEqual(project_reference(root, relative).read_bytes(), payload)
        self.assertEqual(project_reference(root, "recordings/clip.mp4"), root / "inputs/videos/clip.mp4")
        self.assertEqual(project_reference(root, "exports/v1.zip"), root / "outputs/datasets/v1.zip")
        self.assertEqual(Path(self.store.get_asset(self.pid, imported["asset_ids"][0], internal=True)["image_path"]),
                         original)
        self.assertEqual(storage.migrate()[0]["moved"], [])

    def test_layout_conflict_keeps_both_existing_files(self):
        root = self.store.directory(self.pid)
        old = root / "recordings/clip.mp4"
        new = root / "inputs/videos/clip.mp4"
        for path, payload in ((old, b"original"), (new, b"separate copy")):
            path.parent.mkdir(parents=True)
            path.write_bytes(payload)

        report, = ProjectStorage(self.data, self.store).migrate()

        self.assertEqual(report["conflicts"][0]["reason"], "destination_exists")
        self.assertEqual(project_reference(root, "recordings/clip.mp4").read_bytes(), b"original")
        self.assertEqual(project_reference(root, "inputs/videos/clip.mp4").read_bytes(), b"separate copy")


if __name__ == "__main__":
    unittest.main()
