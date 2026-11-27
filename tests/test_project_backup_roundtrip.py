"""Cross-host project archive verification through the ordinary Workbench API."""

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import time
import unittest
from unittest.mock import patch
from urllib.request import Request, urlopen
import zipfile

import cv2
import numpy as np
from PIL import Image

from workbench.acquisition import _opencv_path
from workbench.migrations import migrate
from workbench.model_registry import ModelRegistry
from workbench.project_backup import create_project_backup
from workbench.project_layout import project_area, project_reference
from workbench.server import WorkbenchService
from workbench.store import ProjectStore, dump
from workbench.training import TrainingWorkspace


CLASSES = ["mid_done", "grip", "grip_done"]
TRAINING_CONFIG = {"engine": "pixel_prototype_v1", "epochs": 3}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@contextmanager
def workbench(data_root):
    service = WorkbenchService(data_root).start()
    try:
        yield service
    finally:
        service.close()


class ProjectBackupRoundTripTests(unittest.TestCase):
    def call(self, service, route, payload=None, *, method=None):
        headers = {
            "Content-Type": "application/json",
            "X-Workbench": "1",
            "Origin": service.url,
            "X-Workbench-Token": service.api_token,
        }
        request = Request(
            service.url + route,
            data=None if payload is None else json.dumps(payload).encode(),
            method=method,
            headers=headers,
        )
        with urlopen(request, timeout=15) as response:
            return json.load(response)

    def wait(self, fetch, complete):
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            result = fetch()
            if complete(result):
                return result
            time.sleep(.04)
        self.fail("Workbench operation did not finish")

    def job(self, service, route, payload, *, success=True):
        job = self.call(service, route, payload)
        result = self.wait(
            lambda: self.call(service, f"/api/jobs/{job['id']}"),
            lambda row: row["state"] in {"succeeded", "failed", "cancelled"},
        )
        self.assertEqual(result["state"], "succeeded" if success else "failed", result)
        return result["result"] if success else result

    def training_images(self, directory):
        records = []
        for index in range(12):
            pixels = np.full((64, 80, 3), (20 + index, 25, 30), dtype=np.uint8)
            shapes = []
            for column, (label, color) in enumerate(zip(CLASSES, ((200, 55, 48), (38, 180, 60), (50, 75, 190)))):
                x, y = 4 + column * 25, 10 + index % 4
                pixels[y:y + 24, x:x + 18] = color
                shapes.append({"type": "rectangle", "label": label, "x": x, "y": y, "width": 18, "height": 24})
            path = directory / f"image-{index:02}.png"
            Image.fromarray(pixels).save(path)
            records.append({
                "path": str(path), "name": path.name, "shapes": shapes,
                "batch_id": f"capture-{index:02}",
                "source": {"kind": "camera", "station": "validation", "frame_index": index},
            })
        return records

    def write_video(self, path):
        writer = cv2.VideoWriter(_opencv_path(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (80, 64))
        try:
            self.assertTrue(writer.isOpened(), "MJPG fixture encoder unavailable")
            for index in range(12):
                pixels = np.full((64, 80, 3), index * 12, dtype=np.uint8)
                pixels[10:34, 6:24] = (48, 55, 200)
                writer.write(pixels)
        finally:
            writer.release()
        self.assertGreater(path.stat().st_size, 0)

    def prepare_project(self, service, directory):
        project = self.call(service, "/api/projects", {"name": "跨機握把 QA", "classes": CLASSES})
        pid = project["id"]
        route = f"/api/projects/{pid}"
        source_records = self.training_images(directory)
        imported = source_records[0]
        annotation = Path(imported["path"]).with_suffix(".json")
        annotation.write_text(json.dumps({
            "version": "5.7.0", "flags": {}, "imagePath": imported["name"],
            "imageWidth": 80, "imageHeight": 64, "imageData": None,
            "shapes": [{
                "label": shape["label"], "shape_type": "rectangle",
                "points": [[shape["x"], shape["y"]],
                           [shape["x"] + shape["width"], shape["y"] + shape["height"]]],
                "flags": {},
            } for shape in imported["shapes"]],
        }, ensure_ascii=False), encoding="utf-8")
        preview = self.job(service, f"{route}/import-preview", {"paths": [str(annotation)]})
        self.assertEqual(preview["items"][0]["shape_count"], len(CLASSES))
        confirmed = self.job(service, f"{route}/import-confirm", {
            "token": preview["token"], "selected": [preview["items"][0]["id"]],
        })
        added = service.store.add_assets(pid, source_records[1:])
        photo_ids = confirmed["asset_ids"] + added["asset_ids"]
        first = self.call(service, f"{route}/assets/{photo_ids[0]}")
        shapes = [dict(shape) for shape in first["shapes"]]
        shapes[0]["width"] -= 1
        edited = self.call(service, f"{route}/assets/{photo_ids[0]}",
                           {"revision": first["revision"], "shapes": shapes}, method="PUT")
        revisions = {aid: service.store.get_asset(pid, aid)["revision"] for aid in photo_ids}
        self.assertEqual(revisions[photo_ids[0]], edited["revision"])
        self.call(service, f"{route}/review", {
            "asset_ids": photo_ids, "state": "approved", "revisions": revisions,
            "reason": "", "note": "跨機驗證通過",
        })

        original_video = directory / "capture.avi"
        self.write_video(original_video)
        recording = self.job(service, f"{route}/recordings/import", {"path": str(original_video)})
        duplicate_video = directory / "copy-from-other-folder.avi"
        shutil.copyfile(original_video, duplicate_video)
        duplicate_recording = self.job(service, f"{route}/recordings/import", {"path": str(duplicate_video)})
        self.assertEqual(duplicate_recording["id"], recording["id"])
        video = self.job(service, f"{route}/video", {"video_id": recording["id"], "interval_seconds": .5})
        repeated_samples = self.job(service, f"{route}/video", {"video_id": recording["id"], "interval_seconds": .5})
        self.assertEqual((repeated_samples["added"], repeated_samples["duplicates"]), (0, 3))
        pending, rejected, trashed = video["asset_ids"]
        rejected_revision = service.store.get_asset(pid, rejected)["revision"]
        self.call(service, f"{route}/review", {
            "asset_ids": [rejected], "state": "rejected", "revisions": {rejected: rejected_revision},
            "reason": "模糊", "note": "保留未通過的審核紀錄",
        })
        self.call(service, f"{route}/review-trash", {
            "asset_ids": [trashed], "revisions": {trashed: service.store.get_asset(pid, trashed)["revision"]},
        })

        options = {"purpose": "formal", "strategy": "smart", "ratios": [60, 20, 20], "seed": 42}
        split = self.call(service, f"{route}/split-preview", {"options": options})
        self.call(service, f"{route}/split-apply", {
            "options": options, "revision": split["project_revision"], "fingerprint": split["fingerprint"],
        })
        dataset = self.call(service, f"{route}/dataset-versions", {})
        self.assertTrue(dataset["data_quality"]["independent_sources"])
        self.assertEqual(dataset["asset_count"], len(photo_ids))
        self.assertTrue(all(dataset["splits"][split] for split in ("train", "val", "test")))
        run = self.call(service, f"{route}/training-runs", {
            "dataset_version_id": dataset["id"], "config": TRAINING_CONFIG,
        })
        completed = self.wait(
            lambda: service.training.run(pid, run["run_id"]),
            lambda row: row["status"] in {"completed", "failed", "stopped"},
        )
        self.assertEqual(completed["status"], "completed", completed)
        self.wait(
            lambda: (pid, run["run_id"]) not in service.training.processes,
            bool,
        )
        self.assertTrue(completed["evaluation"]["test"])
        model_export = self.job(service, f"{route}/model-exports", {"model_version_id": run["model_version_id"]})
        data_export = self.job(service, f"{route}/export", {"format": "native", "version": "roundtrip-v1"})
        prediction = self.job(service, f"{route}/predictions", {
            "model_version_id": run["model_version_id"], "asset_ids": [photo_ids[0]],
        })
        self.assertEqual(prediction["assets"][0]["status"], "candidate", prediction)
        return {
            "pid": pid, "photo_ids": photo_ids, "frame_ids": [pending, rejected, trashed],
            "movie_sha256": digest(original_video), "dataset": dataset, "run": completed,
            "model_export": model_export, "data_export": data_export, "prediction": prediction,
            "video_id": recording["id"], "annotation_sha256": digest(annotation),
        }

    def verify_materials(self, service, fixture):
        route = f"/api/projects/{fixture['pid']}"
        catalog = self.call(service, f"{route}/materials")["materials"]
        by_id = {row["id"]: row for row in catalog}
        self.assertEqual(len(by_id), len(catalog))
        kinds = {
            kind: self.call(service, f"{route}/materials?kind={kind}")["materials"]
            for kind in ("image", "video", "annotation")
        }
        for kind, rows in kinds.items():
            self.assertEqual(rows, [row for row in catalog if row["kind"] == kind])
        self.assertEqual(len(kinds["video"]), 1)
        video = by_id[fixture["video_id"]]
        self.assertEqual(video["kind"], "video")
        self.assertEqual(video["sha256"], fixture["movie_sha256"])
        self.assertTrue(video["relative_path"].startswith("inputs/videos/"))
        self.assertTrue(all(row["relative_path"].startswith("inputs/images/") for row in kinds["image"]))
        annotation, = kinds["annotation"]
        self.assertTrue(annotation["relative_path"].startswith("inputs/annotations/"))
        self.assertEqual(annotation["sha256"], fixture["annotation_sha256"])
        self.assertIn(fixture["photo_ids"][0],
                      [link["image_id"] for link in annotation["metadata"]["linked_images"]])
        stored_annotation = service.store.directory(fixture["pid"]) / annotation["relative_path"]
        self.assertEqual(digest(stored_annotation), fixture["annotation_sha256"])
        for asset_id, frame_index in zip(fixture["frame_ids"][:2], (0, 5), strict=True):
            asset = self.call(service, f"{route}/assets/{asset_id}")
            self.assertEqual(asset["source"]["video_id"], video["id"])
            relation, = by_id[asset_id]["metadata"]["video_sources"]
            self.assertEqual(relation["video_id"], video["id"])
            self.assertEqual(relation["frame_index"], frame_index)
            self.assertAlmostEqual(relation["timestamp_seconds"], frame_index / 10)
            self.assertEqual(asset["source"]["timestamp_seconds"], relation["timestamp_seconds"])
            self.assertIn(asset["source"]["timestamp_method"], {"container", "frame_fps"})
            self.assertEqual(asset["source"]["timestamp_method"], relation["timestamp_method"])
        return catalog

    def project_catalog(self, service, fixture):
        pid = fixture["pid"]
        route = f"/api/projects/{pid}"
        project = self.call(service, route)
        asset_ids = fixture["photo_ids"] + fixture["frame_ids"]
        self.assertTrue(service.store.database_integrity(pid)["ok"])
        return {
            "revision": project["revision"], "classes": project["classes"], "stats": project["stats"],
            "split_plan": project["split_plan"],
            "assets": {aid: self.call(service, f"{route}/assets/{aid}")
                       for aid in fixture["photo_ids"] + fixture["frame_ids"][:2]},
            "history": {aid: self.call(service, f"{route}/assets/{aid}/history")["history"] for aid in asset_ids},
            "trash": self.call(service, f"{route}/review-trash-list", {}),
            "lineage": service.store.artifact_catalog(pid),
        }

    def file_fingerprints(self, service, fixture):
        root = service.store.directory(fixture["pid"])
        files = {
            path.relative_to(root).as_posix(): digest(path)
            for area in ("images", "recordings", "imported-annotations", "datasets", "models", "exports", "predictions")
            for path in project_area(root, area).rglob("*")
            if path.is_file() and not path.name.startswith(".")
        }
        run = project_area(root, "runs") / fixture["run"]["run_id"]
        for name in ("run.json", "metrics.jsonl", "evaluation.json", "events.jsonl", "artifact-manifest.json"):
            path = run / name
            files[path.relative_to(root).as_posix()] = digest(path)
        return files

    def write_modified_archive(self, original, target):
        with zipfile.ZipFile(original) as source, zipfile.ZipFile(target, "w") as output:
            image = next(info for info in source.infolist() if info.filename.endswith(".png") and "images/" in info.filename)
            for info in source.infolist():
                contents = source.read(info)
                if info is image:
                    contents = bytes([contents[0] ^ 1]) + contents[1:]
                output.writestr(info, contents)

    def prediction_geometry(self, prediction):
        return [{key: value for key, value in shape.items() if key != "id"}
                for shape in prediction["assets"][0]["shapes"]]

    def test_legacy_project_and_archive_keep_existing_video_provenance(self):
        with tempfile.TemporaryDirectory(prefix="workbench-legacy-materials-") as directory:
            base = Path(directory)
            source_data = base / "source" / "data"
            with patch("workbench.store.migrate", lambda db: migrate(db, through=6)):
                legacy_store = ProjectStore(source_data / "projects")
                project = legacy_store.create_project("舊版來源", classes=CLASSES)
            pid = project["id"]
            legacy_root = legacy_store.directory(pid)
            video = legacy_root / "recordings" / "recording_legacy.avi"
            video.parent.mkdir()
            self.write_video(video)
            video_digest = digest(video)
            capture_date = "2026-11-26T03:00:00+00:00"
            video.with_suffix(".json").write_text(dump({
                "project_id": pid, "extension": ".avi", "fps": 10,
                "frames": 12, "duration_seconds": 1.2, "width": 80, "height": 64,
                "display_name": "recording_legacy.avi", "started_at": capture_date,
            }), encoding="utf-8")
            frame = legacy_root / "images" / "frame_000000005.png"
            current_images = project_area(legacy_root, "images")
            if current_images != frame.parent and current_images.exists():
                current_images.rmdir()
            frame.parent.mkdir(exist_ok=True)
            Image.new("RGB", (80, 64), (25, 30, 35)).save(frame)
            asset_id = "e" * 32
            legacy_source = {
                "kind": "video", "video": str(video), "video_sha256": video_digest,
                "frame_index": 5, "timestamp_seconds": .5, "fps": 10,
            }
            with legacy_store.connection(pid, write=True) as db:
                db.execute(
                    "INSERT INTO assets VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (asset_id, frame.name, 80, 64, digest(frame), frame.name, "legacy-batch", "",
                     dump(legacy_source), 1, "pending", "[]", capture_date, capture_date),
                )
                self.assertEqual(db.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0], 6)
            old_backup = create_project_backup(legacy_store, pid, base / "transfer")

            def verify(service):
                route = f"/api/projects/{pid}"
                rows = self.call(service, f"{route}/materials")["materials"]
                video_row, = [row for row in rows if row["kind"] == "video"]
                image_row, = [row for row in rows if row["kind"] == "image"]
                self.assertEqual(image_row["id"], asset_id)
                self.assertEqual(video_row["relative_path"], "inputs/videos/recording_legacy.avi")
                self.assertEqual(video_row["sha256"], video_digest)
                relation, = image_row["metadata"]["video_sources"]
                self.assertEqual(relation["video_id"], video_row["id"])
                self.assertEqual(relation["frame_index"], 5)
                self.assertAlmostEqual(relation["timestamp_seconds"], .5)
                project = self.call(service, route)
                self.assertEqual(project["classes"], CLASSES)
                original_asset = self.call(service, f"{route}/assets/{asset_id}")
                self.assertEqual(original_asset["id"], asset_id)
                recording, = self.call(service, f"{route}/recordings")["recordings"]
                self.assertEqual(recording["id"], video_row["id"])
                self.assertTrue(Path(recording["path"]).is_relative_to(service.store.directory(pid) / "inputs" / "videos"))
                self.assertEqual(digest(Path(recording["path"])), video_digest)
                with service.store.connection(pid) as db:
                    self.assertGreaterEqual(db.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0], 7)
                    source, = db.execute("SELECT source FROM assets WHERE id = ?", (asset_id,)).fetchone()
                    self.assertEqual(json.loads(source), legacy_source)
                return video_row["id"]

            with workbench(source_data) as upgraded:
                video_id = verify(upgraded)
            with workbench(base / "destination" / "data") as destination:
                self.job(destination, "/api/project-restores", {"path": old_backup["path"]})
                self.assertEqual(verify(destination), video_id)
            print("LEGACY_MATERIALS_ROUNDTRIP_OK", json.dumps({
                "original_database_version": 6, "registered_video_id": video_id,
                "existing_project_upgraded": True, "old_zip_restored": True,
                "legacy_frame_timestamp_seconds": .5, "legacy_cls_order": CLASSES,
            }, ensure_ascii=True))

    def test_restore_after_removing_the_source_machine_data(self):
        unavailable = {"state": "not_installed", "message": "Optional runtime not needed for this fixture"}
        with (
            tempfile.TemporaryDirectory(prefix="workbench-backup-evidence-") as transfer_folder,
            patch.object(ModelRegistry, "_probe_torchvision", return_value=unavailable),
            patch.object(ModelRegistry, "_probe_python_component", return_value=unavailable),
            patch.object(TrainingWorkspace, "_probe_maskrcnn", return_value=False),
        ):
            transfer = Path(transfer_folder)
            with tempfile.TemporaryDirectory(prefix="workbench-source-host-") as source_folder:
                with workbench(Path(source_folder) / "data") as source:
                    fixture = self.prepare_project(source, Path(source_folder))
                    pid = fixture["pid"]
                    route = f"/api/projects/{pid}"
                    run_id = fixture["run"]["run_id"]
                    metrics = self.call(source, f"{route}/training-runs/{run_id}/metrics")["metrics"]
                    # Keep an older read transaction open to retain committed
                    # WAL pages containing the latest audit note during backup.
                    with source.store.connection(pid) as reader:
                        reader.execute("SELECT revision FROM project").fetchone()
                        rejected = fixture["frame_ids"][1]
                        revision = source.store.get_asset(pid, rejected)["revision"]
                        self.call(source, f"{route}/review", {
                            "asset_ids": [rejected], "state": "rejected", "revisions": {rejected: revision},
                            "reason": "模糊", "note": "備份前最後一筆已提交的審核備註",
                        })
                        wal = source.store.directory(pid) / "project.sqlite3-wal"
                        self.assertGreater(wal.stat().st_size, 32)
                        original_catalog = self.project_catalog(source, fixture)
                        original_materials = self.verify_materials(source, fixture)
                        original_files = self.file_fingerprints(source, fixture)
                        backup = self.job(source, f"{route}/backup", {})
                    portable = transfer / "project.zip"
                    shutil.copyfile(backup["path"], portable)
                    self.assertEqual(digest(portable), digest(Path(backup["path"])))
                    self.assertEqual(backup["project_id"], pid)
                    self.assertGreater(backup["files"], 20)

            self.assertFalse(Path(source_folder).exists())
            corrupt = transfer / "altered.zip"
            self.write_modified_archive(portable, corrupt)
            with workbench(transfer / "target-host" / "data") as target:
                restore_route = "/api/project-restores"
                with self.assertLogs(level="ERROR"):
                    altered = self.job(target, restore_route, {"path": str(corrupt)}, success=False)
                self.assertIn("SHA-256", altered["error"])
                self.assertEqual(self.call(target, "/api/projects")["projects"], [])
                restored = self.job(target, restore_route, {"path": str(portable)})
                self.assertEqual(restored["project_id"], pid)
                self.assertEqual(restored["files"], backup["files"])
                self.assertEqual(self.project_catalog(target, fixture), original_catalog)
                self.assertEqual(self.verify_materials(target, fixture), original_materials)
                self.assertEqual(self.file_fingerprints(target, fixture), original_files)
                self.assertEqual(self.call(target, f"{route}/training-runs/{run_id}/metrics")["metrics"], metrics)
                rejected = self.call(target, f"{route}/assets/{fixture['frame_ids'][1]}")
                self.assertEqual(rejected["source"]["review"]["note"], "備份前最後一筆已提交的審核備註")
                with self.assertLogs(level="ERROR"):
                    duplicate = self.job(target, restore_route, {"path": str(portable)}, success=False)
                self.assertIn("已存在", duplicate["error"])
                self.assertEqual(self.project_catalog(target, fixture), original_catalog)
                self.assertEqual(self.file_fingerprints(target, fixture), original_files)

                destination = target.store.directory(pid)
                recordings = self.call(target, f"{route}/recordings")["recordings"]
                self.assertEqual(len(recordings), 1)
                clip = Path(recordings[0]["path"])
                self.assertTrue(clip.is_relative_to(destination / "inputs" / "videos"))
                self.assertEqual(digest(clip), fixture["movie_sha256"])
                self.assertEqual(rejected["source"]["video_sha256"], fixture["movie_sha256"])
                self.assertEqual(project_reference(destination, rejected["source"]["video"]), clip)
                preflight = self.call(target, f"{route}/training-preflight", {
                    "dataset_version_id": fixture["dataset"]["id"], "config": TRAINING_CONFIG,
                })
                self.assertTrue(preflight["validated"], preflight)
                model_id = fixture["run"]["model_version_id"]
                model = target.training.model(pid, model_id)
                self.assertEqual(model["classes"], CLASSES)
                exported = target.training.model_export(pid, fixture["model_export"]["export_id"])
                self.assertTrue(Path(exported["path"]).is_relative_to(destination / "outputs" / "models"))
                self.assertEqual(digest(Path(exported["path"])), fixture["model_export"]["sha256"])
                exported_data = self.call(target, route)["exports"][0]
                self.assertTrue(Path(exported_data["zip_path"]).is_relative_to(destination / "outputs" / "datasets"))
                self.assertTrue(Path(exported_data["zip_path"]).is_file())
                prediction = self.job(target, f"{route}/predictions", {
                    "model_version_id": model_id, "asset_ids": [fixture["photo_ids"][0]],
                })
                self.assertEqual(self.prediction_geometry(prediction), self.prediction_geometry(fixture["prediction"]))
                new_export = self.job(target, f"{route}/model-exports", {"model_version_id": model_id})
                self.assertTrue(Path(new_export["path"]).is_file())
                self.assertNotEqual(new_export["export_id"], exported["export_id"])
                sampled = self.job(target, f"{route}/video", {"path": str(clip), "interval_seconds": .75})
                self.assertGreater(sampled["added"], 0)
                self.assertEqual(sampled["video_name"], clip.name)
                added_frame = self.call(target, f"{route}/assets/{sampled['asset_ids'][0]}")
                self.assertEqual(added_frame["source"]["video_id"], fixture["video_id"])
                all_videos = self.call(target, f"{route}/materials?kind=video")["materials"]
                self.assertEqual([row["id"] for row in all_videos], [fixture["video_id"]])
                print("PROJECT_BACKUP_ROUNDTRIP_OK", json.dumps({
                    "archive_files": backup["files"],
                    "archive_bytes": portable.stat().st_size,
                    "archive_sha256": digest(portable),
                    "original_images": len([path for path in original_files if path.startswith("inputs/images/")]),
                    "recordings": len(recordings),
                    "recording_sha256": fixture["movie_sha256"],
                    "material_catalog_count": len(original_materials),
                    "video_id": fixture["video_id"],
                    "annotation_source_sha256": fixture["annotation_sha256"],
                    "review_and_annotation_history": sum(len(rows) for rows in original_catalog["history"].values()),
                    "cls_order": CLASSES,
                    "dataset_id": fixture["dataset"]["id"], "dataset_splits": fixture["dataset"]["splits"],
                    "run_id": run_id, "model_id": model_id, "export_id": exported["export_id"],
                    "destination_reexport_id": new_export["export_id"],
                    "committed_wal_review_note_restored": True, "restored_video_sampling": sampled["added"],
                    "corruption_and_duplicate_rejected": True, "source_removed_before_restore": True,
                }, ensure_ascii=True))


if __name__ == "__main__":
    unittest.main()
