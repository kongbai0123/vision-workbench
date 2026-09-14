"""Real image/codec and simulated camera lifecycle checks (no user hardware)."""
from pathlib import Path
from types import SimpleNamespace
import io
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image, ImageOps

from workbench import acquisition as a
from sam2_segmentation import Sam2RawOutput
from sam2_segmentation.training_dataset import decode_coco_uncompressed_rle


def wait_until(predicate, timeout=4):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.01)
    raise AssertionError("Timed out waiting for worker")


class FakeCamera:
    def __init__(self, size=(96, 64), disconnect_after=None):
        self.size = size
        self.disconnect_after = disconnect_after
        self.calls = []
        self.frames = 0
        self.released = False

    def get(self, _prop):
        self.calls.append(threading.current_thread().name)
        return 20.

    def read(self):
        self.calls.append(threading.current_thread().name)
        time.sleep(.02)
        self.frames += 1
        if self.disconnect_after and self.frames >= self.disconnect_after:
            return False, None
        width, height = self.size
        frame = np.zeros((height, width, 3), np.uint8)
        frame[:, :, 0] = self.frames % 255
        frame[10:30, 15:45] = (25, 125, 245)
        return True, frame

    def release(self):
        self.calls.append(threading.current_thread().name)
        self.released = True


class AcquisitionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "測試資料"
        self.root.mkdir()

    def tearDown(self):
        a.close_ai()
        self.temp.cleanup()

    def image(self, name="工件.png"):
        frame = np.full((100, 140, 3), 245, np.uint8)
        cv2.rectangle(frame, (25, 20), (115, 80), (10, 35, 75), -1)
        cv2.circle(frame, (70, 50), 12, (245, 245, 245), -1)
        path = self.root / name
        a._save_png(path, frame)
        return path, frame

    def test_preview_roi_supports_box_polygon_and_mask_without_mutating_source(self):
        frame = np.full((24, 32, 3), 180, np.uint8)
        original = frame.copy()
        box = {"type":"rectangle", "x":4, "y":5, "width":10, "height":8, "label":"part"}
        result = a.process_frame(frame, "binary", settings={"threshold":100, "roi_shape":box})
        self.assertTrue(np.all(result[5:13, 4:14] == 255))
        self.assertEqual(int(result[:4].max()), 0)
        polygon = {"type":"polygon", "points":[[2,2],[20,2],[2,20]], "label":"part"}
        result = a.process_frame(frame, "canny", settings={"roi_shape":polygon})
        self.assertEqual(result.shape, frame.shape[:2])
        mask = np.zeros((24,32), np.uint8);mask[7:12,9:16] = 1
        flat = mask.T.reshape(-1);counts=[];active=0;count=0
        for value in flat:
            if int(value)==active:count+=1
            else:counts.append(count);count=1;active=int(value)
        counts.append(count)
        decoded = a._roi_mask(frame,{"type":"mask","counts":counts,"label":"part"})
        self.assertTrue(np.all(decoded[7:12,9:16] == 255))
        self.assertEqual(int(decoded[:6].max()),0)
        np.testing.assert_array_equal(frame, original)

    def test_import_and_constructor_do_not_open_hardware(self):
        with patch.object(cv2, "VideoCapture", side_effect=AssertionError("camera opened")):
            camera = a.CameraService(self.root)
            self.assertEqual(camera.status()["state"], "stopped")
            self.assertFalse(camera.status()["has_frame"])
            camera.close()

    def test_exif_rotated_jpeg_preview_keeps_raw_coordinates_and_source_bytes(self):
        path = self.root / "方向6_原圖.jpg"
        pixels = np.zeros((40, 80, 3), np.uint8)
        pixels[:, :40] = (220, 15, 20)
        pixels[:, 40:] = (10, 20, 220)
        pixels[:10, :15] = (20, 230, 20)
        original = Image.fromarray(pixels)
        exif = Image.Exif()
        exif[274] = 6
        original.save(path, format="JPEG", exif=exif, quality=95)
        original_bytes = path.read_bytes()
        with Image.open(path) as stored:
            self.assertEqual(stored.size, (80, 40))
            self.assertEqual(ImageOps.exif_transpose(stored).size, (40, 80))
        response, mime = a.preview_image(path)
        self.assertEqual(mime, "image/png")
        with Image.open(io.BytesIO(response)) as preview:
            self.assertEqual(preview.size, (80, 40))
            self.assertFalse(preview.getexif())
            preview_rgb = np.asarray(preview).copy()
        inference_bgr = a.read_image(path)
        self.assertEqual(inference_bgr.shape, (40, 80, 3))
        np.testing.assert_allclose(preview_rgb[:, :, ::-1], inference_bgr, atol=2)
        self.assertEqual(path.read_bytes(), original_bytes)
        with Image.open(path) as still_original:
            self.assertEqual(still_original.getexif()[274], 6)

    def test_unoriented_jpeg_and_other_preview_formats_keep_exact_original_bytes(self):
        path, _frame = self.image()
        raw = path.read_bytes()
        self.assertEqual(a.preview_image(path), (raw, "image/png"))
        for orientation in (None, 1):
            jpeg = self.root / f"normal_{orientation}.jpg"
            with Image.open(path) as source:
                if orientation is None:
                    source.save(jpeg, format="JPEG")
                else:
                    exif = Image.Exif()
                    exif[274] = orientation
                    source.save(jpeg, format="JPEG", exif=exif)
            self.assertEqual(a.preview_image(jpeg), (jpeg.read_bytes(), "image/jpeg"))

    def test_unicode_image_and_exact_hole_rle_candidate(self):
        path, _ = self.image()
        result = a.segment_image(path, engine="grabcut", box=[15, 10, 125, 90],
                                 points=[[35, 35]], negative_points=[[70, 50]], label="工件")
        shape = result["shape"]
        mask = decode_coco_uncompressed_rle({"size": [100, 140], "counts": shape["counts"]})
        self.assertEqual(mask[35, 35], 255)
        self.assertEqual(mask[50, 70], 0)
        self.assertEqual(mask[0, 0], 0)
        self.assertEqual(sum(shape["counts"]), 100 * 140)
        self.assertEqual(shape["metadata"]["review_state"], "pending")
        self.assertEqual(shape["label"], "工件")
        self.assertEqual((shape["x"], shape["y"], shape["width"], shape["height"]), (0, 0, 140, 100))
        self.assertFalse(result["diagnostics"]["generated_center_prompt"])

    def test_grabcut_box_does_not_force_a_hole_center_to_foreground(self):
        path, _ = self.image()
        result = a.segment_image(path, "grabcut", box=[15, 10, 125, 90])
        mask = decode_coco_uncompressed_rle({"size": [100, 140], "counts": result["shape"]["counts"]})
        self.assertEqual(mask[50, 70], 0)
        self.assertGreater(np.count_nonzero(mask), 4000)

    def test_prompt_validation_and_cancel(self):
        path, _ = self.image()
        invalid = [dict(points=[[200, 2]]), dict(points=[[20, 20]], negative_points=[[20, 20]]),
                   dict(box=[30, 30, 20, 40]), dict(box=[-1, 0, 20, 20])]
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                a.segment_image(path, "grabcut", **kwargs)
        cancelled = threading.Event()
        cancelled.set()
        with self.assertRaises(a.AcquisitionCancelled):
            a.segment_image(path, "grabcut", cancel_event=cancelled)

    def test_transform_modes_are_explicit_and_do_not_mutate_source(self):
        _, frame = self.image()
        before = frame.copy()
        for mode in ("original", "binary", "adaptive", "canny", "mog2", "flow"):
            output = a.transform_frame(frame, mode, state={})
            self.assertEqual(output.shape[:2], frame.shape[:2])
        np.testing.assert_array_equal(before, frame)
        binary = a.transform_frame(frame, "binary")
        np.testing.assert_array_equal(255 - binary, a.transform_frame(frame, "binary", invert=True))
        with self.assertRaises(ValueError):
            a.transform_frame(frame, "unknown")

    def test_classical_background_core_produces_prompted_full_size_mask(self):
        from classical_segmentation import BackgroundCalibrator
        path, frame = self.image()
        calibrator = BackgroundCalibrator()
        background = np.full_like(frame, 245)
        for _ in range(15):
            calibrator.add_frame(background)
        state = {"background_model": calibrator.build()}
        result = a.process_frame(frame, "classical", state=state,
                                 settings={"points": [[35, 35]], "negative_points": [[70, 50]]})
        self.assertEqual(result.shape, frame.shape[:2])
        self.assertEqual(result[35, 35], 255)
        self.assertEqual(result[50, 70], 0)
        self.assertEqual(result[0, 0], 0)

    def test_processing_preview_does_not_change_raw_snapshot(self):
        camera = a.CameraService(self.root)
        device = FakeCamera()
        with patch.object(camera, "_open_capture", return_value=device):
            camera.start()
            wait_until(lambda: camera.status()["has_frame"])
            camera.set_processing("binary", {"threshold": 100})
            wait_until(lambda: camera.status()["has_processed_frame"])
            processed = cv2.imdecode(np.frombuffer(camera.frame_jpeg(processed=True), np.uint8), cv2.IMREAD_GRAYSCALE)
            raw = a.read_image(camera.snapshot()["path"])
            self.assertEqual(processed.shape, raw.shape[:2])
            self.assertTrue(np.any(raw[:, :, 0] != raw[:, :, 1]))
            camera.set_processing("classical")
            wait_until(lambda: camera.status()["processing_state"] == "needs_background")
            self.assertIn("15", camera.status()["processing_error"])
            fallback = cv2.imdecode(np.frombuffer(camera.frame_jpeg(processed=True), np.uint8), cv2.IMREAD_COLOR)
            self.assertIsNotNone(fallback)
            self.assertEqual(fallback.shape[:2], raw.shape[:2])
            camera.stop()

    def test_slow_processing_does_not_block_camera_capture_or_shutdown(self):
        camera = a.CameraService(self.root)
        entered, release = threading.Event(), threading.Event()
        device = FakeCamera()
        def slow_processing(frame, mode, **kwargs):
            entered.set()
            release.wait(3)
            return np.zeros(frame.shape[:2], np.uint8)
        with patch.object(camera, "_open_capture", return_value=device), patch.object(a, "process_frame", side_effect=slow_processing):
            camera.start()
            wait_until(lambda: camera.status()["has_frame"])
            camera.set_processing("sam2")
            self.assertTrue(entered.wait(2))
            current_count = camera.status()["frame_count"]
            wait_until(lambda: camera.status()["frame_count"] >= current_count + 3)
            self.assertTrue(Path(camera.snapshot()["path"]).is_file())
            camera.stop()
            self.assertEqual(camera.status()["state"], "stopped")
            release.set()
        self.assertTrue(device.released)

    def test_sam2_fake_runtime_uses_local_only_and_true_box_prompt(self):
        path, _ = self.image()
        model = self.root / "模型"
        model.mkdir()
        (model / "config.json").write_text("{}")
        mask = np.zeros((100, 140), bool)
        mask[20:80, 25:115] = True
        mask[40:60, 60:80] = False
        observed = {}

        class Runtime:
            device_name, dtype_name = "cpu", "float32"
            def __init__(self, config):
                observed["config"] = config
            def infer(self, frame, **kwargs):
                observed["kwargs"] = kwargs
                return Sam2RawOutput(mask[None], np.array([.98]))
            def close(self):
                pass

        with patch("sam2_segmentation.HuggingFaceSam2Runtime", Runtime):
            result = a.segment_image(path, "sam2", box=[15, 10, 125, 90], model_dir=model)
            point_result = a.segment_image(path, "sam2", points=[[35, 35]], negative_points=[[70, 50]], model_dir=model)
        self.assertTrue(observed["config"].local_files_only)
        self.assertTrue(observed["config"].allow_cpu_fallback)
        self.assertIsNone(observed["kwargs"]["input_boxes"])
        self.assertEqual(observed["kwargs"]["input_points"], [[[[35, 35], [70, 50]]]])
        self.assertEqual(point_result["shape"]["metadata"]["points"], [[35, 35]])
        decoded = decode_coco_uncompressed_rle({"size": [100, 140], "counts": result["shape"]["counts"]})
        np.testing.assert_array_equal(decoded != 0, mask)

    def test_sam2_missing_model_never_falls_back_to_grabcut(self):
        path, _ = self.image()
        with patch.object(a, "_grabcut", side_effect=AssertionError("silent fallback")):
            with self.assertRaisesRegex(a.AcquisitionError, "config.json"):
                a.segment_image(path, "sam2", points=[[35, 35]], model_dir=self.root / "missing")

    def test_camera_worker_snapshot_and_real_mp4_round_trip(self):
        device = FakeCamera()
        camera = a.CameraService(self.root)
        with patch.object(camera, "_open_capture", return_value=device):
            camera.start(width=96, height=64, fps=20)
            wait_until(lambda: camera.status()["has_frame"])
            snapshot = camera.snapshot()
            self.assertEqual(a.read_image(snapshot["path"]).shape, (64, 96, 3))
            self.assertEqual(snapshot["batch_id"], camera.status()["batch_id"])
            jpeg = cv2.imdecode(np.frombuffer(camera.frame_jpeg(), np.uint8), cv2.IMREAD_COLOR)
            self.assertEqual(jpeg.shape, (64, 96, 3))
            camera.start_recording("project-1")
            wait_until(lambda: camera.status()["record_frames"] >= 6)
            stopped = camera.stop_recording()
            self.assertFalse(stopped["recording"])
            video = stopped["last_recording"]
            self.assertEqual(video["project_id"], "project-1")
            self.assertGreater(video["frames"], 0)
            self.assertTrue(Path(video["path"]).is_file())
            records = a.extract_video(video["path"], self.root, .1)
            self.assertGreaterEqual(len(records), 2)
            self.assertEqual(len({record["batch_id"] for record in records}), 1)
            self.assertEqual(records[0]["source"]["frame_index"], 0)
            camera.stop()
        self.assertTrue(device.released)
        self.assertEqual(set(device.calls), {"WorkbenchCamera"})

    def test_camera_close_finalizes_recording_and_odd_size_is_disclosed(self):
        camera = a.CameraService(self.root)
        device = FakeCamera(size=(95, 65))
        with patch.object(camera, "_open_capture", return_value=device):
            camera.start(width=95, height=65)
            wait_until(lambda: camera.status()["has_frame"])
            camera.start_recording("project-2")
            wait_until(lambda: camera.status()["record_frames"] >= 2)
            camera.close()
        video = camera.status()["last_recording"]
        self.assertEqual((video["width"], video["height"]), (95, 65))
        self.assertEqual((video["encoded_width"], video["encoded_height"]), (96, 66))
        self.assertTrue(Path(video["path"]).is_file())
        with self.assertRaises(a.AcquisitionError):
            camera.start()

    def test_disconnect_exposes_error_and_stale_snapshot_is_not_returned(self):
        camera = a.CameraService(self.root)
        device = FakeCamera(disconnect_after=2)
        with patch.object(camera, "_open_capture", return_value=device):
            camera.start()
            wait_until(lambda: camera.status()["state"] == "error")
        self.assertIn("中斷", camera.status()["error"])
        with self.assertRaises(a.AcquisitionError):
            camera.snapshot()
        self.assertTrue(device.released)
        camera.close()

    def test_camera_open_failure_and_invalid_settings(self):
        camera = a.CameraService(self.root)
        with self.assertRaises(ValueError):
            camera.start(index=-1)
        with patch.object(camera, "_open_capture", side_effect=a.AcquisitionError("no camera")):
            camera.start()
            wait_until(lambda: camera.status()["state"] == "error")
        self.assertEqual(camera.status()["error"], "no camera")
        camera.close()

    def test_video_sampling_intervals_unicode_and_cancellation_cleanup(self):
        video = self.root / "測試影片.mp4"
        writer = cv2.VideoWriter(a._opencv_path(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (48, 32))
        self.assertTrue(writer.isOpened())
        for index in range(25):
            writer.write(np.full((32, 48, 3), index * 5, np.uint8))
        writer.release()
        output = a.extract_video(video, self.root, interval_seconds=.7)
        from hashlib import sha256
        self.assertEqual({item["source"]["video_sha256"] for item in output}, {sha256(video.read_bytes()).hexdigest()})
        self.assertEqual([item["source"]["frame_index"] for item in output], [0, 7, 14, 21])
        event = threading.Event()
        event.set()
        with self.assertRaises(a.AcquisitionCancelled):
            a.extract_video(video, self.root, cancel_event=event)
        self.assertTrue(video.is_file())
        self.assertEqual(len(list((self.root / "incoming").iterdir())), 1)

    def test_screen_capture_uses_explicit_mss_request_only(self):
        class Screen:
            monitors = [{"left": -100, "top": 0, "width": 48, "height": 32}]
            def __enter__(self): return self
            def __exit__(self, *_args): pass
            def grab(self, monitor): return np.zeros((32, 48, 4), np.uint8)
        with patch.dict("sys.modules", {"mss": SimpleNamespace(mss=Screen)}):
            record = a.capture_screen(self.root)
        self.assertEqual(a.read_image(record["path"]).shape, (32, 48, 3))
        self.assertEqual(record["source"]["monitor"]["left"], -100)


if __name__ == "__main__":
    unittest.main()
