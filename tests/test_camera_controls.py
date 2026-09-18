import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from workbench.camera_controls import CameraControls, validate_values
from workbench.camera_profiles import CameraProfiles


class Driver:
    def __init__(self):
        self.value, self.flags, self.reject = -6, 1, False
        self.writes = []

    def GetRange(self, prop):
        if prop != 4:
            raise ValueError('unsupported')
        return -12, 0, 2, -6, 3

    def Get(self, prop):
        return self.value, self.flags

    def Set(self, prop, value, flags):
        self.writes.append((prop, value, flags))
        if not self.reject:
            self.value, self.flags = value, flags


class CameraControlTests(unittest.TestCase):
    def setUp(self):
        self.driver = Driver()
        self.controls = CameraControls.__new__(CameraControls)
        self.controls.interfaces = {'camera': self.driver}

    def test_only_supported_parameters_and_native_ranges(self):
        controls = self.controls.read()
        self.assertEqual(list(controls), ['exposure'])
        self.assertEqual(controls['exposure']['min'], -12)
        self.assertTrue(controls['exposure']['auto'])

    def test_apply_manual_and_verify_readback(self):
        result = self.controls.apply({'exposure': {'value': -8, 'auto': False}})
        self.assertEqual(result['exposure']['value'], -8)
        self.assertFalse(result['exposure']['auto'])
        self.driver.reject = True
        with self.assertRaisesRegex(ValueError, '回讀值'):
            self.controls.apply({'exposure': {'value': -4, 'auto': False}})

    def test_validate_entire_batch_before_write(self):
        for values in ({'exposure': {'value': -7, 'auto': False}},
                       {'exposure': {'value': -8, 'auto': False}, 'gain': {'value': 5, 'auto': False}}):
            with self.assertRaises(ValueError):
                self.controls.apply(values)
        self.assertEqual(self.driver.writes, [])

    def test_reset_uses_driver_defaults_and_auto_support(self):
        self.controls.apply({'exposure': {'value': -8, 'auto': False}})
        self.controls.reset()
        self.assertEqual((self.driver.value, self.driver.flags), (-6, 1))

    def test_invalid_payloads(self):
        for value in (None, [], {'unknown': {}}, {'gain': {'value': float('nan'), 'auto': False}},
                      {'gain': {'value': 1, 'auto': 1}}, {'gain': {'value': True, 'auto': False}},
                      {'gain': {'value': 2**80, 'auto': True}}):
            with self.assertRaises(ValueError):
                validate_values(value)

    def test_fps_reset_cannot_override_requested_pixel_format(self):
        import cv2
        from workbench.acquisition import CameraService

        class Capture:
            def __init__(self):
                self.pixel = None

            def isOpened(self):
                return True

            def set(self, key, value):
                if key == cv2.CAP_PROP_FPS:
                    self.pixel = cv2.VideoWriter_fourcc(*'YUY2')
                if key == cv2.CAP_PROP_FOURCC:
                    self.pixel = value
                return True

        capture = Capture()
        with tempfile.TemporaryDirectory() as folder, patch.object(cv2, 'VideoCapture', return_value=capture):
            service = CameraService(Path(folder))
            result = service._open_capture(dict(index=0, width=1280, height=720, fps=30, pixel_format='MJPG'))
            self.assertIs(result, capture)
            self.assertEqual(capture.pixel, cv2.VideoWriter_fourcc(*'MJPG'))

    def test_profiles_persist_per_device_and_reject_invalid_before_write(self):
        with tempfile.TemporaryDirectory() as folder:
            profiles = CameraProfiles(folder)
            settings = dict(width=1280, height=720, fps=30, pixel_format='MJPG',
                            controls={'exposure': {'value': -8, 'auto': False}}, preview_fps=15)
            profiles.save('Camera A [0]', '固定光源', settings)
            self.assertEqual(CameraProfiles(folder).list('Camera A [0]')['固定光源'], settings)
            self.assertEqual(profiles.list('Camera B [1]'), {})
            before = profiles.path.read_bytes()
            with self.assertRaises(ValueError):
                profiles.save('Camera A [0]', 'broken', {**settings, 'fps': float('nan')})
            self.assertEqual(profiles.path.read_bytes(), before)
            self.assertFalse(Path(folder, 'camera-profiles.tmp').exists())


if __name__ == '__main__':
    unittest.main()
