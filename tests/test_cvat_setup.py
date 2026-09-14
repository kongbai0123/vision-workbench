import io
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from cvat_setup import CvatSetup, CVAT_VERSION, SetupPause


class CvatReadinessTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.setup = CvatSetup(self.tmp.name)
        self.setup._set = Mock()
        self.setup.stopping.wait = Mock()

    def test_about_success_still_waits_for_policy_activation(self):
        setup = self.setup
        setup._project_running = Mock(return_value=True)
        setup._compose = Mock(side_effect=[
            SimpleNamespace(returncode=0),  # compose up
            SimpleNamespace(returncode=1),  # bundle has not activated
            SimpleNamespace(returncode=0),
        ])
        client = Mock()
        client.open.return_value = io.BytesIO(
            ('{"version":"' + CVAT_VERSION.lstrip('v') + '"}').encode())
        with patch('cvat_setup.build_opener', return_value=client):
            setup._start_services()
        self.assertEqual(setup._compose.call_count, 3)
        self.assertIn('health?bundles=true', setup._compose.call_args.args[-1])
        setup.stopping.wait.assert_called_once_with(1)

    def test_authorization_timeout_does_not_mark_services_ready(self):
        self.setup._compose = Mock(return_value=SimpleNamespace(returncode=1))
        with patch('cvat_setup.time.monotonic', side_effect=[0, 1, 121]):
            with self.assertRaisesRegex(RuntimeError, '權限規則尚未就緒'):
                self.setup._wait_authorization()
        self.assertEqual(self.setup._compose.call_count, 1)

    def test_authorization_wait_honors_shutdown(self):
        self.setup.stopping.set()
        self.setup._compose = Mock()
        with self.assertRaises(SetupPause):
            self.setup._wait_authorization()
        self.setup._compose.assert_not_called()

    def test_temporary_probe_failure_is_retried(self):
        self.setup._compose = Mock(side_effect=[
            RuntimeError('probe timeout'), SimpleNamespace(returncode=0)])
        self.setup._wait_authorization()
        self.assertEqual(self.setup._compose.call_count, 2)


if __name__ == '__main__':
    unittest.main()
