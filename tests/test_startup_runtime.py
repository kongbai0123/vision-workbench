"""Preflight regressions need no desktop, model, camera, or installed package."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_runtime.py"
SPEC = importlib.util.spec_from_file_location("workbench_startup_check", SCRIPT)
runtime = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runtime)


class StartupRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "requirements.txt").write_text("PySide6==6.11.2\nnumpy==2.5.3\n", encoding="utf-8")
        self.platform = patch.object(runtime.sys, "platform", "win32")
        self.version = patch.object(runtime.sys, "version_info", (3, 13, 5))
        self.bits = patch.object(runtime.struct, "calcsize", return_value=8)
        self.platform.start()
        self.version.start()
        self.bits.start()
        self.addCleanup(self.platform.stop)
        self.addCleanup(self.version.stop)
        self.addCleanup(self.bits.stop)

    @staticmethod
    def installed(name):
        return {"PySide6": "6.11.2", "numpy": "2.5.3"}[name]

    def test_healthy_pins_are_ready_when_imports_are_explicitly_skipped(self):
        with patch.object(runtime.metadata, "version", side_effect=self.installed):
            report = runtime.check_runtime(self.root, probe_imports=False)
        self.assertTrue(report["ok"])
        self.assertEqual(report["issues"], [])
        self.assertEqual(report["packages"][0]["status"], "ok")
        self.assertEqual(report["import_probe"]["status"], "skipped")

    def test_unsupported_python_skips_native_imports(self):
        with patch.object(runtime.sys, "version_info", (3, 11, 9)), patch.object(
            runtime.metadata, "version", side_effect=self.installed
        ), patch.object(runtime, "_probe_imports") as probe:
            report = runtime.check_runtime(self.root)
        self.assertFalse(report["ok"])
        self.assertIn("Python 3.12", report["issues"][0])
        self.assertIn("bootstrap.ps1", report["issues"][0])
        probe.assert_not_called()

    def test_missing_and_wrong_versions_are_actionable_and_skip_probing(self):
        def lookup(name):
            if name == "numpy":
                raise runtime.metadata.PackageNotFoundError(name)
            return "6.10.0"
        with patch.object(runtime.metadata, "version", side_effect=lookup), patch.object(runtime, "_probe_imports") as probe:
            report = runtime.check_runtime(self.root)
        self.assertFalse(report["ok"])
        self.assertEqual([p["status"] for p in report["packages"]], ["mismatch", "missing"])
        self.assertTrue(any("numpy==2.5.3" in issue for issue in report["issues"]))
        probe.assert_not_called()

    def test_non_windows_and_32_bit_are_explicit_blockers(self):
        with patch.object(runtime.sys, "platform", "linux"), patch.object(runtime.struct, "calcsize", return_value=4), patch.object(
            runtime.metadata, "version", side_effect=self.installed
        ):
            report = runtime.check_runtime(self.root, False)
        self.assertFalse(report["ok"])
        self.assertTrue(any("Windows" in issue for issue in report["issues"]))
        self.assertTrue(any("64-bit" in issue for issue in report["issues"]))

    def test_required_native_import_failure_blocks_startup(self):
        failed = {"ok": False, "modules": [{"name": "PySide6.QtCore", "ok": False, "error": "ImportError: DLL load failed"}]}
        with patch.object(runtime.metadata, "version", side_effect=self.installed), patch.object(runtime, "_probe_imports", return_value=failed) as probe:
            report = runtime.check_runtime(self.root)
        self.assertFalse(report["ok"])
        self.assertIn("DLL load failed", report["issues"][0])
        self.assertEqual(probe.call_count, 1)

    def test_blocked_optional_opencv_does_not_prevent_core_startup(self):
        probes = [
            {"ok": True, "modules": []},
            {"ok": False, "modules": [{"name": "cv2", "ok": False, "error": "ImportError: blocked DLL"}]},
            {"ok": True, "modules": []},
        ]
        with patch.object(runtime.metadata, "version", side_effect=self.installed), patch.object(runtime, "_probe_imports", side_effect=probes) as probe:
            report = runtime.check_runtime(self.root)
        self.assertTrue(report["ok"])
        self.assertEqual(report["issues"], [])
        self.assertIn("cv2", report["warnings"][0])
        self.assertEqual(probe.call_count, 3)

    def test_import_timeout_is_reported_without_hanging(self):
        with patch.object(runtime.subprocess, "run", side_effect=subprocess.TimeoutExpired("python", 45)):
            result = runtime._probe_imports(("PySide6.QtCore",))
        self.assertFalse(result["ok"])
        self.assertIn("timed out", result["error"])

    def test_native_crash_is_reported_without_requiring_a_json_result(self):
        crashed = subprocess.CompletedProcess([], -1073741515, stdout="", stderr="DLL could not load")
        with patch.object(runtime.subprocess, "run", return_value=crashed):
            result = runtime._probe_imports(("PySide6.QtCore",))
        self.assertFalse(result["ok"])
        self.assertIn("-1073741515", result["error"])
        self.assertIn("DLL could not load", result["error"])

    def test_real_import_probe_returns_structured_success_and_import_error(self):
        successful = runtime._probe_imports(("json",))
        self.assertTrue(successful["ok"], successful)
        self.assertEqual(successful["modules"], [{"name": "json", "ok": True}])
        failed = runtime._probe_imports(("workbench_nonexistent_import_fixture",))
        self.assertFalse(failed["ok"])
        self.assertIn("ModuleNotFoundError", failed["modules"][0]["error"])

    def test_probe_parses_last_marker_and_redacts_sensitive_error_text(self):
        payload = {"ok": False, "modules": [{"name": "numpy", "ok": False, "error": "https://alice:password@example.test/?session=secret-value"}]}
        output = "library warning\n" + runtime.PROBE_MARKER + json.dumps(payload) + "\n"
        with patch.object(runtime.subprocess, "run", return_value=subprocess.CompletedProcess([], 1, stdout=output, stderr="")):
            result = runtime._probe_imports(("numpy",))
        self.assertFalse(result["ok"])
        self.assertNotIn("password", str(result))
        self.assertNotIn("secret-value", str(result))

    def test_missing_or_malformed_requirements_fail_clearly(self):
        missing = self.root / "missing.txt"
        report = runtime.check_runtime(self.root, False, missing)
        self.assertFalse(report["ok"])
        self.assertIn("Cannot read requirements", report["issues"][0])
        missing.write_text("numpy>=2\n", encoding="utf-8")
        report = runtime.check_runtime(self.root, False, missing)
        self.assertIn("exact package==version", report["issues"][0])
        missing.write_bytes(b"\xffnot-utf8")
        report = runtime.check_runtime(self.root, False, missing)
        self.assertIn("Cannot read requirements", report["issues"][0])

    def test_cli_json_and_saved_report_match_and_use_failure_exit_code(self):
        path = self.root / "診斷記錄" / "runtime.json"
        requirements = self.root / "執行依賴.txt"
        requirements.write_text("workbench-uninstalled-test-fixture==1.0\n", encoding="utf-8")
        result = subprocess.run(
            [sys.executable, "-B", str(SCRIPT), "--json", "--skip-imports", "--requirements", str(requirements), "--report", str(path)],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=15,
        )
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertTrue(result.stdout.isascii(), result.stdout)
        self.assertIn("\\u", result.stdout)
        report = json.loads(result.stdout)
        self.assertEqual(report, json.loads(path.read_text(encoding="utf-8")))
        self.assertEqual(report["requirements"], str(requirements.resolve()))
        self.assertFalse(report["ok"])
        self.assertEqual(report["packages"][0]["status"], "missing")

    def test_actual_desktop_import_preserves_optional_opencv_fallback(self):
        root = SCRIPT.parents[1]
        desktop_python = root / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        candidate = str(desktop_python) if desktop_python.is_file() else sys.executable
        availability = subprocess.run(
            [candidate, "-I", "-B", "-c", "import importlib.util,sys;sys.exit(0 if all(importlib.util.find_spec(n) for n in ['PySide6','numpy','PIL','yaml']) else 1)"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=15,
        )
        if availability.returncode:
            self.skipTest("Installed core desktop dependencies are needed for the isolated optional-OpenCV regression")
        code = (
            "import importlib.abc,sys,tempfile\n"
            f"sys.path.insert(0,{str(root)!r})\n"
            "class BlockOpenCV(importlib.abc.MetaPathFinder):\n"
            " def find_spec(self,fullname,path=None,target=None):\n"
            "  if fullname=='cv2' or fullname.startswith('cv2.'):\n"
            "   raise ImportError('Fixture: OpenCV blocked by Windows application control')\n"
            "sys.meta_path.insert(0,BlockOpenCV())\n"
            "import workbench.desktop\n"
            "from workbench.server import WorkbenchService\n"
            "with tempfile.TemporaryDirectory(prefix='workbench-no-opencv-') as folder:\n"
            " service=WorkbenchService(folder,port=0)\n"
            " try:\n"
            "  service.start()\n"
            " finally:\n"
            "  service.close()\n"
            "print('DESKTOP_IMPORT_WITHOUT_OPENCV_OK')\n"
        )
        result = subprocess.run(
            [candidate, "-I", "-B", "-c", code], cwd=root,
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=45,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("DESKTOP_IMPORT_WITHOUT_OPENCV_OK", result.stdout)


if __name__ == "__main__":
    unittest.main()
