"""Windows launcher integration checks without pip, GUI, or model requests."""
import base64
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
POWERSHELL = shutil.which("powershell.exe") if os.name == "nt" else None


def ps_literal(value):
    return "'" + str(value).replace("'", "''") + "'"


@unittest.skipUnless(os.name == "nt" and POWERSHELL, "Windows PowerShell 5.1 integration")
class StartupLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="workbench startup with spaces ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "project folder"
        (self.root / "scripts").mkdir(parents=True)
        for name in ("launch.ps1", "bootstrap.ps1", "vision-workbench.bat"):
            shutil.copyfile(ROOT / name, self.root / name)
        shutil.copyfile(ROOT / "scripts" / "runtime.ps1", self.root / "scripts" / "runtime.ps1")

    def powershell(self, body, timeout=30):
        source = (
            "$ErrorActionPreference = 'Stop'\n"
            "[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)\n"
            "$OutputEncoding = [Console]::OutputEncoding\n"
            + body
        )
        encoded = base64.b64encode(source.encode("utf-16-le")).decode("ascii")
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        return subprocess.run(
            [POWERSHELL, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-EncodedCommand", encoded],
            cwd=self.root, env=env, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout,
        )

    def run_launcher(self, *arguments):
        return self.powershell(
            "& " + ps_literal(self.root / "launch.ps1") + " " + " ".join(arguments) + "\nexit $LASTEXITCODE"
        )

    def setup_tripwire(self, exit_code=1):
        (self.root / "bootstrap.ps1").write_text(
            "$counter = Join-Path $PSScriptRoot 'setup-count.txt'\n"
            "$count = if (Test-Path -LiteralPath $counter) { [int](Get-Content -LiteralPath $counter) } else { 0 }\n"
            "Set-Content -LiteralPath $counter -Value ($count + 1)\n"
            f"exit {exit_code}\n", encoding="utf-8"
        )

    def use_unhealthy_runtime_stub(self):
        (self.root / "scripts" / "runtime.ps1").write_text(
            "function Get-WorkbenchRuntimeReport {\n"
            " param([string]$ProjectRoot)\n"
            " [pscustomobject]@{ok=$false;issues=@('fixture runtime missing');warnings=@();python=$null;packages=@()}\n"
            "}\n", encoding="utf-8"
        )

    def use_healthy_runtime_stub(self):
        with (self.root / "scripts" / "runtime.ps1").open("a", encoding="utf-8") as source:
            source.write(
                "\nfunction Get-WorkbenchRuntimeReport {\n"
                " param([string]$ProjectRoot)\n"
                " [pscustomobject]@{ok=$true;issues=@();warnings=@();python=$null;packages=@()}\n"
                "}\n"
            )

    def test_diagnose_missing_runtime_writes_report_without_running_setup(self):
        self.setup_tripwire()
        result = self.run_launcher("-Diagnose")
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse((self.root / "setup-count.txt").exists())
        path = self.root / "data" / "logs" / "startup-check.json"
        self.assertTrue(path.exists(), result.stdout + result.stderr)
        report = json.loads(path.read_text(encoding="utf-8-sig"))
        self.assertFalse(report["ok"])
        self.assertIn("runtime is missing", report["issues"][0])
        self.assertEqual(len(report["download_markers"]), 4)
        self.assertIn("Runtime is not ready", result.stdout)

    def test_no_auto_repair_leaves_setup_untouched_and_reports_failure(self):
        self.setup_tripwire()
        result = self.run_launcher("-NoAutoRepair")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.root / "setup-count.txt").exists())
        self.assertIn("Runtime is not ready", result.stdout)
        self.assertTrue((self.root / "data" / "logs" / "startup-check.json").exists())

    def test_automatic_setup_failure_runs_once_and_stops_visibly(self):
        self.use_unhealthy_runtime_stub()
        self.setup_tripwire(exit_code=1)
        result = self.run_launcher()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual((self.root / "setup-count.txt").read_text().strip(), "1")
        self.assertIn("Automatic setup failed", result.stdout)
        self.assertIn("Logs:", result.stdout)

    def test_setup_success_with_unhealthy_runtime_does_not_repeat_or_launch_gui(self):
        self.use_unhealthy_runtime_stub()
        self.setup_tripwire(exit_code=0)
        result = self.run_launcher()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual((self.root / "setup-count.txt").read_text().strip(), "1")
        self.assertIn("Runtime is not ready", result.stdout)
        self.assertNotIn("Starting Vision Workbench", result.stdout)

    def test_explicit_repair_failure_runs_setup_once(self):
        self.setup_tripwire(exit_code=1)
        result = self.run_launcher("-Repair")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual((self.root / "setup-count.txt").read_text().strip(), "1")
        self.assertIn("Setup failed", result.stdout)

    def test_bootstrap_initialization_failure_saves_transcript_before_any_pip(self):
        (self.root / "scripts" / "runtime.ps1").write_text(
            "function Initialize-WorkbenchEnvironment {\n"
            " param([string]$ProjectRoot,[string]$Name='.venv')\n"
            " throw 'fixture initialization failed before pip'\n"
            "}\n", encoding="utf-8"
        )
        result = self.powershell("& " + ps_literal(self.root / "bootstrap.ps1") + "\nexit $LASTEXITCODE")
        self.assertNotEqual(result.returncode, 0)
        logs = list((self.root / "data" / "logs").glob("bootstrap-*.log"))
        self.assertEqual(len(logs), 1, result.stdout + result.stderr)
        self.assertIn("fixture initialization failed before pip", logs[0].read_text(encoding="utf-8-sig"))
        self.assertIn("Diagnostic log:", result.stdout)

    def test_bootstrap_transcript_includes_native_stdout_and_stderr(self):
        command = self.root / "fake installer.cmd"
        command.write_text(
            "@echo off\r\necho FIXTURE_NATIVE_STDOUT\r\necho FIXTURE_NATIVE_STDERR 1>&2\r\nexit /b 19\r\n",
            encoding="ascii",
        )
        with (self.root / "scripts" / "runtime.ps1").open("a", encoding="utf-8") as source:
            source.write(
                "\nfunction Initialize-WorkbenchEnvironment {\n"
                " param([string]$ProjectRoot,[string]$Name='.venv')\n"
                " return " + ps_literal(command) + "\n}\n"
            )
        result = self.powershell("& " + ps_literal(self.root / "bootstrap.ps1") + "\nexit $LASTEXITCODE")
        self.assertNotEqual(result.returncode, 0)
        logs = list((self.root / "data" / "logs").glob("bootstrap-*.log"))
        self.assertEqual(len(logs), 1, result.stdout + result.stderr)
        transcript = logs[0].read_text(encoding="utf-8-sig")
        self.assertIn("FIXTURE_NATIVE_STDOUT", transcript)
        self.assertIn("FIXTURE_NATIVE_STDERR", transcript)
        self.assertIn("Dependency installation failed", transcript)

    def test_concurrent_setup_lock_stops_before_environment_or_pip_changes(self):
        logs = self.root / "data" / "logs"
        logs.mkdir(parents=True)
        called = self.root / "initialization-called.txt"
        with (self.root / "scripts" / "runtime.ps1").open("a", encoding="utf-8") as source:
            source.write(
                "\nfunction Initialize-WorkbenchEnvironment {\n"
                " param([string]$ProjectRoot,[string]$Name='.venv')\n"
                " Set-Content -LiteralPath " + ps_literal(called) + " -Value 'unexpected'\n"
                " throw 'Fixture must not initialize an environment while another setup owns the lock'\n}\n"
            )
        result = self.powershell(
            "$heldLock = [System.IO.File]::Open(" + ps_literal(logs / "setup.lock") + ",[System.IO.FileMode]::OpenOrCreate,[System.IO.FileAccess]::ReadWrite,[System.IO.FileShare]::None)\n"
            "try {\n"
            " & " + ps_literal(self.root / "bootstrap.ps1") + "\n"
            " $setupExit = $LASTEXITCODE\n"
            "} finally { $heldLock.Dispose() }\n"
            "exit $setupExit\n"
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Another setup is running", result.stdout)
        self.assertFalse(called.exists())
        self.assertFalse((self.root / ".venv").exists())

    def test_relative_data_root_follows_powershell_location_not_dotnet_cwd(self):
        self.setup_tripwire()
        location = Path(self.temp.name) / "different working folder"
        location.mkdir()
        relative = "relative 資料 folder"
        result = self.powershell(
            "Set-Location -LiteralPath " + ps_literal(location) + "\n"
            "if ([Environment]::CurrentDirectory -eq (Get-Location).Path) { throw 'Fixture must use differing process and PowerShell locations' }\n"
            "& " + ps_literal(self.root / "launch.ps1") + " -Diagnose -DataRoot " + ps_literal(relative) + "\nexit $LASTEXITCODE"
        )
        self.assertNotEqual(result.returncode, 0)
        expected = location / relative / "logs" / "startup-check.json"
        self.assertTrue(expected.exists(), result.stdout + result.stderr)
        self.assertFalse((self.root / relative).exists())
        self.assertFalse((self.root / "setup-count.txt").exists())

    def test_normal_startup_preserves_unicode_data_root_with_trailing_backslash(self):
        python = self.compatible_python()
        create = subprocess.run(
            [python, "-B", "-m", "venv", "--without-pip", str(self.root / ".venv")],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30,
        )
        self.assertEqual(create.returncode, 0, create.stdout + create.stderr)
        self.use_healthy_runtime_stub()
        (self.root / "main.py").write_text(
            "import json,sys\nfrom pathlib import Path\n"
            "Path(__file__).with_name('captured-argv.json').write_text(json.dumps(sys.argv[1:]),encoding='utf-8')\n",
            encoding="utf-8",
        )
        data_root = str(Path(self.temp.name) / "影像 資料夾") + "\\"
        result = self.run_launcher("-NoAutoRepair", "-DataRoot", ps_literal(data_root))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        argv = json.loads((self.root / "captured-argv.json").read_text(encoding="utf-8"))
        self.assertEqual(argv, ["--data-root", data_root])
        self.assertTrue((Path(data_root) / "logs" / "startup-check.json").exists())

    def test_windows_argument_quoting_roundtrips_empty_quotes_spaces_and_backslashes(self):
        python = self.compatible_python()
        arguments = ["", "plain", "with spaces", "trailing path\\", 'embedded"quote', 'slash\\"quote', "非 ASCII 路徑\\"]
        output = self.root / "arguments.json"
        code = "import json,sys;print(json.dumps(sys.argv[1:],ensure_ascii=True))"
        values = "@(" + ",".join(ps_literal(arg) for arg in arguments) + ")"
        body = (
            ". " + ps_literal(self.root / "scripts" / "runtime.ps1") + "\n"
            "$values = " + values + "\n"
            "$arguments = @('-B','-c'," + ps_literal(code) + ") + $values\n"
            "$quoted = ($arguments | ForEach-Object { ConvertTo-WorkbenchArgument $_ }) -join ' '\n"
            "$native = Start-Process -FilePath " + ps_literal(python) + " -ArgumentList $quoted -Wait -PassThru -WindowStyle Hidden -RedirectStandardOutput " + ps_literal(output) + "\n"
            "exit $native.ExitCode\n"
        )
        result = self.powershell(body)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(json.loads(output.read_text(encoding="utf-8")), arguments)

    def compatible_python(self):
        candidates = [sys.executable]
        local = os.environ.get("LOCALAPPDATA")
        if local:
            candidates.append(str(Path(local) / "Programs" / "Python" / "Python313" / "python.exe"))
        for candidate in candidates:
            if not Path(candidate).is_file():
                continue
            result = subprocess.run(
                [candidate, "-B", "-c", "import sys,struct;sys.exit(0 if sys.version_info[:2]==(3,13) and struct.calcsize('P')==8 else 1)"],
                capture_output=True, timeout=15,
            )
            if result.returncode == 0:
                return candidate
        self.skipTest("A local 64-bit Python 3.13 is required; no installer is run by this test")

    def test_invalid_environment_is_preserved_and_healthy_environment_is_reused(self):
        python = self.compatible_python()
        old = self.root / ".venv"
        old.mkdir()
        (old / "sentinel.txt").write_text("preserve this environment", encoding="utf-8")
        body = (
            ". " + ps_literal(self.root / "scripts" / "runtime.ps1") + "\n"
            "function Find-WorkbenchPython { [pscustomobject]@{ Command = " + ps_literal(python) + "; Prefix = @() } }\n"
            "$first = Initialize-WorkbenchEnvironment -ProjectRoot " + ps_literal(self.root) + "\n"
            "if (-not (Test-WorkbenchPython -Command $first)) { throw 'Invalid new environment' }\n"
            "Set-Content -LiteralPath (Join-Path " + ps_literal(old) + " 'healthy-sentinel.txt') -Value 'retained'\n"
            "$second = Initialize-WorkbenchEnvironment -ProjectRoot " + ps_literal(self.root) + "\n"
            "if ($first -ne $second) { throw 'Healthy runtime was unexpectedly replaced' }\n"
            "Write-Output 'ENVIRONMENT_REUSED'\n"
        )
        result = self.powershell(body, timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        backups = list(self.root.glob(".venv-backup-*"))
        self.assertEqual(len(backups), 1, result.stdout + result.stderr)
        self.assertEqual((backups[0] / "sentinel.txt").read_text(), "preserve this environment")
        self.assertTrue((old / "Scripts" / "python.exe").is_file())
        self.assertEqual((old / "healthy-sentinel.txt").read_text().strip(), "retained")
        self.assertIn("ENVIRONMENT_REUSED", result.stdout)

    def test_unexpected_environment_name_is_rejected_without_moving_files(self):
        sentinel = self.root / "untouched.txt"
        sentinel.write_text("unchanged", encoding="utf-8")
        result = self.powershell(
            ". " + ps_literal(self.root / "scripts" / "runtime.ps1") + "\n"
            "Initialize-WorkbenchEnvironment -ProjectRoot " + ps_literal(self.root) + " -Name '..\\outside'\n"
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Unexpected environment name", result.stderr)
        self.assertEqual(sentinel.read_text(), "unchanged")
        self.assertEqual(list(self.root.glob(".venv-backup-*")), [])

    def test_linked_invalid_environment_is_rejected_without_moving_target(self):
        python = self.compatible_python()
        target = Path(self.temp.name) / "external environment"
        target.mkdir()
        (target / "sentinel.txt").write_text("external remains", encoding="utf-8")
        link = self.root / ".venv"
        result = self.powershell(
            ". " + ps_literal(self.root / "scripts" / "runtime.ps1") + "\n"
            "function Find-WorkbenchPython { [pscustomobject]@{ Command = " + ps_literal(python) + "; Prefix = @() } }\n"
            "New-Item -ItemType Junction -Path " + ps_literal(link) + " -Target " + ps_literal(target) + " | Out-Null\n"
            "Initialize-WorkbenchEnvironment -ProjectRoot " + ps_literal(self.root) + "\n"
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("environment is a link", result.stderr)
        self.assertEqual((target / "sentinel.txt").read_text(), "external remains")
        self.assertTrue(link.is_dir())
        self.assertEqual(list(self.root.glob(".venv-backup-*")), [])


if __name__ == "__main__":
    unittest.main()
