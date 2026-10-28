"""Dependency-free preflight checks for the Windows desktop runtime.

This script does not create an application, contact the network, or inspect user
projects. Native imports run in child processes so a blocked DLL cannot crash
the launcher itself.
"""
from __future__ import annotations

import argparse
from importlib import metadata
import json
from pathlib import Path
import re
import struct
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORE_IMPORTS = (
    "PySide6.QtCore", "PySide6.QtWidgets", "PySide6.QtWebEngineWidgets",
    "PySide6.QtWebChannel", "numpy", "PIL.Image", "yaml",
)
OPTIONAL_IMPORTS = ("cv2", "mss")
IMPORT_TIMEOUT = 45
PROBE_MARKER = "WORKBENCH_IMPORT_RESULT="
PIN_PATTERN = re.compile(r"([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s;]+)")
REPAIR_HINT = "Run powershell -NoProfile -ExecutionPolicy Bypass -File .\\bootstrap.ps1 to repair the runtime."


def _safe_detail(value: str, limit: int = 2000) -> str:
    """Keep useful error text without including URL credentials or session tokens."""
    value = re.sub(r"(https?://)[^\s/@]+:[^\s/@]+@", r"\1[redacted]@", str(value))
    value = re.sub(r"([?&](?:token|session|key|api_key)=)[^\s&#]+", r"\1[redacted]", value, flags=re.I)
    value = re.sub(r"\b(?:github_pat_|gh[pousr]_|sk-)[A-Za-z0-9_-]+", "[redacted]", value)
    return value.strip()[-limit:]


def _read_pins(path: Path) -> tuple[list[tuple[str, str]], list[str]]:
    pins: list[tuple[str, str]] = []
    issues: list[str] = []
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeError) as exc:
        return [], [f"Cannot read requirements: {_safe_detail(str(exc))}"]
    for number, raw in enumerate(lines, 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        match = PIN_PATTERN.fullmatch(line)
        if not match:
            issues.append(f"Unsupported requirement on line {number}; expected an exact package==version pin.")
            continue
        pins.append(match.groups())
    if not pins and not issues:
        issues.append("The requirements file contains no pinned dependencies.")
    return pins, issues


def _probe_imports(modules: tuple[str, ...]) -> dict:
    code = (
        "import importlib,json,sys\n"
        "result={'ok':True,'modules':[]}\n"
        f"for name in {modules!r}:\n"
        " try:\n"
        "  importlib.import_module(name)\n"
        "  result['modules'].append({'name':name,'ok':True})\n"
        " except Exception as error:\n"
        "  result['ok']=False\n"
        "  result['modules'].append({'name':name,'ok':False,'error':type(error).__name__+': '+str(error)})\n"
        f"print({PROBE_MARKER!r}+json.dumps(result))\n"
        "sys.exit(0 if result['ok'] else 1)\n"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-I", "-B", "-c", code],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=IMPORT_TIMEOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"Import check timed out after {IMPORT_TIMEOUT} seconds.", "modules": []}
    except OSError as exc:
        return {"ok": False, "error": f"Could not start import check: {_safe_detail(str(exc))}", "modules": []}
    payload = None
    for line in result.stdout.splitlines():
        if line.startswith(PROBE_MARKER):
            try:
                payload = json.loads(line[len(PROBE_MARKER):])
            except (ValueError, TypeError):
                continue
    if not isinstance(payload, dict) or not isinstance(payload.get("modules"), list):
        detail = _safe_detail(result.stderr or result.stdout)
        return {
            "ok": False,
            "error": f"Import check exited without a result (exit {result.returncode})." + (f" {detail}" if detail else ""),
            "modules": [],
        }
    for item in payload["modules"]:
        if "error" in item:
            item["error"] = _safe_detail(item["error"])
    payload["ok"] = payload.get("ok") is True and result.returncode == 0
    if result.returncode and not any(item.get("ok") is False for item in payload["modules"]):
        payload["error"] = f"Import check failed after loading modules (exit {result.returncode})."
    return payload


def _probe_errors(probe: dict) -> list[str]:
    errors = [f"{item['name']}: {item.get('error', 'import failed')}"
              for item in probe.get("modules", []) if item.get("ok") is False]
    if probe.get("error"):
        errors.append(probe["error"])
    return errors or ["Native import check failed."]


def check_runtime(project_root=None, probe_imports=True, requirements_path=None) -> dict:
    """Return an actionable report without requiring any third-party package."""
    root = Path(project_root or PROJECT_ROOT).resolve()
    requirements = Path(requirements_path).resolve() if requirements_path else root / "requirements.txt"
    bits = struct.calcsize("P") * 8
    issues: list[str] = []
    warnings: list[str] = []
    report = {
        "ok": False,
        "issues": issues,
        "warnings": warnings,
        "python": {"executable": sys.executable, "version": ".".join(map(str, sys.version_info[:3])), "bits": bits},
        "platform": sys.platform,
        "requirements": str(requirements),
        "packages": [],
        "import_probe": {"status": "skipped"},
        "repair_command": REPAIR_HINT,
    }
    if not ((3, 12) <= sys.version_info[:2] < (3, 15)):
        issues.append("Python 3.12-3.14 is required; Python 3.13 is recommended. Rebuild the environment with bootstrap.ps1.")
    if sys.platform != "win32":
        issues.append("The desktop launcher requires Windows; this platform is unsupported.")
    if bits != 64:
        issues.append("A 64-bit Python installation is required for the desktop runtime.")
    pins, pin_issues = _read_pins(requirements)
    issues.extend(pin_issues)
    for name, expected in pins:
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            installed = None
        except (OSError, ValueError) as exc:
            installed = None
            issues.append(f"Cannot inspect {name}: {_safe_detail(str(exc))}")
        status = "missing" if installed is None else "ok" if installed == expected else "mismatch"
        report["packages"].append({"name": name, "expected": expected, "installed": installed, "status": status})
        if status == "missing":
            issues.append(f"Missing dependency: {name}=={expected}.")
        elif status == "mismatch":
            issues.append(f"Dependency version mismatch: {name} requires {expected}, found {installed}.")
    if probe_imports and not issues:
        core = _probe_imports(CORE_IMPORTS)
        report["import_probe"] = {"status": "ok" if core["ok"] else "failed", "core": core, "optional": []}
        if not core["ok"]:
            issues.extend(_probe_errors(core))
        else:
            for name in OPTIONAL_IMPORTS:
                optional = _probe_imports((name,))
                report["import_probe"]["optional"].append({"name": name, **optional})
                if not optional["ok"]:
                    warnings.extend(f"Optional feature unavailable: {error}" for error in _probe_errors(optional))
    elif issues:
        report["import_probe"]["reason"] = "Resolve platform, Python, or dependency issues before checking native imports."
    else:
        report["import_probe"]["reason"] = "Import probing was disabled."
    report["ok"] = not issues
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Check the Vision Workbench desktop runtime without installing anything.")
    parser.add_argument("--json", action="store_true", help="Print one JSON report")
    parser.add_argument("--report", type=Path, help="Write the JSON report to this file")
    parser.add_argument("--skip-imports", action="store_true", help="Only check platform, Python and installed versions")
    parser.add_argument("--requirements", type=Path, help="Check another exact-pinned requirements file")
    args = parser.parse_args(argv)
    report = check_runtime(probe_imports=not args.skip_imports, requirements_path=args.requirements)
    if args.report:
        try:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            report["issues"].append(f"Cannot write diagnostic report: {_safe_detail(str(exc))}")
            report["ok"] = False
    if args.json:
        # Windows PowerShell 5.1 decodes native stdout using a legacy code page.
        # ASCII JSON preserves Unicode paths regardless of that console setting.
        print(json.dumps(report, ensure_ascii=True))
    else:
        print("Vision Workbench runtime: " + ("READY" if report["ok"] else "NOT READY"))
        print(f"Python {report['python']['version']} ({report['python']['bits']}-bit): {report['python']['executable']}")
        for issue in report["issues"]:
            print("ERROR: " + issue)
        for warning in report["warnings"]:
            print("WARNING: " + warning)
        if not report["ok"]:
            print(REPAIR_HINT)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
