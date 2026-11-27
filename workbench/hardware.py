"""Read the local hardware inventory without importing a model runtime."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess


_WINDOWS_INVENTORY = """
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
@{
    cpu = (Get-CimInstance Win32_Processor -ErrorAction SilentlyContinue |
        Select-Object -First 1 -ExpandProperty Name)
    memory = (Get-CimInstance Win32_ComputerSystem -ErrorAction SilentlyContinue).TotalPhysicalMemory
    gpus = @(Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty Name)
} | ConvertTo-Json -Compress
"""


def _read_command(args, timeout):
    try:
        result = subprocess.run(
            args, stdin=subprocess.DEVNULL, capture_output=True, encoding="utf-8",
            errors="replace", timeout=timeout,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return result.stdout if result.returncode == 0 else ""


def _windows_summary(system32):
    powershell = system32 / "WindowsPowerShell" / "v1.0" / "powershell.exe"
    output = _read_command([powershell, "-NoProfile", "-NonInteractive", "-Command", _WINDOWS_INVENTORY], 6)
    try:
        summary = json.loads(output)
    except ValueError:
        return {}
    return summary if isinstance(summary, dict) else {}


def _system_memory():
    try:
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, ValueError):
        return None


def _nvidia_adapters(executable):
    if executable is None:
        return []
    output = _read_command(
        [executable, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"], 3,
    )
    devices = []
    for row in csv.reader(output.splitlines()):
        if len(row) != 2 or not row[0].strip():
            continue
        try:
            mebibytes = int(row[1])
        except ValueError:
            mebibytes = 0
        devices.append({
            "name": row[0].strip(),
            "memory_bytes": mebibytes * 1024**2 if mebibytes > 0 else None,
        })
    return devices


def hardware_inventory():
    windows = {}
    if os.name == "nt":
        system32 = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"
        windows = _windows_summary(system32)
        cpu_name = windows.get("cpu") or os.environ.get("PROCESSOR_IDENTIFIER", "")
        nvidia_paths = (
            system32 / "nvidia-smi.exe",
            Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "NVIDIA Corporation" / "NVSMI" / "nvidia-smi.exe",
        )
        executable = next((path for path in nvidia_paths if path.is_file()), None)
    else:
        executable = shutil.which("nvidia-smi")
        cpu_name = platform.processor()

    nvidia = _nvidia_adapters(executable)
    nvidia_names = {device["name"].casefold() for device in nvidia}
    adapters = [
        {"name": name.strip(), "memory_bytes": None}
        for name in (windows.get("gpus") or [])
        if isinstance(name, str) and name.strip() and name.strip().casefold() not in nvidia_names
    ]
    memory = windows.get("memory") or _system_memory()
    return {
        "cpu": {
            "name": cpu_name.strip() or None,
            "logical_cores": os.cpu_count(),
        },
        "memory": {"total_bytes": memory if memory and memory > 0 else None},
        "gpus": adapters + nvidia,
    }
