"""Detect local source changes and restart the desktop app safely."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time


SOURCE_SUFFIXES = {".py", ".js", ".mjs", ".css", ".html", ".json", ".ps1", ".bat"}
SOURCE_DIRECTORIES = ("workbench", "web", "composer_core", "classical_segmentation", "sam2_segmentation")
SOURCE_FILES = ("main.py", "cvat_setup.py", "setup_cvat.ps1", "bootstrap.ps1", "vision-workbench.bat",
                "requirements.txt", "requirements-ai.txt", "requirements-training.txt", "requirements-lock.txt")


def source_snapshot(root):
    root = Path(root).resolve()
    paths = [root/name for name in SOURCE_FILES]
    for directory in SOURCE_DIRECTORIES:
        folder = root/directory
        if folder.is_dir():
            paths.extend(path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in SOURCE_SUFFIXES)
    return {path.relative_to(root).as_posix():hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths if path.is_file()}


def changed_sources(before, after):
    return sorted(key for key in before.keys() | after.keys() if before.get(key) != after.get(key))


def validate_sources(root):
    root = Path(root).resolve()
    for name in source_snapshot(root):
        if name.endswith(".py"):
            path=root/name
            compile(path.read_bytes(),str(path),"exec")


def missing_runtime_requirements(root):
    """Allow updates after dependencies were already installed into this runtime."""
    from importlib.metadata import version,PackageNotFoundError
    missing=[]
    for line in (Path(root)/'requirements.txt').read_text('utf-8').splitlines():
        line=line.strip()
        if not line or line.startswith('#'):continue
        name,separator,required=line.partition('==')
        if not separator:
            missing.append(line);continue
        try:installed=version(name)
        except PackageNotFoundError:installed=None
        if installed!=required:missing.append(line)
    return missing


def schedule_restart(root):
    root=Path(root).resolve();logs=root/"data"/"logs";logs.mkdir(parents=True,exist_ok=True)
    with (logs/"update.log").open("ab") as log:
        return subprocess.Popen([sys.executable,str(Path(__file__).resolve()),"--wait",str(os.getpid()),str(root)],
            cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=log,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0)


def wait_and_restart(pid, root):
    if os.name=="nt":
        import ctypes
        from ctypes import wintypes
        kernel=ctypes.WinDLL("kernel32",use_last_error=True)
        kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
        kernel.OpenProcess.restype=wintypes.HANDLE
        kernel.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD]
        kernel.WaitForSingleObject.restype=wintypes.DWORD
        kernel.CloseHandle.argtypes=[wintypes.HANDLE]
        handle=kernel.OpenProcess(0x00100000,False,pid)
        if handle:
            try:
                if kernel.WaitForSingleObject(handle,120000)!=0: raise RuntimeError("更新逾時：原本的工作台仍在執行。")
            finally: kernel.CloseHandle(handle)
    else:
        deadline=time.monotonic()+120
        while time.monotonic()<deadline:
            try: os.kill(pid,0)
            except ProcessLookupError: break
            time.sleep(.2)
        else: raise RuntimeError("更新逾時：原本的工作台仍在執行。")
    subprocess.Popen([sys.executable,str(Path(root)/"main.py")],cwd=root.parent,
        stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0)


if __name__=="__main__" and len(sys.argv)==4 and sys.argv[1]=="--wait":
    wait_and_restart(int(sys.argv[2]),Path(sys.argv[3]))
