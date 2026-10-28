"""Detect local source changes and restart the desktop app safely."""
from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone


SOURCE_SUFFIXES = {".py", ".js", ".mjs", ".css", ".html", ".json", ".ps1", ".bat"}
SOURCE_DIRECTORIES = ("workbench", "web", "scripts", "composer_core", "classical_segmentation", "sam2_segmentation")
SOURCE_FILES = ("main.py", "cvat_setup.py", "setup_cvat.ps1", "bootstrap.ps1", "launch.ps1", "vision-workbench.bat",
                "requirements.txt", "requirements-ai.txt", "requirements-training.txt", "requirements-lock.txt")
PROTECTED_DIRECTORIES = ("data/", "models/", "weights/", "runs/", ".venv/", ".venv-models/", ".venv-training/")
VERSION_TAG = re.compile(r"refs/tags/v(\d+)\.(\d+)\.(\d+)(?:\^\{\})?$")


def _git_executable():
    """Find Git in standard Windows locations when the desktop PATH is stale."""
    executable = shutil.which("git")
    if executable:
        return executable
    if os.name != "nt":
        return None
    roots = [os.environ.get("ProgramW6432"), os.environ.get("ProgramFiles"),
             os.environ.get("ProgramFiles(x86)")]
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        roots.append(str(Path(local_app_data) / "Programs"))
    for root in dict.fromkeys(value for value in roots if value):
        for relative in (Path("Git") / "cmd" / "git.exe", Path("Git") / "bin" / "git.exe"):
            candidate = Path(root) / relative
            if candidate.is_file():
                return str(candidate)
    return None


def github_version_status(root, current_version, timeout=8):
    """Read published version tags from origin without changing the checkout."""
    root = Path(root).resolve()
    current = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", current_version)
    if not current:
        return {"state": "unavailable", "message": "無法辨識目前程式版本。"}
    if not (root / ".git").exists():
        return {"state": "unavailable", "message": "此安裝沒有 Git 資料，無法追蹤 GitHub 版本。"}
    git = _git_executable()
    if not git:
        return {"state": "unavailable", "message": "找不到 Git for Windows（git.exe）；請安裝 Git 後重新啟動工作台。"}
    try:
        result = subprocess.run(
            [git, "-C", str(root), "ls-remote", "--tags", "origin", "refs/tags/v*"],
            capture_output=True, text=True, check=True, timeout=timeout,
            encoding="utf-8", errors="replace",
            env=dict(os.environ, GIT_TERMINAL_PROMPT="0", GCM_INTERACTIVE="Never"),
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
    except subprocess.TimeoutExpired:
        return {"state": "unavailable", "message": f"GitHub 版本檢查逾時（{timeout} 秒），請重試或檢查網路。"}
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or error.stdout or "Git 未能讀取 origin 的版本標籤").strip().splitlines()[-1]
        return {"state": "unavailable", "message": f"GitHub 版本檢查失敗：{detail}"}
    except OSError as error:
        detail = error.strerror or str(error)
        return {"state": "unavailable", "message": f"無法啟動 Git（{detail}）。請確認 Git 安裝狀態。"}
    versions = []
    for line in result.stdout.splitlines():
        match = VERSION_TAG.search(line)
        if match:
            versions.append(tuple(map(int, match.groups())))
    if not versions:
        return {"state": "unavailable", "message": "origin 尚無 vX.Y.Z 版本標籤。"}
    newest = max(versions)
    latest = ".".join(map(str, newest))
    if newest > tuple(map(int, current.groups())):
        return {"state": "available", "current": current_version, "latest": latest,
                "message": f"GitHub 已有 v{latest}；目前安裝為 v{current_version}。"}
    return {"state": "current", "current": current_version, "latest": latest,
            "message": f"目前安裝 v{current_version}；GitHub 最新標籤為 v{latest}。"}

def _git(root, *args, timeout=10, check=True):
    executable = _git_executable()
    if not executable:
        raise FileNotFoundError("找不到 Git for Windows（git.exe）")
    return subprocess.run(
        [executable, "-C", str(root), *args], capture_output=True, text=True,
        encoding="utf-8", errors="replace", check=check, timeout=timeout,
        env=dict(os.environ, GIT_TERMINAL_PROMPT="0", GCM_INTERACTIVE="Never"),
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )


def _dirty_entries(root):
    records = _git(root, "status", "--porcelain=v1", "-z", "--untracked-files=all").stdout.split("\0")
    entries, index = [], 0
    while index < len(records):
        record = records[index]
        index += 1
        if not record:
            continue
        code, paths = record[:2], [record[3:]]
        if "R" in code or "C" in code:
            paths.append(records[index])
            index += 1
        entries.append((code, paths))
    return entries


def _backup_paths(entries, incoming):
    paths = set()
    folders = {*SOURCE_DIRECTORIES, "src", "tests", "docs", ".github"}
    for code, names in entries:
        if code != "??":
            paths.update(names)
            continue
        for name in names:
            path = Path(name)
            is_source = path.parts[0] in folders or name in {
                *SOURCE_FILES, "README.md", "CHANGELOG.md", "CONTRIBUTING.md", "LICENSE",
                "pyproject.toml", ".gitignore", ".gitattributes"}
            collides = any(item == name or item.startswith(name + "/") or name.startswith(item + "/")
                           for item in incoming)
            if is_source or collides:
                paths.add(name)
    return sorted(paths)


def _unsafe_git_state(entries, submodules):
    if any(code in {"DD", "AU", "UD", "UA", "DU", "AA", "UU"} for code, _ in entries):
        return "Git 存在尚未解決的衝突；已保留檔案，更新沒有開始。"
    if any(name in submodules for _, names in entries for name in names):
        return "子專案有本機修改，已保留內容；主程式更新沒有開始。"
    if any(code != "??" and name.startswith(PROTECTED_DIRECTORIES)
           for code, names in entries for name in names):
        return "Git 追蹤到專案資料或執行環境的本機修改；已保留資料，程式更新沒有開始。"
    return None


@contextmanager
def _update_lock(git_dir):
    # OS locks disappear when a process crashes; a leftover file never blocks the next update.
    with (git_dir / "workbench-update.lock").open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def pull_remote_update(root, timeout=120, progress=None):
    """Preserve local source edits, fast-forward, and retain a recoverable backup.

    Data/model directories and unrelated untracked artifacts stay in place.
    A failed fetch never touches edits; a failed merge restores the original index.
    """
    root = Path(root).resolve()
    if not (root / ".git").exists():
        return {"state": "unavailable", "message": "此安裝沒有 Git 資料，無法由工作台更新。"}
    report = progress or (lambda message: None)
    backup = {}
    old_head = target = None

    def head():
        return _git(root, "rev-parse", "HEAD").stdout.strip()

    def save_backup(sha, folder, metadata):
        # Retain recovery identity before any fallible durable-ref/manifest write.
        backup["backup_ref"] = sha
        _git(root, "cat-file", "-e", sha + "^{commit}")
        _git(root, "update-ref", "refs/workbench-update-backups/" + folder.name, sha)
        metadata["backup_ref"] = sha
        manifest = folder / "manifest.json"
        manifest.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        backup.update(backup_ref=sha, backup_path=str(manifest))

    try:
        git_dir = Path(_git(root, "rev-parse", "--absolute-git-dir").stdout.strip())
        with _update_lock(git_dir):
            if any((git_dir / name).exists() for name in (
                    "MERGE_HEAD", "CHERRY_PICK_HEAD", "REVERT_HEAD", "rebase-merge", "rebase-apply")):
                return {"state": "blocked", "message": "Git 尚有合併或重整未完成；已保留檔案，更新沒有開始。"}
            branch = _git(root, "symbolic-ref", "--quiet", "--short", "HEAD", check=False)
            if branch.returncode:
                return {"state": "blocked", "message": "目前安裝未位於可追蹤的 Git 分支，更新沒有開始。"}
            upstream = _git(root, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}", check=False)
            if upstream.returncode:
                return {"state": "unavailable", "message": "目前分支未設定遠端追蹤來源，更新沒有開始。"}
            upstream = upstream.stdout.strip()
            remote = _git(root, "config", "--get", "branch." + branch.stdout.strip() + ".remote").stdout.strip()
            if remote == ".":
                return {"state": "unavailable", "message": "目前分支沒有 GitHub 遠端更新來源。"}
            entries = _dirty_entries(root)
            submodules = {record.split("\t", 1)[1] for record in
                          _git(root, "ls-files", "--stage", "-z").stdout.split("\0")
                          if record.startswith("160000 ")}
            blocker = _unsafe_git_state(entries, submodules)
            if blocker:
                return {"state": "blocked", "message": blocker}
            report("正在下載並確認 GitHub 更新…")
            _git(root, "fetch", "--prune", remote, timeout=timeout)
            old_head, target = head(), _git(root, "rev-parse", upstream).stdout.strip()
            ancestry = _git(root, "merge-base", "--is-ancestor", old_head, target, check=False)
            if ancestry.returncode:
                return {"state": "blocked", "message": "本機有不同的 Git 提交，已保留版本；自動更新不會覆蓋這些提交。"}
            changed = _git(root, "diff", "--name-only", "-z", old_head, target).stdout.split("\0")
            if any(name.startswith(PROTECTED_DIRECTORIES) or Path(name).suffix.lower() in {
                    ".pt", ".pth", ".onnx", ".ckpt", ".safetensors"} for name in changed if name):
                return {"state": "blocked", "message": "遠端更新涉及專案資料或執行環境；已保留本機內容，沒有套用。"}
            incoming = _git(root, "ls-tree", "-r", "--name-only", "-z", target).stdout.split("\0")
            current_entries = _dirty_entries(root)
            blocker = _unsafe_git_state(current_entries, submodules)
            if blocker:
                return {"state": "blocked", "message": blocker}
            paths = _backup_paths(current_entries, set(incoming) - {""})
            if paths:
                report("正在自動備份本機程式修改…")
                identifier = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:12]
                folder = git_dir / "workbench-update-backups" / identifier
                folder.mkdir(parents=True)
                label = "Vision Workbench update backup " + identifier
                metadata = {"created": datetime.now(timezone.utc).isoformat(), "head_before": old_head,
                            "target": target, "branch": branch.stdout.strip(), "paths": paths, "label": label}
                pathspec = folder / "paths.txt"
                # A whole-tree positive pathspec also covers staged deletions/renames.
                # Literal exclusions leave unrelated untracked artifacts in place.
                exclusions = [name for code, names in current_entries if code == "??"
                              for name in names if name not in paths]
                specs = ["."] + [":(top,literal,exclude)" + name for name in exclusions]
                pathspec.write_bytes(b"\0".join(name.encode("utf-8") for name in specs) + b"\0")
                previous = _git(root, "rev-parse", "--verify", "refs/stash", check=False).stdout.strip()
                try:
                    _git(root, "stash", "push", "--include-untracked", "--message", label,
                         "--pathspec-from-file=" + str(pathspec), "--pathspec-file-nul", timeout=timeout)
                finally:
                    saved = _git(root, "rev-parse", "--verify", "refs/stash", check=False).stdout.strip()
                    if saved and saved != previous:
                        subject = _git(root, "show", "-s", "--format=%s", saved).stdout.strip()
                        if label in subject:
                            save_backup(saved, folder, metadata)
                if not backup:
                    raise RuntimeError("無法確認本機備份，原始檔案未套用遠端更新。")
            if old_head == target and not backup:
                return {"state": "current", "message": "目前程式已與 GitHub 最新版本同步。"}
            report("正在套用更新；本機修改已保留在備份中…" if backup else "正在套用 GitHub 更新…")
            _git(root, "merge", "--ff-only", "--no-edit", target, timeout=timeout)
            return {"state": "updated", "message": "GitHub 程式已同步，正在驗證並準備重新啟動。",
                    "head": target, **backup}
    except Exception as error:
        logging.exception("GitHub update did not complete")
        # A timeout may be reported after the fast-forward already completed.
        if old_head and target:
            try:
                current = head()
                if current == target and target != old_head:
                    return {"state": "updated", "message": "GitHub 程式已同步，正在驗證並準備重新啟動。",
                            "head": target, **backup}
                if backup and current == old_head:
                    with _update_lock(git_dir):
                        if head() == old_head:
                            _git(root, "stash", "apply", "--index", backup["backup_ref"], timeout=timeout)
            except Exception:
                logging.exception("Local update backup could not be reapplied")
                return {"state": "error", "message": "更新未完成；本機修改已備份，原視窗與專案資料仍保留。",
                        **backup}
        detail = str(error)
        if isinstance(error, subprocess.CalledProcessError):
            detail = (error.stderr or error.stdout or "Git 更新失敗").strip().splitlines()[-1]
        if isinstance(error, subprocess.TimeoutExpired):
            detail = "下載逾時，請確認網路後再按更新。"
        return {"state": "error", "message": "更新未完成，原本修改已保留：" + detail, **backup}


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


def schedule_restart(root, data_root=None):
    root=Path(root).resolve();data_root=Path(data_root or root/"data").resolve()
    logs=data_root/"logs";logs.mkdir(parents=True,exist_ok=True)
    with (logs/"update.log").open("ab") as log:
        return subprocess.Popen([sys.executable,str(Path(__file__).resolve()),"--wait",str(os.getpid()),str(root),
            "--data-root",str(data_root),"--ensure-runtime"],
            cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=log,
            env=dict(os.environ,PYTHONIOENCODING="utf-8"),
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0)


def wait_and_restart(pid, root, data_root=None, ensure_runtime=False):
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
    root=Path(root).resolve()
    if ensure_runtime and (root/"requirements.txt").exists():
        missing=missing_runtime_requirements(root)
        if missing:
            print("正在更新桌面執行環境："+"、".join(missing),flush=True)
            subprocess.run([sys.executable,"-m","pip","install","--disable-pip-version-check",
                "-r",str(root/"requirements.txt")],cwd=root,stdin=subprocess.DEVNULL,
                check=True,timeout=1800,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0)
            if missing_runtime_requirements(root):
                raise RuntimeError("桌面執行環境尚未完成更新，請確認網路後重試。")
    command=[sys.executable,str(root/"main.py")]
    if data_root is not None:
        command.extend(["--data-root",str(Path(data_root).resolve())])
    subprocess.Popen(command,cwd=root.parent,
        stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0)


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--wait",type=int,required=True)
    parser.add_argument("root",type=Path)
    parser.add_argument("--data-root",type=Path)
    parser.add_argument("--ensure-runtime",action="store_true")
    args=parser.parse_args()
    try:
        wait_and_restart(args.wait,args.root,args.data_root,args.ensure_runtime)
    except Exception as error:
        import traceback
        traceback.print_exc()
        if os.name=="nt":
            import ctypes
            logs=Path(args.data_root or args.root/"data")/"logs"/"update.log"
            ctypes.windll.user32.MessageBoxW(None,
                "更新尚未完成，專案資料已保留。請確認網路後重新開啟工作台並重試。\n"+str(error)+"\n紀錄："+str(logs),
                "Vision Workbench 更新",0x10)
        raise SystemExit(1)
