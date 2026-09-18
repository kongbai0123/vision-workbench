"""Short-lived, single-use uploads for browser-selected model checkpoints."""
from __future__ import annotations

import os
from pathlib import Path
import secrets
import shutil
import threading


MAX_MODEL_UPLOAD_BYTES = 8 * 1024 * 1024 * 1024


class ModelUploadStore:
    """Stream checkpoints to a private session directory and issue opaque tokens."""

    def __init__(self, data_root: Path):
        self.base = Path(data_root).resolve() / "model-import-uploads"
        self.root = self.base / secrets.token_hex(12)
        self.root.mkdir(parents=True, exist_ok=False)
        self._entries = {}
        self._lock = threading.Lock()

    @staticmethod
    def _filename(value: str) -> str:
        name = str(value or "").replace("\\", "/").rsplit("/", 1)[-1].strip()
        if not name or Path(name).suffix.lower() != ".pt":
            raise ValueError("請選擇副檔名為 .pt 的模型權重")
        if any(ord(character) < 32 for character in name):
            raise ValueError("模型檔名包含無效字元")
        return name[:240]

    def save(self, stream, size: int, filename: str) -> dict:
        name = self._filename(filename)
        if not isinstance(size, int) or size <= 0:
            raise ValueError("模型檔案是空的")
        if size > MAX_MODEL_UPLOAD_BYTES:
            raise ValueError("模型檔案超過 8 GiB 上限")
        token = secrets.token_urlsafe(32)
        partial = self.root / f"{token}.part"
        target = self.root / f"{token}.pt"
        remaining = size
        try:
            with partial.open("xb") as handle:
                while remaining:
                    chunk = stream.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise ValueError("模型檔案上傳不完整，請重新選擇檔案")
                    handle.write(chunk)
                    remaining -= len(chunk)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(partial, target)
        except Exception:
            partial.unlink(missing_ok=True)
            target.unlink(missing_ok=True)
            raise
        with self._lock:
            self._entries[token] = (target, name, size)
        return {"upload_token": token, "filename": name, "bytes": size}

    def claim(self, token: str) -> tuple[Path, str]:
        if not isinstance(token, str) or not token:
            raise ValueError("請先選擇要匯入的 .pt 權重")
        with self._lock:
            entry = self._entries.pop(token, None)
        if entry is None:
            raise ValueError("模型上傳已失效或已被使用，請重新選擇檔案")
        path, name, _size = entry
        if not path.is_file():
            raise ValueError("找不到已上傳的模型檔案，請重新選擇檔案")
        return path, name

    @staticmethod
    def discard(path: Path) -> None:
        Path(path).unlink(missing_ok=True)

    def close(self) -> None:
        with self._lock:
            self._entries.clear()
        shutil.rmtree(self.root, ignore_errors=True)
        try:
            self.base.rmdir()
        except OSError:
            pass
