"""Local camera presets, kept outside the repository in the data directory."""
import json
import math
import os
import threading
from pathlib import Path

from .camera_controls import validate_values


class CameraProfiles:
    def __init__(self, root):
        self.path = Path(root) / "camera-profiles.json"
        self.lock = threading.RLock()

    def list(self, device):
        with self.lock:
            if not self.path.exists():
                return {}
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return data.get(device, {})

    def save(self, device, name, settings):
        if not isinstance(device, str) or not device.strip() or len(device) > 256:
            raise ValueError("請選擇有效相機")
        if not isinstance(name, str) or not name.strip() or len(name) > 80:
            raise ValueError("設定檔名稱需為 1–80 個字元")
        if not isinstance(settings, dict):
            raise ValueError("設定檔內容無效")
        for key in ("width", "height"):
            if type(settings.get(key)) is not int or not 32 <= settings[key] <= 8192:
                raise ValueError("設定檔解析度無效")
        fps = settings.get("fps")
        if isinstance(fps, bool) or not isinstance(fps, (float, int)) or not math.isfinite(fps) or not 1 <= fps <= 240:
            raise ValueError("設定檔 FPS 無效")
        pixel = settings.get("pixel_format")
        if not isinstance(pixel, str) or len(pixel) != 4 or not pixel.isascii() or not pixel.isprintable():
            raise ValueError("設定檔格式無效")
        controls = validate_values(settings.get("controls", {}))
        preview_fps = settings.get("preview_fps", 15)
        if preview_fps not in (5, 10, 15, 30):
            raise ValueError("預覽更新率無效")
        clean = {key: settings[key] for key in ("width", "height", "fps", "pixel_format")}
        clean.update(controls=controls, preview_fps=preview_fps)
        with self.lock:
            data = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
            profiles = data.setdefault(device, {})
            if name.strip() not in profiles and len(profiles) >= 50:
                raise ValueError("每台相機最多保存 50 個設定檔")
            profiles[name.strip()] = clean
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(".tmp")
            temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(temporary, self.path)
            return profiles
