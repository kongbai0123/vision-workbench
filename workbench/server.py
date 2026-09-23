"""Loopback-only application service. All modules operate on ProjectStore."""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
import logging
import mimetypes
import os
from pathlib import Path
import secrets
import threading
import hashlib
from collections import OrderedDict
from http.cookies import SimpleCookie
from .routes import validate_method, MethodNotAllowed
from urllib.parse import parse_qs, unquote, urlsplit

from . import __version__
from .jobs import JobManager
from .model_uploads import ModelUploadStore
from .store import ConflictError, ProjectStore, dump
from .training import TrainingWorkspace

APP_ROOT = Path(__file__).resolve().parents[1]


class WorkbenchService:
    def __init__(self, data_root=None, port=0, dialog=None):
        self.data_root = Path(data_root or APP_ROOT / "data").resolve()
        self.data_root.mkdir(parents=True, exist_ok=True)
        self.store = ProjectStore(self.data_root / "projects")
        from .review_workflow import ReviewWorkflow
        self.review_workflow = ReviewWorkflow(self.store)
        self.exports = self.data_root / "exports"
        self.exports.mkdir(exist_ok=True)
        self.incoming = self.data_root / "incoming"
        self.incoming.mkdir(exist_ok=True)
        self.model_uploads = ModelUploadStore(self.data_root)
        self.jobs = JobManager()
        self.training = TrainingWorkspace(self.data_root, self.store)
        self.dialog = dialog
        self._camera = None
        from .camera_profiles import CameraProfiles
        self.camera_profiles = CameraProfiles(self.data_root)
        self._camera_lock = threading.Lock()
        self._ai_lock = threading.Lock()
        from .interactive_runtime import InteractiveRuntime
        def unload_ai():
            from .acquisition import close_ai
            close_ai()
        self._ai_runtime = InteractiveRuntime(self.data_root, unload_ai)
        self._cvat = None
        self._cvat_lock = threading.Lock()
        self._cvat_tickets = {}
        self._cvat_session = []
        self._cvat_baseline = None
        self.httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.httpd.daemon_threads = True
        self.httpd.service = self
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.api_token = secrets.token_urlsafe(32)
        self.image_cache = OrderedDict()
        self.image_cache_lock = threading.Lock()
        self.entry_url = self.url + '/?session=' + self.api_token
        self.thread = threading.Thread(target=self.httpd.serve_forever, name="workbench-http", daemon=True)
        self.started = False

    @property
    def camera(self):
        with self._camera_lock:
            if self._camera is None:
                from .acquisition import CameraService
                self._camera = CameraService(self.incoming)
            return self._camera

    @property
    def cvat(self):
        with self._cvat_lock:
            if self._cvat is None:
                from cvat_setup import CvatSetup
                self._cvat = CvatSetup(APP_ROOT)
            return self._cvat

    def issue_cvat_ticket(self, result):
        ticket = secrets.token_urlsafe(32)
        with self._cvat_lock:
            self._cvat_tickets = {key:value for key,value in self._cvat_tickets.items() if value[0] > __import__('time').monotonic()}
            self._cvat_tickets[ticket] = (__import__('time').monotonic()+120, dict(result))
        return ticket

    def consume_cvat_ticket(self, ticket):
        with self._cvat_lock:
            entry = self._cvat_tickets.pop(ticket, None)
        if not entry or entry[0] < __import__('time').monotonic():
            raise ValueError("CVAT 啟動連結已失效，請重新切換編輯器。")
        return entry[1]

    def start(self):
        self.thread.start()
        self.started = True
        return self

    def close(self):
        if self.started:
            self.httpd.shutdown()
        self.httpd.server_close()
        self.jobs.close()
        self.model_uploads.close()
        self._ai_runtime.close()
        self.training.close()
        import shutil
        shutil.rmtree(self.data_root / 'model-trials', ignore_errors=True)
        if self._camera:
            self._camera.close()
        if self._cvat is not None:
            self._cvat.stop()
        if 'workbench.acquisition' in __import__('sys').modules:
            from .acquisition import close_ai
            close_ai()
        if self.started:
            self.thread.join(timeout=5)
        self.started = False

    def import_paths(self, pid, paths):
        if not isinstance(paths, list) or not paths or not all(isinstance(p, str) and p.strip() for p in paths):
            raise ValueError("請選擇有效的圖片或資料夾")
        self.store.get_project(pid, include_assets=False)
        def run(progress):
            from .pipeline import import_sources
            progress("檢查影像與標註配對", 0, phase='scan')
            parsed = import_sources(paths, progress=lambda message, percent: progress(message, percent))
            records = parsed.get("records", [])
            issues = parsed.get("issues", [])
            if not records:
                details = "；".join(str(item.get("message", item)) if isinstance(item, dict) else str(item) for item in issues)
                raise ValueError(details or "沒有找到可匯入的圖片")
            progress("保存原圖", 0, phase='save')
            result = self.store.add_assets(pid, records, progress=lambda n,total: progress(f"保存原圖 {n} / {total}",round(n/total*95)))
            result["issues"] = issues
            return result
        return self.jobs.submit("import", run)

    def pipeline_job(self, pid, action, payload):
        snapshot = self.store.snapshot(pid)
        format_key = payload.get("format", "native")
        if format_key not in {"native", "coco", "yolo_detection", "yolo_segmentation", "labelme", "classification", "jsonl"}:
            raise ValueError("匯出格式無效")
        tolerance = float(payload.get("tolerance", 0))
        import math
        if not math.isfinite(tolerance) or not 0 <= tolerance <= 100:
            raise ValueError("輪廓誤差必須為 0–100 像素")
        def run(progress):
            from .pipeline import export_project, validate_project
            progress("檢查圖片雜湊、標註、審核與分組")
            if action == "validate":
                return validate_project(snapshot, format_key, tolerance)
            output = Path(payload.get("output_dir") or self.exports).resolve()
            # Never export over the source or inside its immutable image store.
            if output.is_relative_to(self.store.root):
                raise ValueError("匯出位置不能放在工作專案的原圖儲存區")
            result = export_project(snapshot, output, format_key, payload.get("version", "v1"),
                                    tolerance, payload.get("acknowledge_loss") is True)
            result["project_revision"] = snapshot["revision"]
            return self.store.record_export(pid, result)
        return self.jobs.submit(action, run)

    def ai_job(self, pid, payload):
        asset = self.store.get_asset(pid, payload.get("asset_id"), internal=True)
        label = self.store.require_class(pid, payload.get("label"))
        if payload.get("revision") != asset["revision"]:
            raise ConflictError("圖片已修改，請先儲存最新內容再執行 AI")
        def run(progress):
            from .acquisition import segment_image
            progress("載入本機模型並執行分割")
            def predict():
                return segment_image(asset["image_path"], engine=payload.get("engine", "sam2"),
                        points=payload.get("points"), negative_points=payload.get("negative_points"),
                        box=payload.get("box"), label=label, model_dir=APP_ROOT/"models"/"sam2.1-hiera-tiny")
            if payload.get('engine') == 'grabcut':
                result = predict()
            else:
                result = self._ai_runtime.run(predict, lambda: progress('SAM2 處理／等待運算資源'))
            progress('候選已產生')
            result.update(asset_id=asset["id"], revision=asset["revision"])
            return result
        return self.jobs.submit('ai-cpu' if payload.get('engine') == 'grabcut' else 'ai', run)


class Handler(BaseHTTPRequestHandler):
    server_version = "VisionWorkbench/2.0"
    protocol_version = "HTTP/1.1"

    def handle(self):
        try:
            super().handle()
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
            # A closing Chromium image request is not an application failure.
            pass

    @property
    def app(self):
        return self.server.service

    def log_message(self, fmt, *args):
        if 'session=' not in self.path:
            logging.debug("HTTP %s", fmt % args)

    def authorized(self):
        cookie = SimpleCookie()
        try:
            cookie.load(self.headers.get('Cookie', ''))
        except Exception:
            return False
        value = self.headers.get('X-Workbench-Token') or (cookie['workbench_session'].value if 'workbench_session' in cookie else '')
        return secrets.compare_digest(value, self.app.api_token)

    def host_valid(self):
        expected = urlsplit(self.app.url).netloc
        return self.headers.get("Host", "") == expected

    def body(self):
        self.require_mutation_access()
        if self.headers.get_content_type() != "application/json":
            raise ValueError("請使用 JSON 請求")
        try:
            size = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            raise ValueError("請求長度無效")
        if not 0 <= size <= 128*1024*1024:
            raise ValueError("單次標註請求超過 128 MiB")
        raw = self.rfile.read(size)
        self._request_body_consumed = True
        payload = json.loads(raw or b"{}")
        if not isinstance(payload, dict):
            raise ValueError("請求內容必須為物件")
        return payload

    def require_mutation_access(self):
        origin = self.headers.get("Origin")
        if not self.authorized() or not self.host_valid() or (origin and origin != self.app.url) or self.headers.get("X-Workbench") != "1":
            self.drain_small_request_body()
            raise PermissionError("僅接受軟體本機介面的操作")

    def drain_small_request_body(self):
        """Keep Windows from replacing a local 4xx response with a TCP reset."""
        if getattr(self, '_request_body_consumed', False):
            return
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if 0 < size <= 65536:
                previous = self.connection.gettimeout()
                self.connection.settimeout(.5)
                try:
                    self.rfile.read(size)
                    self._request_body_consumed = True
                finally:
                    self.connection.settimeout(previous)
        except (ValueError, OSError):
            pass

    def model_upload(self):
        self.require_mutation_access()
        if self.headers.get_content_type() != "application/octet-stream":
            raise ValueError("模型檔案必須以二進位格式上傳")
        try:
            size = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            raise ValueError("請求長度無效")
        filename = unquote(self.headers.get("X-Workbench-Filename", ""))
        return self.json(self.app.model_uploads.save(self.rfile, size, filename), 201)

    def send_bytes(self, body, content_type, status=200, *, etag=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "private, max-age=31536000, immutable" if etag else "no-store")
        if etag:
            self.send_header('ETag', etag)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self' qrc:; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; connect-src 'self'; object-src 'none'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(body)

    def image_response(self, key, body, kind):
        with self.app.image_cache_lock:
            if len(body) <= 2 * 1024 * 1024:
                self.app.image_cache[key] = (body, kind)
                while len(self.app.image_cache) > 32:
                    self.app.image_cache.popitem(last=False)
        return self.send_bytes(body, kind, etag=key)

    def json(self, value, status=200):
        self.send_bytes(dump(value).encode("utf-8"), "application/json; charset=utf-8", status)

    def handle_error(self, exc):
        if isinstance(exc, (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)):
            return
        self.drain_small_request_body()
        status = 409 if isinstance(exc, ConflictError) else 403 if isinstance(exc, PermissionError) else 404 if isinstance(exc, FileNotFoundError) else 400 if isinstance(exc, ValueError) else 500
        if isinstance(exc, MethodNotAllowed):
            status = 405
        if status == 500:
            logging.exception("Application API failure")
        self.close_connection = True
        try:
            self.json({"error":str(exc), "message":str(exc)}, status)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def do_GET(self):
        try:
            if not self.host_valid():
                raise PermissionError("本機 Host 無效")
            path = unquote(urlsplit(self.path).path)
            if path == '/' and 'session' in parse_qs(urlsplit(self.path).query):
                supplied = parse_qs(urlsplit(self.path).query)['session'][0]
                if not secrets.compare_digest(supplied, self.app.api_token):
                    raise PermissionError('啟動憑證無效')
                self.send_response(303)
                self.send_header('Set-Cookie', f'workbench_session={self.app.api_token}; HttpOnly; SameSite=Strict; Path=/')
                self.send_header('Location', '/')
                self.send_header('Cache-Control', 'no-store')
                self.send_header('Content-Length', '0')
                self.end_headers()
                return
            if path.startswith('/api/') and not self.authorized():
                raise PermissionError('需要本機工作階段憑證')
            parts = path.strip("/").split("/")
            if path == "/api/system":
                return self.json({"name":"Vision Workbench", "version":__version__, "default_export_path":str(self.app.exports),
                    "data_root":str(self.app.data_root), "desktop":self.app.dialog is not None,
                    "capabilities":{"sam2":bool(importlib.util.find_spec("torch") and importlib.util.find_spec("transformers")),
                                    "grabcut":True,"camera":True,"offline":True,
                                    "training":self.app.training.capabilities()}})
            if path == "/api/projects":
                return self.json({"projects":self.app.store.list_projects()})
            if path == "/api/model-catalog":
                refresh = parse_qs(urlsplit(self.path).query).get("refresh", ["0"])[0] == "1"
                return self.json(self.app.training.refresh_components() if refresh else self.app.training.capabilities())
            if parts[:2] == ['api','model-trials'] and len(parts)==5 and parts[3]=='frames':
                image=self.app.training.model_trial_image(parts[2],parts[4])
                return self.send_bytes(image.read_bytes(),'image/jpeg')
            if parts[:2] == ["api", "jobs"] and len(parts) == 3:
                return self.json(self.app.jobs.get(parts[2]))
            if path == "/api/camera/devices":
                return self.json({"devices":self.app.camera.devices()})
            if path == "/api/camera/status":
                return self.json(self.app.camera.status())
            if path == "/api/camera/profiles":
                device = parse_qs(urlsplit(self.path).query).get("device", [""])[0]
                return self.json({"profiles": self.app.camera_profiles.list(device)})
            if path == "/api/camera/capabilities":
                index=int(parse_qs(urlsplit(self.path).query).get("index",["0"])[0])
                return self.json(self.app.camera.capabilities(index))
            if path == "/api/cvat/status":
                return self.json(self.app.cvat.status())
            if path == "/api/camera/frame":
                processed = parse_qs(urlsplit(self.path).query).get("processed", ["0"])[0] == "1"
                return self.send_bytes(self.app.camera.frame_jpeg(processed=processed), "image/jpeg")
            if parts[:2] == ["api", "projects"] and len(parts) >= 3:
                pid = parts[2]
                if len(parts) == 3:
                    return self.json(self.app.store.get_project(pid))
                if len(parts) == 4 and parts[3] == "classes":
                    return self.json(self.app.store.class_usage(pid))
                if len(parts) == 4 and parts[3] == "training":
                    self.app.store.get_project(pid, include_assets=False)
                    return self.json(self.app.training.overview(pid))
                if len(parts) == 5 and parts[3:] == ['training', 'status']:
                    return self.json(self.app.training.status(pid))
                if len(parts) == 6 and parts[3] == "training-runs":
                    self.app.store.get_project(pid, include_assets=False)
                    if parts[5] == "metrics":
                        return self.json(self.app.training.run_metrics(pid, parts[4]))
                if len(parts) >= 5 and parts[3] == "assets":
                    aid = parts[4]
                    if len(parts) == 5:
                        return self.json(self.app.store.get_asset(pid, aid))
                    if len(parts) == 6 and parts[5] == "image":
                        image = self.app.store.image_path(pid, aid)
                        image_query = parse_qs(urlsplit(self.path).query)
                        key = '"' + hashlib.sha256((str(image) + str(image.stat().st_mtime_ns) + urlsplit(self.path).query).encode()).hexdigest() + '"'
                        if self.headers.get('If-None-Match') == key:
                            return self.send_bytes(b'', 'image/jpeg', 304, etag=key)
                        with self.app.image_cache_lock:
                            cached = self.app.image_cache.get(key)
                        if cached:
                            return self.send_bytes(*cached, etag=key)
                        crop_value = image_query.get("crop", [""])[0]
                        if crop_value:
                            from PIL import Image
                            import io
                            try:
                                x, y, width, height = [int(value) for value in crop_value.split(",")]
                                max_width, max_height = [int(value) for value in image_query.get("max", ["760,420"])[0].split(",")]
                            except (TypeError, ValueError):
                                raise ValueError("圖片裁切參數無效")
                            if width < 1 or height < 1 or not (64 <= max_width <= 1200 and 64 <= max_height <= 900):
                                raise ValueError("圖片裁切範圍無效")
                            with Image.open(image) as source:
                                left, top = max(0, x), max(0, y)
                                right, bottom = min(source.width, x + width), min(source.height, y + height)
                                if right <= left or bottom <= top:
                                    raise ValueError("圖片裁切位置超出範圍")
                                preview = source.convert("RGB").crop((left, top, right, bottom))
                                preview.thumbnail((max_width, max_height))
                                buffer = io.BytesIO()
                                preview.save(buffer, "JPEG", quality=88, optimize=True)
                            return self.image_response(key, buffer.getvalue(), "image/jpeg")
                        if image_query.get("thumbnail", ["0"])[0] == "1":
                            from PIL import Image
                            import io
                            with Image.open(image) as source:
                                source.thumbnail((320, 240))
                                thumbnail = source.convert("RGB")
                                buffer = io.BytesIO()
                                thumbnail.save(buffer, "JPEG", quality=80)
                            return self.image_response(key, buffer.getvalue(), "image/jpeg")
                        from .acquisition import preview_image
                        content, kind = preview_image(image)
                        return self.image_response(key, content, kind)
                    if len(parts) == 6 and parts[5] == "history":
                        return self.json({"history":self.app.store.history(pid, aid)})
            if path.startswith("/api/"):
                raise FileNotFoundError("找不到 API")
            relative = "index.html" if path == "/" else path.lstrip("/")
            web = APP_ROOT / "web"
            target = (web / relative).resolve()
            if not target.is_relative_to(web.resolve()) or not target.is_file():
                raise FileNotFoundError("找不到介面檔案")
            kind = "text/javascript" if target.suffix in {".js", ".mjs"} else mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            self.send_bytes(target.read_bytes(), kind + ("; charset=utf-8" if kind.startswith("text/") else ""))
        except Exception as exc:
            self.handle_error(exc)

    def do_POST(self):
        self.mutate("POST")

    def do_PUT(self):
        self.mutate("PUT")

    def do_PATCH(self):
        self.mutate("PATCH")

    def do_DELETE(self):
        self.mutate("DELETE")

    def mutate(self, method):
        self._request_body_consumed = False
        try:
            path = urlsplit(self.path).path
            validate_method(path, method)
            if path == "/api/model-uploads" and method == "POST":
                return self.model_upload()
            payload = self.body()
            from vision_workbench.contracts import validate_request
            validate_request(path, method, payload)
            parts = path.strip("/").split("/")
            if path == "/api/projects" and method == "POST":
                return self.json(self.app.store.create_project(payload.get("name")))
            if parts[:2] == ["api", "model-components"] and len(parts) == 4 and parts[3] == "install" and method == "POST":
                component_id = parts[2]
                return self.json(self.app.jobs.submit("model-install", lambda progress:
                    self.app.training.install_component(component_id, progress)))
            if parts[:2] == ["api", "jobs"] and len(parts) == 4 and method == "POST":
                return self.json(self.app.jobs.control(parts[2], parts[3]))
            if path == "/api/dialog":
                if not self.app.dialog:
                    raise ValueError("瀏覽器模式請輸入完整路徑；桌面版可使用原生檔案選擇器")
                return self.json({"paths":self.app.dialog(payload.get("kind"))})
            if path == "/api/cvat/setup":
                return self.json(self.app.cvat.start())
            if path == "/api/cvat/reboot":
                if payload.get("confirm") is not True:
                    raise ValueError("重新啟動需要明確確認。")
                return self.json(self.app.cvat.request_reboot())
            if path == "/api/cvat/reboot/cancel":
                return self.json(self.app.cvat.cancel_reboot())
            if path == "/api/cvat/launch":
                pid = payload.get("project_id")
                snapshot = self.app.store.snapshot(pid)
                def launch_cvat(progress):
                    progress(f"啟動 CVAT · {snapshot['name']}")
                    result = self.app.cvat.launch()
                    self.app._cvat_session = list(result.get("auth_cookies", []))
                    from .cvat_bridge import CvatProjectBridge
                    linked = CvatProjectBridge(self.app.data_root, self.app.store).ensure_project(
                        snapshot, result.get("auth_cookies", []), progress, store=self.app.store)
                    self.app._cvat_baseline = self.app.store.snapshot(pid)
                    asset_ids=linked.get('asset_ids',[])
                    frame=asset_ids.index(payload.get('asset_id')) if payload.get('asset_id') in asset_ids else 0
                    result.update(url=linked['url']+f'?frame={frame}',job_id=linked['job_id'],asset_ids=asset_ids,project_id=pid)
                    return {"ticket":self.app.issue_cvat_ticket(result), "project_id":pid,
                            "project_name":snapshot["name"], "url":result["url"],
                            "skipped_masks":linked.get("skipped_masks",0)}
                return self.json(self.app.jobs.submit("cvat", launch_cvat))
            if path == "/api/cvat/import":
                pid = payload.get("project_id")
                snapshot = self.app._cvat_baseline
                if snapshot is None or snapshot['id']!=pid:
                    raise ValueError('CVAT 工作階段與目前專案不一致，請重新開啟。')
                if not self.app._cvat_session:
                    raise ValueError("CVAT 工作階段已結束，請重新開啟後再讀回標註。")
                def import_cvat(progress):
                    from .cvat_bridge import CvatProjectBridge
                    from .editor_sync import commit_updates
                    progress("讀取 CVAT 已儲存標註",20)
                    bridge=CvatProjectBridge(self.app.data_root, self.app.store)
                    updates = bridge.read_annotations(snapshot,self.app._cvat_session)
                    updated=commit_updates(self.app.store,pid,updates,source='cvat')
                    self.app._cvat_baseline=self.app.store.snapshot(pid)
                    bridge.mark_synced(self.app._cvat_baseline,self.app._cvat_session)
                    return {"updated":updated}
                return self.json(self.app.jobs.submit("cvat-import", import_cvat))
            if path == "/api/open-folder":
                if payload.get("model_export_id"):
                    pid, export_id = payload.get("project_id"), payload["model_export_id"]
                    self.app.training.model_export(pid, export_id)
                    folder = self.app.training.model_exports_dir(pid) / export_id
                elif payload.get("export_id"):
                    project = self.app.store.get_project(payload.get("project_id"))
                    item = next((e for e in project["exports"] if e["id"] == payload["export_id"]), None)
                    if item is None:
                        raise FileNotFoundError("找不到匯出記錄")
                    folder = Path(item["path"])
                elif payload.get("project_id"):
                    folder = self.app.store.directory(payload["project_id"])
                else:
                    folder = self.app.exports
                if not folder.is_dir():
                    raise FileNotFoundError("找不到資料夾")
                if os.name == "nt":
                    os.startfile(folder)
                return self.json({"path":str(folder)})
            if path == "/api/camera/start":
                return self.json(self.app.camera.start(index=int(payload.get("index", 0)),width=int(payload.get("width",1280)),
                                  height=int(payload.get("height",720)),fps=float(payload.get("fps",30)),
                                  pixel_format=payload.get("pixel_format","MJPG"), controls=payload.get("controls")))
            if path == "/api/camera/controls":
                return self.json(self.app.camera.set_controls(payload.get("values"), payload.get("reset", False)))
            if path == "/api/camera/profiles":
                return self.json({"profiles": self.app.camera_profiles.save(payload.get("device"), payload.get("name"), payload.get("settings"))})
            if path == "/api/camera/stop":
                return self.json(self.app.camera.stop())
            if path == "/api/camera/processing":
                return self.json(self.app.camera.set_processing(payload.get("mode", "original"), payload.get("settings")))
            if path == "/api/camera/record/start":
                self.app.store.get_project(payload.get("project_id"), include_assets=False)
                return self.json(self.app.camera.start_recording(payload["project_id"]))
            if path == "/api/camera/record/stop":
                return self.json(self.app.camera.stop_recording())
            if parts[:2] == ["api", "predictions"] and len(parts) == 4 and parts[3] == "accept" and method == "POST":
                return self.json(self.app.training.accept_predictions(parts[2], payload.get("asset_ids")))
            if parts[:2] != ["api", "projects"] or len(parts) < 3:
                raise FileNotFoundError("找不到 API")
            pid = parts[2]
            if len(parts) == 3 and method == "PATCH":
                return self.json(self.app.store.update_project(pid,name=payload.get("name"),classes=payload.get("classes")))
            if len(parts) == 3 and method == "DELETE":
                return self.json(self.app.store.delete_project(pid,payload.get("revision")))
            if len(parts) == 4 and parts[3] == "classes" and method == "POST":
                return self.json(self.app.store.manage_classes(pid,payload.get("classes"),
                    payload.get("replacements",{}),payload.get("revision"),payload.get("delete_objects",[])))
            if len(parts) == 5 and parts[3] == "assets" and method == "PUT":
                return self.json(self.app.store.save_asset(pid,parts[4],payload.get("shapes"),payload.get("revision")))
            if len(parts) == 5 and parts[3] == "assets" and method == "DELETE":
                return self.json(self.app.store.delete_asset(pid,parts[4],payload.get("revision")))
            if len(parts) == 4 and parts[3] == "assets" and method == "DELETE":
                return self.json(self.app.store.delete_assets(pid,payload.get("asset_ids"),payload.get("revisions")))
            if len(parts) == 6 and parts[3] == "assets" and parts[5] == "restore":
                return self.json(self.app.store.restore(pid,parts[4],payload.get("history_id"),payload.get("revision")))
            if len(parts) == 5 and parts[3] == "training-runs" and parts[4] and method == "POST":
                raise FileNotFoundError("找不到訓練操作")
            if len(parts) == 6 and parts[3] == "training-runs" and parts[5] == "stop" and method == "POST":
                return self.json(self.app.training.stop_run(pid, parts[4]))
            if len(parts) != 4:
                raise FileNotFoundError("找不到 API")
            from .api_actions import dispatch_project_action
            return dispatch_project_action(self, pid, parts[3], payload)
        except Exception as exc:
            self.handle_error(exc)
