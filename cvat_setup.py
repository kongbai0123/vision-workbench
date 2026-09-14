"""Explicit-click CVAT provisioning; reading status never installs system software.

Windows prerequisites are handled by setup_cvat.ps1. Only this workspace's named
Compose project is managed, using Docker Desktop's local Linux named pipe rather
than the user's current (possibly remote) Docker context. Volumes are never deleted.
"""
from __future__ import annotations

import copy
import hashlib
import http.cookiejar
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import socket
import subprocess
import threading
import time
from urllib.request import Request, build_opener, HTTPCookieProcessor, ProxyHandler

ROOT = Path(__file__).resolve().parent
NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
CVAT_VERSION = "v2.49.0"
TRAEFIK_IMAGE = "traefik:v3.7.13"  # Docker 29 API negotiation; upstream v3.3 is incompatible.
CVAT_COMMIT = "595a13312adf153575d6f7d8e97f76aa541545f6"
CVAT_ORIGIN = "http://cvat.localhost:8768"
DOCKER_PIPE = "npipe:////./pipe/dockerDesktopLinuxEngine"
# Every upstream file is pinned to an immutable commit AND a SHA-256 digest.
UPSTREAM_FILES = {
    "docker-compose.yml": "4db4f8d13755ded59a9f5378dcf69c95e45c74a2a36bfdb1a4ef2aea306239bd",
    "components/analytics/grafana_conf.yml": "bb828236e1ff8a6b94cd1fc949c65a94a362a35753cc7253548eec6b7b034127",
    "components/analytics/vector/vector.toml": "f22d3e316b2644c48c2c28916664670954b294152200c11392129ade69010bf6",
    "components/analytics/grafana/dashboards/all_events.json": "6a9ff91babc85e017a306fa2c5adb435f76463863eb1873c3e6a676e234739f6",
    "components/analytics/grafana/dashboards/management.json": "9c4bbb7a84b6a5e197496dfaa569dca1ec7ad8d3496f760e88f45e9dcc81ac7d",
    "components/analytics/grafana/dashboards/monitoring.json": "d2662ad47057a6491a9d176af739033d8f43f2c248960d95ebcfa448441ea472",
}
STEP_LABELS = [
    ("system", "檢查 Windows 與硬體"),
    ("wsl", "準備 WSL 2"),
    ("docker", "安裝與啟動 Docker"),
    ("download", "下載 CVAT"),
    ("services", "啟動 CVAT 服務"),
    ("account", "建立本地帳號與驗證"),
]
ACTIVE_PHASES = {"checking", "uac", "wsl", "downloading_docker", "installing_docker",
                 "waiting_docker", "download", "services", "account"}
# Must match $stateSchema in setup_cvat.ps1. A state file written by an older build
# keeps its wording forever while can_setup is False, so its text/step are not trusted.
STATE_SCHEMA = 2
REBOOT_PENDING_TEXT = "Windows 有其他更新待重新啟動，必須先重新啟動，才能繼續安裝 WSL 2。"
REBOOT_DELAY = 20  # Seconds of grace before Windows restarts; cancellable via shutdown /a.


class SetupPause(RuntimeError):
    def __init__(self, phase, text):
        self.phase = phase
        super().__init__(text)


def read_json(path):
    try:
        value = json.loads(Path(path).read_text("utf-8-sig"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), "utf-8")
    temporary.replace(path)


def make_local_compose(config, project, upstream):
    """Scope upstream's generated JSON without requiring a bundled YAML library."""
    config = copy.deepcopy(config)
    # The first Compose parse uses --no-interpolate. Resolve only pinned upstream
    # placeholders here so shell $$ variables are evaluated just once at runtime.
    values = {"CVAT_HOST": "cvat.localhost", "CVAT_VERSION": CVAT_VERSION, "CVAT_BASE_URL": CVAT_ORIGIN}
    def interpolate(value):
        if isinstance(value, str):
            return re.sub(r"(?<!\$)\$\{([A-Za-z_][A-Za-z_0-9]*)(?:(?::-|-)([^}]*))?\}",
                          lambda match: values.get(match[1], match[2] or ""), value)
        if isinstance(value, list):
            return [interpolate(item) for item in value]
        if isinstance(value, dict):
            return {key: interpolate(item) for key, item in value.items()}
        return value
    config = interpolate(config)
    config["name"] = project
    network_name = project + "_cvat"
    for section in ("networks", "volumes"):
        for name in config.get(section, {}):
            config[section][name] = {"name": project + "_" + name}
    for name, service in config["services"].items():
        service.pop("container_name", None)
        service.pop("ports", None)
        service["restart"] = "no"  # Only the hub starts its services.
        service.setdefault("labels", {})["annotation.hub.project"] = project
        for volume in service.get("volumes", []):
            if volume.get("type") == "bind":
                source = volume["source"]
                if source == "/var/run/docker.sock":
                    continue
                resolved = (upstream / source).resolve()
                if not resolved.is_relative_to(upstream.resolve()):
                    raise ValueError("CVAT 設定含有工作區外的掛載路徑。")
                volume["source"] = str(resolved)
        if name.startswith("cvat_") and service.get("image", "").startswith("cvat/"):
            service["image"] = service["image"].split(":")[0] + ":" + CVAT_VERSION
        if "environment" in service:
            service["environment"]["CVAT_HOST"] = "cvat.localhost"
            service["environment"]["CVAT_BASE_URL"] = CVAT_ORIGIN
    proxy = config["services"]["traefik"]
    proxy["image"] = TRAEFIK_IMAGE
    proxy["ports"] = [{"target": 8080, "published": "8768", "host_ip": "127.0.0.1", "protocol": "tcp"}]
    proxy["environment"]["TRAEFIK_PROVIDERS_DOCKER_NETWORK"] = network_name
    proxy["environment"]["TRAEFIK_PROVIDERS_DOCKER_CONSTRAINTS"] = "Label(`annotation.hub.project`, `" + project + "`)"
    config["services"]["cvat_grafana"]["environment"]["GF_SERVER_ROOT_URL"] = CVAT_ORIGIN + "/analytics"
    return config


class CvatSetup:
    def __init__(self, root=None):
        self.root = Path(root or ROOT).resolve()
        self.folder = self.root / "data" / "cvat"
        self.upstream = self.folder / "upstream"
        self.compose_file = self.folder / "compose.json"
        self.state_file = self.folder / "setup-state.json"
        self.installer_state = self.folder / "installer-state.json"
        self.project = "annotation_cvat_" + hashlib.sha256(str(self.root).lower().encode()).hexdigest()[:12]
        self.lock = threading.RLock()
        self.launch_lock = threading.Lock()
        self.stopping = threading.Event()
        self.worker = None
        self.launching = False
        self.owned_running = False
        self._probe_cache = ({}, 0)
        self._session = None
        self._state = read_json(self.state_file)

    def _run(self, command, timeout=60, input_text=None, env=None, check=True, private=False):
        try:
            result = subprocess.run(command, cwd=self.root, input=input_text, capture_output=True,
                                    text=True, encoding="utf-8", errors="replace", timeout=timeout,
                                    creationflags=NO_WINDOW, env=env)
        except subprocess.TimeoutExpired:
            raise RuntimeError("作業逾時；可重試並沿用已下載的內容。") from None
        if check and result.returncode:
            # Do not copy command lines, subprocess output or credentials into UI/logs.
            name = "CVAT 帳號初始化" if private else Path(command[0]).name
            raise RuntimeError(f"{name} 未成功完成（代碼 {result.returncode}）。請檢查環境後重試。")
        return result

    def _powershell(self, mode, timeout=20):
        return self._run(["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                          "-File", str(self.root / "setup_cvat.ps1"), "-Mode", mode], timeout=timeout)

    def _probe(self, force=False):
        with self.lock:
            cached, at = self._probe_cache
            if not force and time.monotonic() - at < 20:
                return dict(cached)
        if os.name != "nt":
            result = {"supported": False, "reason": "一鍵環境準備目前支援 Windows 10／11 x64。", "boot_id": ""}
        else:
            try:
                result = json.loads(self._powershell("Probe").stdout.lstrip("\ufeff"))
            except (OSError, RuntimeError, ValueError):
                result = {"supported": True, "reason": "尚無法完整偵測環境；按一鍵準備後重新檢查。", "boot_id": ""}
        with self.lock:
            self._probe_cache = (result, time.monotonic())
        return dict(result)

    def _set(self, phase, text, step=None, **extra):
        with self.lock:
            self._state.update(phase=phase, text=text, updated_at=time.time(), **extra)
            if step:
                self._state["step"] = step
            atomic_json(self.state_file, self._state)

    def _installer_snapshot(self):
        result = read_json(self.installer_state)
        # A previous attempt's installer state must not overwrite a new Python phase.
        if result.get("updated_at", 0) >= self._state.get("updated_at", 0):
            return result
        return {}

    def status(self):
        probe = self._probe()
        with self.lock:
            state = dict(self._state)
            installer = self._installer_snapshot()
            state.update({k: v for k, v in installer.items() if k in ("phase", "text", "step", "boot_id")})
            busy = self.launching or bool(self.worker and self.worker.is_alive())
            # A prerequisite helper may outlive a closed hub; never start a second installer.
            helper_busy = self._helper_running(installer)
            busy = busy or helper_busy
            installed = (self.compose_file.exists() and state.get("installed_version") == CVAT_VERSION
                         and state.get("project") == self.project)
            phase = state.get("phase", "not_installed")
            description = state.get("text", "首次使用需準備 WSL 2、Docker 與 CVAT。")
            if phase in ACTIVE_PHASES and not busy:
                phase, description = "interrupted", "上次準備尚未完成，可按繼續準備。"
            rebooted = state.get("boot_id") and probe.get("boot_id") and probe["boot_id"] != state["boot_id"]
            reboot_cleared = rebooted or not probe.get("reboot_pending", True)
            stale_state = state.get("schema") != STATE_SCHEMA
            if phase == "reboot_required" and reboot_cleared:
                phase, description = "resume_required", "已重新啟動 Windows，按繼續準備即可完成 CVAT。"
            elif phase == "reboot_required" and stale_state:
                description = REBOOT_PENDING_TEXT
            if phase == "blocked" and probe.get("supported"):
                phase, description = "resume_required", "系統已符合需求，可以繼續準備 CVAT。"
            if not probe.get("supported", True):
                phase, description = "blocked", probe.get("reason", "系統不符合需求。")
            prerequisites_present = all(probe.get(key, True) for key in ("docker_installed", "wsl_ready", "features_ready"))
            if installed and not prerequisites_present and phase == "ready":
                phase, description = "resume_required", "偵測到 Docker／WSL 環境不完整，請按繼續準備修復。"
            ready = installed and prerequisites_present and phase not in ("blocked", "reboot_required", "error", "waiting_action", "interrupted") and not busy
            if ready and phase not in ("error", "waiting_action"):
                phase = "ready"
                description = "已備妥 · 進入時自動啟動" if not self.owned_running else "執行中 · 可直接進入"
            current_step = state.get("step", "system")
            if phase == "reboot_required" and stale_state:
                current_step = "system"
            if installed and not prerequisites_present:
                current_step = "docker" if probe.get("wsl_ready") and probe.get("features_ready") else "wsl"
            ids = [row[0] for row in STEP_LABELS]
            index = ids.index(current_step) if current_step in ids else 0
            steps = [{"id": sid, "label": label,
                      "state": "done" if ready or n < index else
                               ("running" if busy else "attention") if n == index else "pending"}
                     for n, (sid, label) in enumerate(STEP_LABELS)]
            return {"phase": phase, "ready": ready, "text": description, "steps": steps,
                    "started_at": state.get("started_at"),
                    "busy": busy, "running": self.owned_running,
                    "can_setup": not busy and phase not in ("blocked", "reboot_required"),
                    "reboot_required": phase == "reboot_required", "version": CVAT_VERSION,
                    "can_reboot": phase == "reboot_required" and os.name == "nt" and not busy,
                    "resume": phase in ("resume_required", "interrupted", "error", "waiting_action"),
                    "requirements": probe, "installed": installed,
                    "notice": "首次下載需要網路；Windows 可能要求管理員確認或重新啟動。Docker 首次啟動可能要求同意授權條款。"}

    @staticmethod
    def _helper_running(snapshot):
        if snapshot.get("phase") not in ACTIVE_PHASES or not snapshot.get("pid"):
            return False
        if os.name != "nt":
            return False
        # OpenProcess is read-only; stale PID reuse is bounded by a 2-hour installer window.
        if time.time() - snapshot.get("updated_at", 0) > 7200:
            return False
        import ctypes
        kernel = ctypes.windll.kernel32
        kernel.OpenProcess.restype = ctypes.c_void_p
        handle = kernel.OpenProcess(0x1000, False, int(snapshot["pid"]))
        if not handle:
            return False
        code = ctypes.c_ulong()
        try:
            return bool(kernel.GetExitCodeProcess(ctypes.c_void_p(handle), ctypes.byref(code))) and code.value == 259
        finally:
            kernel.CloseHandle(ctypes.c_void_p(handle))

    def request_reboot(self):
        """Only ever reached from an explicit confirmation dialog in the hub UI."""
        if os.name != "nt":
            raise RuntimeError("僅 Windows 桌面版可由中心觸發重新啟動。")
        snapshot = self.status()
        if not snapshot["can_reboot"]:
            raise RuntimeError("目前不需要重新啟動，或仍有準備作業進行中。")
        # A delay (not /t 0) keeps this reversible: cancel_reboot runs shutdown /a.
        self._run(["shutdown.exe", "/r", "/t", str(REBOOT_DELAY), "/c",
                   "標註中心：重新啟動以完成 Windows 更新，回來後可繼續準備 CVAT。"], timeout=20)
        return {"scheduled": True, "seconds": REBOOT_DELAY}

    def cancel_reboot(self):
        if os.name != "nt":
            raise RuntimeError("僅 Windows 桌面版可取消重新啟動。")
        self._run(["shutdown.exe", "/a"], timeout=20, check=False)
        return {"scheduled": False}

    def start(self):
        with self.lock:
            snapshot = self.status()
            if snapshot["busy"]:
                return snapshot
            if not snapshot["can_setup"]:
                return snapshot
            if self.stopping.is_set():
                raise RuntimeError("中心正在關閉。")
            self._set("checking", "正在檢查本機環境…", "system", started_at=time.time())
            self.worker = threading.Thread(target=self._setup, name="cvat-setup", daemon=True)
            self.worker.start()
        return self.status()

    def _setup(self):
        try:
            probe = self._probe(force=True)
            if not probe.get("supported", True):
                raise SetupPause("blocked", probe.get("reason", "此系統不支援一鍵準備。"))
            self._set("checking", "正在準備執行環境…", "wsl", boot_id=probe.get("boot_id", ""))
            # This is the sole path that can request UAC/install Windows software.
            self._powershell("Setup", timeout=7200)
            prerequisite = read_json(self.installer_state)
            if prerequisite.get("phase") != "prerequisites_ready":
                raise SetupPause(prerequisite.get("phase", "error"), prerequisite.get("text", "環境準備未完成，可重試。"))
            self._check_closing()
            self._ensure_docker()
            self._set("download", f"正在下載並驗證 CVAT {CVAT_VERSION}…", "download")
            self._prepare_compose()
            self._compose("pull", timeout=3600)
            self._check_closing()
            self._start_services()
            self._set("account", "正在建立本地帳號並驗證登入…", "account")
            self._session = self._login()
            self._set("ready", "CVAT 已備妥，可以直接進入。", "account",
                      installed_version=CVAT_VERSION, project=self.project)
        except SetupPause as error:
            self._set(error.phase, str(error))
            if self.owned_running:
                self._stop_owned()
        except Exception as error:
            self._set("error", str(error) if isinstance(error, RuntimeError) else "準備失敗，請檢查網路與系統環境後重試。")
            if self.owned_running:
                self._stop_owned()
        finally:
            if self.stopping.is_set() and self.owned_running:
                self._stop_owned()
            self._probe_cache = ({}, 0)

    def _check_closing(self):
        if self.stopping.is_set():
            raise SetupPause("interrupted", "準備已暫停；重新開啟中心後可繼續。")

    def _docker_binary(self):
        bases = [Path(os.environ.get("LOCALAPPDATA", "C:/missing")) / "Programs/DockerDesktop",
                 Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Docker/Docker"]
        for base in bases:
            candidate = base / "resources/bin/docker.exe"
            if candidate.is_file():
                return str(candidate)
        candidate = shutil.which("docker.exe" if os.name == "nt" else "docker")
        if candidate:
            return candidate
        raise RuntimeError("尚未找到 Docker，請按一鍵準備。")

    def _docker(self, *args, **kwargs):
        env = os.environ.copy()
        binary = self._docker_binary()
        # A hub opened before Docker installation still has the old PATH.
        env["PATH"] = str(Path(binary).parent) + os.pathsep + env.get("PATH", "")
        for name in list(env):
            if name.startswith(("DOCKER_", "COMPOSE_", "CVAT_")):
                env.pop(name)
        env.update(CVAT_HOST="cvat.localhost", CVAT_VERSION=CVAT_VERSION, CVAT_BASE_URL=CVAT_ORIGIN)
        return self._run([binary, "--host", DOCKER_PIPE, *args], env=env, **kwargs)

    def _compose(self, *args, **kwargs):
        return self._docker("compose", "--project-name", self.project, "--file", str(self.compose_file), *args, **kwargs)

    def _ensure_docker(self):
        self._set("waiting_docker", "正在啟動 Docker；首次使用請在 Docker 視窗確認授權條款。", "docker")
        try:
            if self._docker("info", "--format", "{{.OSType}}", timeout=10, check=False).stdout.strip() == "linux":
                return
        except (OSError, RuntimeError):
            pass
        executable = Path(self._docker_binary()).parent.parent.parent / "Docker Desktop.exe"
        if not executable.is_file():
            raise SetupPause("waiting_action", "找不到 Docker Desktop，請修復安裝後重試。")
        # Docker's first-run agreement is an interactive consent screen. Show that
        # application so the user can complete it; installer/helper windows stay hidden.
        self._run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
                   "Start-Process -FilePath $env:ANNOTATION_DOCKER_EXE -WindowStyle Normal"],
                  env={**os.environ, "ANNOTATION_DOCKER_EXE": str(executable)})
        deadline = time.monotonic() + 240
        while time.monotonic() < deadline:
            self._check_closing()
            try:
                result = self._docker("info", "--format", "{{.OSType}}", timeout=10, check=False)
                if result.stdout.strip() == "linux":
                    return
            except (OSError, RuntimeError):
                pass
            self.stopping.wait(2)
        raise SetupPause("waiting_action", "Docker 尚未就緒。請在 Docker Desktop 完成首次授權／登入或依其錯誤提示處理，再按繼續準備。")

    def _download_file(self, relative, expected):
        target = self.upstream / relative
        if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == expected:
            return
        self._check_closing()
        url = f"https://raw.githubusercontent.com/cvat-ai/cvat/{CVAT_COMMIT}/{relative}"
        client = build_opener()
        with client.open(Request(url, headers={"User-Agent": "LocalAnnotationHub/1"}), timeout=90) as response:
            content = response.read(8 * 1024 * 1024 + 1)
        if hashlib.sha256(content).hexdigest() != expected:
            raise RuntimeError("CVAT 下載檔案校驗失敗，已停止安裝；請重試。")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)

    def _prepare_compose(self):
        for path, digest in UPSTREAM_FILES.items():
            self._download_file(path, digest)
        # Compose performs YAML parsing only; this does not start a container.
        result = self._docker("compose", "--project-name", self.project, "--file", str(self.upstream / "docker-compose.yml"),
                              "config", "--format", "json", "--no-path-resolution", "--no-interpolate", timeout=30)
        config = make_local_compose(json.loads(result.stdout), self.project, self.upstream)
        atomic_json(self.compose_file, config)

    def _project_running(self):
        result = self._compose("ps", "--status", "running", "--quiet", check=False, timeout=15)
        return bool(result.stdout.strip())

    def _start_services(self):
        self._check_closing()
        self._set("services", "正在啟動 CVAT 並等待資料庫初始化…", "services")
        already_running = self._project_running()
        with socket.socket() as probe:
            probe.settimeout(1)
            if probe.connect_ex(("127.0.0.1", 8768)) == 0 and not already_running:
                raise RuntimeError("連接埠 8768 已被其他軟體使用，請先釋放後重試。")
        # up can start part of the project before returning an error. Record ownership
        # before it runs so a failed start can still stop those exact containers.
        self.owned_running = True
        self._compose("up", "--detach", timeout=300)
        deadline = time.monotonic() + 360
        client = build_opener(ProxyHandler({}))
        while time.monotonic() < deadline:
            self._check_closing()
            try:
                request = Request("http://127.0.0.1:8768/api/server/about", headers={"Host": "cvat.localhost:8768"})
                with client.open(request, timeout=3) as response:
                    data = json.load(response)
                if data.get("version") == CVAT_VERSION.lstrip("v"):
                    break
            except (OSError, ValueError):
                pass
            self.stopping.wait(1)
        else:
            raise RuntimeError("CVAT 尚未完成初始化；已保留環境與資料，請重試。")
        self._wait_authorization()

    def _wait_authorization(self):
        # /api/server/about and login work before OPA activates its policy bundle.
        # Creating projects in that interval raises HTTP 500 (missing OPA result).
        # Probe from the server container; OPA stays private to the Compose network.
        self._set("services", "正在等待 CVAT 權限規則載入…", "services")
        script = ("from urllib.request import build_opener, ProxyHandler; "
                  "client=build_opener(ProxyHandler({})); "
                  "response=client.open('http://opa:8181/health?bundles=true', timeout=3); "
                  "response.read()")
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            self._check_closing()
            try:
                result = self._compose("exec", "--no-TTY", "cvat_server", "python", "-c", script,
                                       timeout=10, check=False)
                if result.returncode == 0:
                    return
            except (OSError, RuntimeError):
                pass
            self.stopping.wait(1)
        raise RuntimeError("CVAT 權限規則尚未就緒；已保留環境與資料，請重試啟動。")

    def _login(self):
        # A dedicated local account is persistent; its generated password is never
        # written to disk or command-line arguments. Refresh only when opening anew.
        username = "hub_" + self.project.removeprefix("annotation_cvat_")
        password = secrets.token_urlsafe(36)
        script = ("import json,sys; from django.contrib.auth import get_user_model; "
                  "p=json.load(sys.stdin); u,created=get_user_model().objects.get_or_create(username=p['username']); "
                  "u.is_superuser=True; u.is_staff=True; u.is_active=True; "
                  "u.email='hub@annotation.local'; u.set_password(p['password']); u.save()")
        self._compose("exec", "--no-TTY", "cvat_server", "python", "manage.py", "shell", "-c", script,
                      input_text=json.dumps({"username": username, "password": password}), timeout=90, private=True)
        jar = http.cookiejar.CookieJar()
        client = build_opener(ProxyHandler({}), HTTPCookieProcessor(jar))
        request = Request("http://127.0.0.1:8768/api/auth/login",
                          data=json.dumps({"username": username, "password": password}).encode(),
                          headers={"Content-Type": "application/json", "Host": "cvat.localhost:8768"})
        with client.open(request, timeout=30) as response:
            response.read()
        if not any(cookie.name == "sessionid" for cookie in jar):
            raise RuntimeError("CVAT 本地登入未完成，請重試。")
        return [{"name": cookie.name, "value": cookie.value, "domain": "cvat.localhost", "path": "/",
                 "httpOnly": cookie.name == "sessionid", "secure": False}
                for cookie in jar if cookie.name in ("sessionid", "csrftoken")]

    def launch(self):
        with self.launch_lock:
            self._check_closing()
            if not self.status()["ready"]:
                raise RuntimeError("CVAT 尚未完成準備，請先按一鍵準備／繼續準備。")
            self.launching = True
            try:
                self._set('waiting_docker', '正在準備 Docker 執行環境…', 'docker', started_at=time.time())
                self._ensure_docker()
                self._start_services()
                self._set('account', '服務已回應，正在登入本地帳號…', 'account')
                self._session = self._login()
                self._set("ready", "CVAT 執行中。", "account")
                return {"kind": "web", "url": CVAT_ORIGIN + "/projects", "cookies": [],
                        "auth_cookies": list(self._session)}
            except Exception as error:
                self._set(error.phase if isinstance(error, SetupPause) else "error", str(error))
                self._stop_owned()
                raise
            finally:
                self.launching = False

    def _stop_owned(self):
        if not self.owned_running:
            return
        try:
            # stop, not down -v; no global Docker/WSL shutdown or unrelated project.
            self._compose("stop", "--timeout", "10", timeout=45)
            self.owned_running = False
            self._session = None
        except (OSError, RuntimeError):
            self._set("error", "本中心的 CVAT 服務未成功停止；下次開啟時可重新檢查。")

    def stop(self):
        self.stopping.set()
        # Do not interrupt an elevated installer midway. Its state survives closing.
        if self.launching or self.worker and self.worker.is_alive():
            return
        self._stop_owned()

    shutdown = stop
