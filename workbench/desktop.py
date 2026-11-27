"""Native Windows window and file dialogs around the unified application."""
from __future__ import annotations
import json
import logging
from pathlib import Path
import sys
import threading
import time

from PySide6.QtCore import QByteArray, QEasingCurve, QObject, QEvent, QLockFile, QPropertyAnimation, QTimer, QUrl, Qt, Signal, Slot
from PySide6.QtGui import QAction, QColor, QDesktopServices, QIcon, QKeySequence, QPainter, QPalette, QPixmap
from PySide6.QtNetwork import QNetworkCookie
from PySide6.QtWebChannel import QWebChannel
from PySide6.QtWidgets import (QApplication, QFileDialog, QGraphicsOpacityEffect, QLabel, QMainWindow,
                               QMessageBox, QProgressBar, QStackedWidget, QToolBar, QVBoxLayout, QWidget)
from PySide6.QtWebEngineCore import QWebEnginePage, QWebEngineProfile, QWebEngineScript
from PySide6.QtWebEngineWidgets import QWebEngineView

from .server import APP_ROOT, WorkbenchService
from . import __version__
from .desktop_update import (changed_sources, github_version_status, pull_remote_update,
                             schedule_restart, source_snapshot, validate_sources, missing_runtime_requirements)

STARTUP_BACKGROUND = "#10171e"


class StartupCover(QWidget):
    def __init__(self, view):
        super().__init__(view)
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor(STARTUP_BACKGROUND))
        self.setPalette(palette)
        self.setAutoFillBackground(True)
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.addStretch()
        mark = QLabel("VW", self)
        mark.setAlignment(Qt.AlignmentFlag.AlignCenter)
        mark.setFixedSize(52, 52)
        mark.setStyleSheet(
            "background:#192a32;border:1px solid #33575a;border-radius:8px;"
            "color:#78dfcc;font-size:18px;font-weight:700"
        )
        title = QLabel("Vision Workbench", self)
        title.setStyleSheet("color:#e5edf2;font-size:22px;font-weight:600")
        status = QLabel("正在載入本機工作空間…", self)
        status.setStyleSheet("color:#a7b8c7;font-size:14px")
        for label in (mark, title, status):
            layout.addWidget(label, alignment=Qt.AlignmentFlag.AlignHCenter)
        self.progress = QProgressBar(self)
        self.progress.setAccessibleName("本機工作空間載入進度")
        self.progress.setRange(0, 0)
        self.progress.setTextVisible(False)
        self.progress.setFixedSize(210, 5)
        self.progress.setStyleSheet(
            "QProgressBar {background:#263843;border:0;border-radius:2px;}"
            "QProgressBar::chunk {background:#61c7b5;border-radius:2px;}"
        )
        layout.addWidget(self.progress, alignment=Qt.AlignmentFlag.AlignHCenter)
        layout.addStretch()
        view.installEventFilter(self)
        self.setGeometry(view.rect())
        self.raise_()
        self.show()

    def eventFilter(self, watched, event):
        if watched is self.parentWidget() and event.type() == QEvent.Type.Resize:
            self.setGeometry(watched.rect())
            self.raise_()
        return super().eventFilter(watched, event)

    def dismiss(self, animate=True):
        self.parentWidget().removeEventFilter(self)
        if not animate:
            self.hide()
            return
        effect = QGraphicsOpacityEffect(self)
        effect.setOpacity(1.0)
        self.setGraphicsEffect(effect)
        self.fade = QPropertyAnimation(effect, b"opacity", self)
        self.fade.setDuration(180)
        self.fade.setStartValue(1.0)
        self.fade.setEndValue(0.0)
        self.fade.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.fade.finished.connect(self.hide)
        self.fade.start()


class DialogBridge(QObject):
    requested = Signal(object)

    def __init__(self):
        super().__init__()
        self.parent_window = None
        self.requested.connect(self.choose)

    def __call__(self, kind):
        if kind not in {"images", "folder", "video", "output", "annotation-models", "project-backup"}:
            raise ValueError("檔案選擇器類型無效")
        request = {"kind":kind, "event":threading.Event(), "paths":[]}
        self.requested.emit(request)
        if not request["event"].wait(900):
            raise TimeoutError("檔案選擇已逾時，請重試")
        return request["paths"]

    @Slot(object)
    def choose(self, request):
        try:
            kind = request["kind"]
            folder_titles = {
                "folder": "匯入整份影像資料夾",
                "output": "選擇匯出位置",
                "annotation-models": "選擇已備妥的標註模型資料夾",
            }
            if kind in folder_titles:
                value = QFileDialog.getExistingDirectory(self.parent_window, folder_titles[kind])
                request["paths"] = [value] if value else []
            elif kind == "project-backup":
                value, _ = QFileDialog.getOpenFileName(
                    self.parent_window, "選擇專案備份", "", "Vision Workbench 專案備份 (*.vwb-project.zip)",
                )
                request["paths"] = [value] if value else []
            elif kind == "model":
                value, _ = QFileDialog.getOpenFileName(self.parent_window, "匯入 YOLO／RT-DETR 權重", "", "PyTorch 權重 (*.pt)")
                request["paths"] = [value] if value else []
            elif kind == "video":
                value, _ = QFileDialog.getOpenFileName(self.parent_window,"選擇影片", "", "影片 (*.mp4 *.avi *.mov *.mkv *.webm);;所有檔案 (*)")
                request["paths"] = [value] if value else []
            else:
                values, _ = QFileDialog.getOpenFileNames(self.parent_window,"匯入圖片或標註", "", "影像與標註 (*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff *.json *.jsonl *.txt *.yaml *.yml);;所有檔案 (*)")
                request["paths"] = values
        finally:
            request["event"].set()


class Page(QWebEnginePage):
    def javaScriptConsoleMessage(self, level, message, line, source):
        logging.info("Web %s:%s %s", Path(source).name, line, message)


class FileDropBridge(QObject):
    """Intercept local file URLs before Chromium navigates to a dropped file.

    WebEngine's child renderer receives drag events, so filter its descendants
    as well as the view. Paths come from Qt, not the browser's redacted File API.
    The page checks the active project, drop-zone bounds and busy state again.
    """
    def __init__(self, view):
        super().__init__(view)
        self.view = view
        self.view.setAcceptDrops(True)
        QApplication.instance().installEventFilter(self)

    def eventFilter(self, target, event):
        event_type = event.type()
        if event_type not in (QEvent.Type.DragEnter, QEvent.Type.DragMove,
                              QEvent.Type.DragLeave, QEvent.Type.Drop):
            return False
        ancestor = target
        while ancestor is not None and ancestor is not self.view:
            ancestor = ancestor.parent()
        if ancestor is None:
            return False
        if event_type == QEvent.Type.DragLeave:
            self.view.page().runJavaScript("window.workbenchNativeDrag?.('leave',0,0,[])")
            event.accept()
            return True
        urls = event.mimeData().urls()
        if not self.view.isEnabled() or not urls or any(not url.isLocalFile() for url in urls):
            event.ignore()
            return True
        point = target.mapTo(self.view, event.position().toPoint())
        zoom = self.view.zoomFactor()
        paths = list(dict.fromkeys(url.toLocalFile() for url in urls))
        kind = 'drop' if event_type == QEvent.Type.Drop else 'move'
        args = json.dumps([kind, point.x() / zoom, point.y() / zoom, paths if kind == 'drop' else []])
        self.view.page().runJavaScript(f"window.workbenchNativeDrag?.(...{args})")
        event.setDropAction(Qt.DropAction.CopyAction)
        event.accept()
        return True


class WorkbenchBridge(QObject):
    def __init__(self, window):
        super().__init__(window)
        self.window = window

    @Slot(str, result=bool)
    def openCvat(self, ticket):
        try:
            self.window.open_cvat(self.window.service.consume_cvat_ticket(ticket))
            return True
        except (ValueError, RuntimeError) as error:
            QMessageBox.warning(self.window, "CVAT 無法開啟", str(error))
            return False

    @Slot()
    def showWorkbench(self):
        self.window.show_workbench()

    @Slot(str,str,result=str)
    def openLabelme(self,pid,asset_id):
        try:
            self.window.open_labelme(pid,asset_id)
            return ''
        except Exception as error:
            logging.exception('Labelme could not open')
            return str(error)

    @Slot(str)
    def finishEditorSync(self,error):
        self.window.finish_editor_sync(error)

    @Slot(result=str)
    def updateStatus(self):
        return json.dumps(self.window.desktop_update_status(), ensure_ascii=False)

    @Slot(result=str)
    def applyUpdate(self):
        return json.dumps(self.window.apply_update(), ensure_ascii=False)

    @Slot(result=str)
    def remoteVersionStatus(self):
        self.window.start_remote_version_check()
        return json.dumps(self.window.remote_version_status, ensure_ascii=False)

    @Slot(result=str)
    def checkRemoteVersion(self):
        self.window.start_remote_version_check(force=True)
        return json.dumps(self.window.remote_version_status, ensure_ascii=False)

    @Slot(result=str)
    def applyRemoteUpdate(self):
        return json.dumps(self.window.apply_remote_update(), ensure_ascii=False)


class MainWindow(QMainWindow):
    def __init__(self, service, bridge):
        super().__init__()
        self.service, self.bridge = service, bridge
        bridge.parent_window = self
        self.closing = self.allow_close = False
        self.update_pending = False
        self.source_baseline = source_snapshot(APP_ROOT)
        self.pending_source_changes = []
        self.remote_version_status = {"state": "checking", "message": "正在檢查 GitHub 版本標籤…"}
        self.remote_version_checked_at = 0.0
        self.remote_version_thread = None
        self.remote_version_result = None
        self.remote_update_result = None
        self.remote_update_phase = None
        self.remote_update_thread = None
        self.remote_update_started = 0.0
        self.remote_camera_result = None
        self.camera_shutdown = None
        self.setWindowTitle("Vision Workbench｜影像資料工作台")
        self.resize(1520, 960)
        self.setMinimumSize(1000, 680)
        self.setObjectName("workbenchDesktop")
        self.setStyleSheet(
            f"#workbenchDesktop, #workbenchDesktopStack {{ background-color: {STARTUP_BACKGROUND}; }}"
        )
        self.fullscreen_action = QAction("切換全螢幕", self)
        self.fullscreen_action.setShortcut(QKeySequence("F11"))
        self.fullscreen_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        self.fullscreen_action.triggered.connect(self.toggle_fullscreen)
        self.addAction(self.fullscreen_action)
        self.view = QWebEngineView(self)
        self.profile = QWebEngineProfile(self)
        self.profile.setHttpCacheType(QWebEngineProfile.HttpCacheType.MemoryHttpCache)
        self.profile.downloadRequested.connect(self.save_download)
        self.page = Page(self.profile, self.view)
        self.page.setBackgroundColor(QColor(STARTUP_BACKGROUND))
        self.view.setPage(self.page)
        self.file_drop_bridge = FileDropBridge(self.view)
        self.channel = QWebChannel(self.page)
        self.native_bridge = WorkbenchBridge(self)
        self.channel.registerObject("workbenchNative", self.native_bridge)
        self.page.setWebChannel(self.channel)
        self.stack = QStackedWidget(self)
        self.stack.setObjectName("workbenchDesktopStack")
        self.stack.addWidget(self.view)
        self.setCentralWidget(self.stack)
        self.startup_cover = StartupCover(self.view)
        self.startup_pending = True
        self.startup_has_loaded = False
        self.startup_timeout = QTimer(self)
        self.startup_timeout.setSingleShot(True)
        self.startup_timeout.setInterval(12000)
        self.startup_timeout.timeout.connect(self.startup_timed_out)
        self.cvat_view = None
        self.cvat_profile = None
        self.cvat_url = None
        self.labelme_editor=None
        self.editor_transition=None
        self.external_mode=None
        self.cvat_context={}
        self.cvat_toolbar = QToolBar("專案標註編輯器", self)
        self.cvat_toolbar.setMovable(False)
        for title,target in [('內建編輯器','builtin'),('Labelme','labelme'),('CVAT','cvat'),('儲存並前往審核','review')]:
            action=QAction(title,self)
            action.triggered.connect(lambda checked=False,t=target:self.transition_editor(t))
            self.cvat_toolbar.addAction(action)
        self.editor_status=QLabel('切換時自動儲存並同步專案')
        self.cvat_toolbar.addSeparator();self.cvat_toolbar.addWidget(self.editor_status)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self.cvat_toolbar)
        self.cvat_toolbar.hide()
        self.view.loadFinished.connect(self.loaded)
        pixmap = QPixmap(64, 64)
        pixmap.fill(QColor("#172531"))
        painter = QPainter(pixmap)
        painter.setPen(QColor("#45dac2"))
        painter.drawRect(12, 12, 39, 39)
        painter.drawLine(20, 31, 29, 42)
        painter.drawLine(29, 42, 45, 22)
        painter.end()
        self.setWindowIcon(QIcon(pixmap))
        self.update_timer = QTimer(self)
        self.update_timer.setInterval(2500)
        self.update_timer.timeout.connect(self.check_update_indicator)
        self.update_timer.timeout.connect(self.poll_remote_version_check)
        self.update_timer.timeout.connect(self.poll_remote_update)
        self.startup_timeout.start()
        self.view.setUrl(QUrl(service.entry_url))

    def start_remote_version_check(self, force=False):
        if self.remote_version_thread and self.remote_version_thread.is_alive():
            return
        if self.remote_version_result is not None:
            self.poll_remote_version_check()
        if not force and time.monotonic() - self.remote_version_checked_at < 600:
            return
        self.remote_version_status = {"state": "checking", "message": "正在檢查 GitHub 版本標籤…"}
        self.page.runJavaScript(f"window.workbenchRemoteVersionStatus?.({json.dumps(self.remote_version_status, ensure_ascii=False)})")

        def check():
            self.remote_version_result = github_version_status(APP_ROOT, __version__)

        self.remote_version_thread = threading.Thread(target=check, daemon=True)
        self.remote_version_thread.start()

    def poll_remote_version_check(self):
        if self.remote_version_result is not None:
            self.remote_version_status = self.remote_version_result
            self.remote_version_result = None
            self.remote_version_checked_at = time.monotonic()
            self.page.runJavaScript(f"window.workbenchRemoteVersionStatus?.({json.dumps(self.remote_version_status, ensure_ascii=False)})")
        elif time.monotonic() - self.remote_version_checked_at >= 600:
            self.start_remote_version_check()

    def apply_remote_update(self):
        if self.update_pending or self.closing:
            return self.remote_update_result or {"state": "updating", "message": "更新已在進行中，請稍候。"}
        self.update_pending = True
        self.remote_update_phase = "waiting"
        self.remote_update_started = time.monotonic()
        self.set_remote_update_status({"state": "waiting", "message": "正在確認目前工作，完成儲存後自動更新並重新啟動…"})
        self.prepare_remote_update()
        return self.remote_update_result

    def set_remote_update_status(self, result):
        self.remote_update_result = result
        self.page.runJavaScript(f"window.workbenchRemoteUpdateStatus?.({json.dumps(result, ensure_ascii=False)})")

    def remote_update_busy(self):
        return bool(self.service.jobs.active() or self.service.training.active_runs()
                    or self.service._cvat is not None and self.service._cvat.status().get("busy"))

    def prepare_remote_update(self):
        if self.remote_update_phase != "waiting":
            return
        try:
            if self.remote_update_busy() or self.editor_transition:
                self.set_remote_update_status({"state": "waiting", "message": "已排入更新，等待匯入、AI、訓練或編輯器同步完成後自動繼續。"})
                QTimer.singleShot(500, self.prepare_remote_update)
                return
            if self.external_mode:
                self.remote_update_phase = "editor"
                self.remote_update_started = time.monotonic()
                self.set_remote_update_status({"state": "saving", "message": "正在儲存並同步標註編輯器，完成後自動更新…"})
                self.transition_editor("update")
                return
            self.remote_update_phase = "saving"
            self.remote_update_started = time.monotonic()
            self.set_remote_update_status({"state": "saving", "message": "正在保存專案與介面狀態…"})
            self.page.runJavaScript("""window.__workbenchRemoteUpdate={state:'saving'};
                Promise.resolve().then(()=>{
                    if(!window.workbenchState || !window.workbenchFlush)throw Error('介面尚未就緒，無法確認存檔。');
                    const current=window.workbenchState();
                    if(current.transitioning || current.loading)return false;
                    return Promise.resolve(window.workbenchFlush()).then(saved=>{
                        if(saved!==true)throw Error('尚未完成儲存，請稍後重試。');
                        const after=window.workbenchState();
                        if(after.dirty)throw Error('仍有尚未完成的編輯，請先完成並儲存目前工作。');
                        return !(after.busy || after.transitioning || after.loading || after.saving);
                    });
                }).then(ready=>window.__workbenchRemoteUpdate={state:ready?'ready':'waiting'})
                .catch(e=>window.__workbenchRemoteUpdate={state:'error',message:String(e.message||e)});""")
            QTimer.singleShot(120, self.poll_remote_update_save)
        except Exception as error:
            logging.exception("Could not prepare GitHub update")
            self.finish_remote_update({"state": "error", "message": "更新尚未開始；目前工作已保留。" + str(error)})

    def poll_remote_update_save(self):
        if self.remote_update_phase != "saving":
            return
        if time.monotonic() - self.remote_update_started > 120:
            self.finish_remote_update({"state": "error", "message": "等待儲存逾時；目前工作已保留，請稍後重試。"})
            return
        self.page.runJavaScript("JSON.stringify(window.__workbenchRemoteUpdate||{state:'error',message:'介面尚未就緒'})", self.finish_remote_update_save)

    def finish_remote_update_save(self, value):
        if self.remote_update_phase != "saving":
            return
        try:
            saved = json.loads(value or "{}")
            if not isinstance(saved, dict):
                raise ValueError("無法確認儲存狀態")
            if saved.get("state") == "saving":
                QTimer.singleShot(150, self.poll_remote_update_save)
                return
            if saved.get("state") == "waiting" or saved.get("state") == "ready" and self.remote_update_busy():
                self.remote_update_phase = "waiting"
                self.prepare_remote_update()
                return
            if saved.get("state") != "ready":
                self.finish_remote_update({"state": "blocked", "message": saved.get("message") or "請先完成並儲存目前工作。"})
                return
            if self.external_mode or self.editor_transition:
                self.remote_update_phase = "waiting"
                self.prepare_remote_update()
                return
            # Saving is complete. Keep the whole editor disabled until restart or a retryable failure.
            self.view.setEnabled(False)
            self.cvat_toolbar.setEnabled(False)
            camera = getattr(self.service, "_camera", None)
            if camera is not None:
                snapshot = camera.status()
                if snapshot.get("state") in {"starting", "stopping", "running"} or snapshot.get("running") or snapshot.get("recording"):
                    self.stop_remote_update_camera(camera)
                    return
            self.start_remote_update()
        except Exception as error:
            logging.exception("Could not start GitHub update")
            self.finish_remote_update({"state": "error", "message": "更新尚未開始；目前工作已保留。" + str(error)})

    def stop_remote_update_camera(self, camera):
        self.remote_update_phase = "camera"
        self.remote_update_started = time.monotonic()
        self.remote_camera_result = None
        self.set_remote_update_status({"state": "saving", "message": "正在完成錄影並停止相機，確認檔案保存後自動更新…"})

        def stop_camera():
            try:
                if camera.status().get("recording"):
                    saved = camera.stop_recording()
                    recording = saved.get("last_recording") or {}
                    if saved.get("recording") or saved.get("error") or not recording.get("path") or recording.get("frames", 0) <= 0:
                        raise RuntimeError(saved.get("error") or "錄影檔案尚未完成，暫存內容已保留。")
                result = camera.stop()
                if not isinstance(result, dict):
                    raise ValueError("無法確認相機停止狀態")
                self.remote_camera_result = result
            except Exception as error:
                logging.exception("Could not safely stop camera before update")
                self.remote_camera_result = {"state": "error", "error": str(error)}

        threading.Thread(target=stop_camera, daemon=True, name="workbench-update-camera").start()
        QTimer.singleShot(100, self.poll_remote_update_camera)

    def poll_remote_update_camera(self):
        if self.remote_update_phase != "camera":
            return
        try:
            result = self.remote_camera_result
            if result is None:
                if time.monotonic() - self.remote_update_started > 30:
                    raise RuntimeError("等待相機停止逾時；錄影與目前工作已保留，請稍後重試。")
                QTimer.singleShot(100, self.poll_remote_update_camera)
                return
            if result.get("state") in {"starting", "stopping", "running"} or result.get("running") or result.get("recording"):
                if time.monotonic() - self.remote_update_started > 30:
                    raise RuntimeError("相機驅動仍在回應；更新尚未開始，請稍後重試。")
                self.remote_camera_result = self.service._camera.status()
                QTimer.singleShot(250, self.poll_remote_update_camera)
                return
            if result.get("error") or result.get("state") != "stopped":
                raise RuntimeError(result.get("error") or "無法確認相機與錄影已安全停止。")
            if self.remote_update_busy():
                self.view.setEnabled(True)
                self.cvat_toolbar.setEnabled(True)
                self.remote_update_phase = "waiting"
                self.prepare_remote_update()
                return
            self.start_remote_update()
        except Exception as error:
            logging.exception("Camera update handoff failed")
            self.finish_remote_update({"state": "blocked", "message": "更新尚未開始；相機錄影與目前工作已保留。" + str(error)})

    def start_remote_update(self):
        self.remote_update_phase = "updating"
        self.remote_update_started = time.monotonic()
        self.set_remote_update_status({"state": "updating", "message": "內容已儲存，正在備份本機修改並從 GitHub 更新…"})

        def update():
            try:
                result = pull_remote_update(
                    APP_ROOT, progress=lambda message: setattr(self, "remote_update_result", {"state": "updating", "message": str(message)}))
                if not isinstance(result, dict):
                    raise ValueError("GitHub 更新服務回應格式無效")
                self.remote_update_result = result
            except Exception as error:
                logging.exception("GitHub update failed")
                self.remote_update_result = {"state": "error", "message": "GitHub 更新未完成；目前視窗與工作內容已保留。" + str(error)}

        self.remote_update_thread = threading.Thread(target=update, daemon=True, name="workbench-update")
        self.remote_update_thread.start()
        QTimer.singleShot(100, self.poll_remote_update)

    def finish_remote_update(self, result):
        self.remote_update_phase = None
        self.update_pending = False
        self.view.setEnabled(True)
        self.cvat_toolbar.setEnabled(True)
        self.set_remote_update_status(result)

    def poll_remote_update(self):
        if self.remote_update_phase == "editor" and time.monotonic() - self.remote_update_started > 120:
            self.finish_editor_sync("等待標註儲存逾時；原內容已保留，請稍後重試。")
            return
        if self.remote_update_phase != "updating":
            return
        result = self.remote_update_result
        if not result or result.get("state") in {"updating", "checking"}:
            if result:
                self.page.runJavaScript(f"window.workbenchRemoteUpdateStatus?.({json.dumps(result, ensure_ascii=False)})")
            return
        try:
            retry_restart = result.get("state") in {"current", "up_to_date"} and changed_sources(self.source_baseline, source_snapshot(APP_ROOT))
            if result.get("state") == "updated" or retry_restart:
                # Any pulled commit requires a restart, including changes only to tests or docs.
                validate_sources(APP_ROOT)
                schedule_restart(APP_ROOT, data_root=self.service.data_root)
                self.remote_update_phase = "restarting"
                self.set_remote_update_status({**result, "state": "restarting", "message": "更新已驗證，正在自動重新啟動工作台…"})
                self.allow_close = True
                QTimer.singleShot(250, QApplication.closeAllWindows)
            else:
                self.finish_remote_update(result)
        except Exception as error:
            logging.exception("Could not restart updated workbench")
            self.finish_remote_update({**result, "state": "error", "message": "程式已更新，重新啟動尚未完成；目前視窗與工作內容已保留，請重試。" + str(error)})

    def toggle_fullscreen(self):
        """Switch the native workbench between full-screen and windowed mode."""
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def open_cvat(self, result):
        url = result.get("url")
        if not isinstance(url, str) or not url.startswith("http://cvat.localhost:8768/"):
            raise ValueError("CVAT 位址無效。")
        if self.cvat_view is None:
            self.cvat_profile = QWebEngineProfile("vision-workbench-cvat", self)
            profile_root = self.service.data_root / "browser-cvat"
            self.cvat_profile.setPersistentStoragePath(str(profile_root / "storage"))
            self.cvat_profile.setCachePath(str(profile_root / "cache"))
            self.cvat_view = QWebEngineView(self.stack)
            self.cvat_view.setPage(Page(self.cvat_profile, self.cvat_view))
            script=QWebEngineScript();script.setName('vision-workbench-sync')
            script.setInjectionPoint(QWebEngineScript.InjectionPoint.DocumentReady)
            script.setWorldId(QWebEngineScript.ScriptWorldId.MainWorld)
            script.setSourceCode((APP_ROOT/'web'/'cvat-host.js').read_text('utf-8'))
            self.cvat_view.page().scripts().insert(script)
            self.stack.addWidget(self.cvat_view)
        destination = QUrl(url)
        store = self.cvat_profile.cookieStore()
        for value in result.get("auth_cookies", []):
            cookie = QNetworkCookie(QByteArray(value["name"].encode()), QByteArray(value["value"].encode()))
            cookie.setDomain(value.get("domain", "cvat.localhost"))
            cookie.setPath(value.get("path", "/"))
            cookie.setHttpOnly(value.get("httpOnly", value['name']=='sessionid'))
            cookie.setSecure(value.get("secure", False))
            store.setCookie(cookie, destination)
        self.cvat_url = url
        self.cvat_context=result
        self.external_mode='cvat'
        self.editor_status.setText('CVAT · 切換時自動儲存並同步專案')
        QTimer.singleShot(120, lambda: self.cvat_view.setUrl(destination))
        self.stack.setCurrentWidget(self.cvat_view)
        self.cvat_toolbar.show()

    def show_workbench(self):
        self.transition_editor('builtin')

    def open_labelme(self,pid,asset_id):
        from .labelme_bridge import create_labelme_editor
        editor=create_labelme_editor(self.service.store,pid,self.service.data_root,asset_id)
        if self.labelme_editor:
            self.stack.removeWidget(self.labelme_editor);self.labelme_editor.deleteLater()
        self.labelme_editor=editor;self.stack.addWidget(editor)
        self.stack.setCurrentWidget(editor);self.external_mode='labelme'
        self.editor_status.setText('Labelme · 切換時自動儲存並同步專案')
        self.cvat_toolbar.show()

    def transition_editor(self,target):
        if self.editor_transition or not self.external_mode:return
        source=self.external_mode
        self.editor_transition={'source':source,'target':target,'started':time.monotonic()}
        self.cvat_toolbar.setEnabled(False)
        self.editor_status.setText('正在儲存並同步專案…')
        if source=='labelme':
            if not self.labelme_editor.flush():
                self.finish_editor_sync(self.labelme_editor.last_sync_error);return
            self.editor_transition['asset_id']=(self.labelme_editor.current_asset or {}).get('id')
            self.labelme_editor.setEnabled(False)
            self.request_editor_sync()
        else:
            self.cvat_view.page().runJavaScript(
                f'window.workbenchCvatHost?.save({int(self.cvat_context["job_id"])})')
            self.cvat_view.setEnabled(False)
            QTimer.singleShot(100,self.poll_cvat_save)

    def poll_cvat_save(self):
        self.cvat_view.page().runJavaScript(
            "JSON.stringify(window.workbenchCvatHost||{state:'error',message:'CVAT 尚未載入完成'})",self.cvat_save_state)

    def cvat_save_state(self,value):
        if not self.editor_transition:return
        try:state=json.loads(value or '{}')
        except ValueError:state={}
        if state.get('state')=='saving' and time.monotonic()-self.editor_transition['started']<120:
            QTimer.singleShot(150,self.poll_cvat_save);return
        if state.get('state')!='saved':
            self.finish_editor_sync(state.get('message') or 'CVAT 儲存尚未完成，請稍後重試。');return
        frames=self.cvat_context.get('asset_ids',[]);frame=state.get('frame',0)
        if 0<=frame<len(frames):self.editor_transition['asset_id']=frames[frame]
        self.request_editor_sync()

    def request_editor_sync(self):
        request={k:v for k,v in self.editor_transition.items() if k!='started'}
        self.page.runJavaScript(f'window.workbenchExternalSync?.({json.dumps(request)})')

    def finish_editor_sync(self,error):
        transition=self.editor_transition
        if not transition:return
        self.editor_transition=None;self.cvat_toolbar.setEnabled(True)
        if self.cvat_view:self.cvat_view.setEnabled(True)
        if self.labelme_editor:self.labelme_editor.setEnabled(True)
        if error:
            self.editor_status.setText('尚未同步 · 內容已保留，請修正後重試')
            if transition['target']=='update':
                self.finish_remote_update({"state":"blocked","message":"標註尚未同步；原內容已保留。"+str(error)})
            QMessageBox.warning(self,'標註尚未同步',error);return
        self.external_mode=None
        self.stack.setCurrentWidget(self.view);self.cvat_toolbar.hide()
        if transition['source']=='cvat':self.cvat_view.setUrl(QUrl('about:blank'))
        if transition['target']=='update':
            self.remote_update_phase='waiting'
            self.prepare_remote_update()
        elif transition['target']=='close':QTimer.singleShot(0,self.close)
        else:self.page.runJavaScript(f'window.workbenchExternalNavigate?.({json.dumps(transition["target"])})')

    def check_update_indicator(self):
        try:
            self.pending_source_changes = changed_sources(self.source_baseline, source_snapshot(APP_ROOT))
        except OSError:
            self.page.runJavaScript("window.workbenchUpdateStatus?.(null)")
            return None
        self.page.runJavaScript(f"window.workbenchUpdateStatus?.({len(self.pending_source_changes)})")
        return list(self.pending_source_changes)

    def desktop_update_status(self):
        if self.update_pending:
            return {"state":"updating", "count":len(self.pending_source_changes),
                    "changes":list(self.pending_source_changes), "blockers":[],
                    "message":"正在保存工作內容並準備重新啟動。"}
        changes=self.check_update_indicator()
        if changes is None:
            return {"state":"unavailable", "count":None, "changes":[], "blockers":[],
                    "message":"暫時無法讀取程式檔案，請稍後再試。"}
        if not changes:
            return {"state":"current", "count":0, "changes":[], "blockers":[],
                    "message":"目前執行中的程式與本機檔案一致。"}
        blockers=[]
        if any(name.startswith("requirements") for name in changes):
            missing=missing_runtime_requirements(APP_ROOT)
            if missing: blockers.append("需要先執行 bootstrap.ps1 安裝："+"、".join(missing))
        if self.external_mode:
            blockers.append("請先從 CVAT 或 Labelme 儲存並返回工作台。")
        if self.service.jobs.active() or self.service.training.active_runs() or self.service._cvat is not None and self.service._cvat.status().get("busy"):
            blockers.append("請等待匯入、AI、訓練、CVAT 同步或元件安裝工作完成。")
        return {"state":"blocked" if blockers else "available", "count":len(changes),
                "changes":changes, "blockers":blockers,
                "message":f"偵測到 {len(changes)} 個程式檔案有新修改。"}

    def notify_update_progress(self, state, message):
        payload=json.dumps({"state":state,"message":message,"count":len(self.pending_source_changes)},ensure_ascii=False)
        self.page.runJavaScript(f"window.workbenchUpdateProgress?.({payload})")

    def apply_update(self):
        if self.update_pending or self.closing:
            return {"state":"updating", "message":"更新已在進行中，請稍候。"}
        update=self.desktop_update_status()
        if update["state"] != "available":
            return update
        changes=self.check_update_indicator()
        self.update_pending=True
        self.notify_update_progress("saving", "正在保存專案與介面狀態…")
        self.page.runJavaScript("""window.__workbenchUpdate={state:'saving'};
            Promise.resolve().then(()=>window.workbenchFlush()).then(()=>window.__workbenchUpdate={state:'ready'})
            .catch(e=>window.__workbenchUpdate={state:'error',message:String(e.message||e)});""")
        QTimer.singleShot(120,self.poll_update)
        return {"state":"updating", "count":len(changes), "changes":changes,
                "message":"正在保存專案與介面狀態…"}

    def poll_update(self):
        self.page.runJavaScript("JSON.stringify(window.__workbenchUpdate||{state:'error',message:'介面尚未就緒'})",self.finish_update)

    def finish_update(self, value):
        try:state=json.loads(value or "{}")
        except ValueError:state={"state":"error","message":"無法確認儲存狀態"}
        if state.get("state")=="saving":QTimer.singleShot(150,self.poll_update);return
        if state.get("state")!="ready":
            self.update_pending=False
            self.notify_update_progress("error",state.get("message","請先完成並儲存目前工作。"));return
        try:
            changes=changed_sources(self.source_baseline,source_snapshot(APP_ROOT))
            if not changes:
                self.update_pending=False
                self.notify_update_progress("current","目前執行中的程式與本機檔案一致。");return
            validate_sources(APP_ROOT);schedule_restart(APP_ROOT,data_root=self.service.data_root)
            self.notify_update_progress("restarting","更新已驗證，工作台正在重新啟動…")
            self.allow_close=True;QTimer.singleShot(250,QApplication.closeAllWindows)
        except (OSError,SyntaxError,ValueError) as error:
            self.update_pending=False
            self.notify_update_progress("error","更新未完成；目前視窗會繼續保留。"+str(error))

    def save_download(self, download):
        suggested = Path(download.downloadFileName()).name
        if suggested.startswith('M') and suggested.endswith(('.json', '.csv', '.png')):
            filename, _ = QFileDialog.getSaveFileName(self, "儲存模型評估報告", str(Path.home() / "Downloads" / suggested))
            if not filename:
                download.cancel()
                return
            destination = Path(filename)
            download.setDownloadDirectory(str(destination.parent))
            download.setDownloadFileName(destination.name)
            download.accept()
            return
        # The editor offers a recovery copy if an external revision conflicts.
        folder = self.service.data_root / "recovery"
        folder.mkdir(parents=True,exist_ok=True)
        import uuid
        filename = "annotations-" + uuid.uuid4().hex[:12] + ".json"
        download.setDownloadDirectory(str(folder))
        download.setDownloadFileName(filename)
        download.accept()
        logging.info("Editor recovery download: %s", folder / filename)

    def loaded(self, ok):
        if not ok:
            self.finish_startup(animate=False)
            QMessageBox.critical(self, "介面載入失敗", "請檢查 data/logs/workbench.log 後重新啟動。專案資料仍保留在本機。")
        else:
            self.startup_has_loaded = True
            if self.startup_pending:
                self.startup_cover.raise_()
                self.poll_startup()
            self.check_update_indicator()
            self.start_remote_version_check()
            # ES modules may finish installing the callback just after loadFinished.
            # Retry once, then keep the indicator current in every desktop session.
            QTimer.singleShot(400, self.check_update_indicator)
            self.update_timer.start()

    def poll_startup(self):
        if self.startup_pending:
            self.page.runJavaScript("window.workbenchStartupSettled === true", self.startup_checked)

    def startup_checked(self, settled):
        if not self.startup_pending:
            return
        if settled is True:
            self.finish_startup()
        else:
            QTimer.singleShot(80, self.poll_startup)

    def startup_timed_out(self):
        logging.warning("Desktop startup did not report that the interface settled; revealing the page.")
        self.finish_startup()

    def finish_startup(self, animate=True):
        if not self.startup_pending:
            return
        self.startup_pending = False
        self.startup_timeout.stop()
        self.startup_cover.dismiss(animate)

    def closeEvent(self, event):
        if self.allow_close:
            event.accept()
            return
        event.ignore()
        if self.update_pending:
            return
        if self.external_mode:
            self.transition_editor('close');return
        if self.closing:
            return
        if self.service.jobs.active():
            QMessageBox.information(self,"工作仍在執行","請等待匯入、AI 或匯出完成後再關閉，確保資料完整保存。")
            return
        if self.startup_pending or not self.startup_has_loaded:
            self.finish_startup(animate=False)
            self.finish_close()
            return
        self.closing = True
        self.view.setEnabled(False)
        self.page.runJavaScript("""window.__workbenchClose = {state:'saving'};
            Promise.resolve().then(()=>{if(!window.workbenchFlush) throw Error('介面尚未就緒，無法確認存檔。'); return window.workbenchFlush();})
            .then(()=>window.__workbenchClose={state:'ready'})
            .catch(e=>window.__workbenchClose={state:'error',message:String(e.message||e)});""")
        QTimer.singleShot(100, self.poll_close)

    def poll_close(self):
        self.page.runJavaScript("JSON.stringify(window.__workbenchClose||{state:'ready'})", self.close_state)

    def close_state(self, value):
        try:
            state = json.loads(value or '{"state":"ready"}')
        except ValueError:
            state = {"state":"error","message":"無法確認存檔，請先手動儲存"}
        if state["state"] == "saving":
            QTimer.singleShot(150, self.poll_close)
        elif state["state"] == "error":
            self.closing = False
            self.view.setEnabled(True)
            QMessageBox.warning(self,"尚未完成儲存",state.get("message","請先儲存工作"))
        else:
            if self.service.jobs.active():
                self.closing = False
                self.view.setEnabled(True)
                QMessageBox.information(self,"工作仍在執行","請等待背景工作完成後再關閉。")
                return
            if self.service._camera:
                self.camera_shutdown = None
                def stop_camera():
                    try:
                        self.camera_shutdown = self.service.camera.stop()
                    except Exception as exc:
                        self.camera_shutdown = {"state":"error", "error":str(exc)}
                threading.Thread(target=stop_camera,daemon=True,name="workbench-close-camera").start()
                QTimer.singleShot(100,self.poll_camera_close)
            else:
                self.finish_close()

    def poll_camera_close(self):
        result = self.camera_shutdown
        if result is None:
            QTimer.singleShot(100,self.poll_camera_close)
            return
        if result.get("state") in {"starting", "stopping", "running"} or result.get("recording"):
            self.closing = False
            self.view.setEnabled(True)
            QMessageBox.warning(self,"相機尚未停止","相機驅動仍在回應，請等待相機停止並完成錄影檔後再次關閉。")
            return
        if result.get("error"):
            self.closing = False
            self.view.setEnabled(True)
            answer = QMessageBox.question(self,"相機停止時發生錯誤",str(result["error"])+"\n\n請保留 data/incoming 中的錄影與日誌供檢查。仍要關閉視窗嗎？")
            if answer != QMessageBox.StandardButton.Yes:
                return
        self.finish_close()

    def finish_close(self):
        if self.service.jobs.active():
            self.closing = False
            self.view.setEnabled(True)
            return
        self.allow_close = True
        self.close()


def run_desktop(data_root=None):
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setApplicationName("Vision Workbench")
    root = Path(data_root or APP_ROOT / "data").resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = QLockFile(str(root / "desktop.lock"))
    lock.setStaleLockTime(0)
    if not lock.tryLock(0):
        QMessageBox.information(None,"Vision Workbench 已開啟","請切換至已開啟的影像資料工作台。")
        return 0
    bridge = DialogBridge()
    service = WorkbenchService(root, dialog=bridge).start()
    window = MainWindow(service, bridge)
    window.showFullScreen()
    try:
        return app.exec()
    finally:
        service.close()
        lock.unlock()
