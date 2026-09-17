"""Native Windows window and file dialogs around the unified application."""
from __future__ import annotations
import json
import logging
from pathlib import Path
import sys
import threading
import time

from PySide6.QtCore import QByteArray, QObject, QEvent, QLockFile, QTimer, QUrl, Qt, Signal, Slot
from PySide6.QtGui import QAction, QColor, QDesktopServices, QIcon, QKeySequence, QPainter, QPixmap
from PySide6.QtNetwork import QNetworkCookie
from PySide6.QtWebChannel import QWebChannel
from PySide6.QtWidgets import (QApplication, QFileDialog, QLabel, QMainWindow,
                               QMessageBox, QStackedWidget, QToolBar)
from PySide6.QtWebEngineCore import QWebEnginePage, QWebEngineProfile, QWebEngineScript
from PySide6.QtWebEngineWidgets import QWebEngineView

from .server import APP_ROOT, WorkbenchService
from .desktop_update import changed_sources, schedule_restart, source_snapshot, validate_sources, missing_runtime_requirements


class DialogBridge(QObject):
    requested = Signal(object)

    def __init__(self):
        super().__init__()
        self.parent_window = None
        self.requested.connect(self.choose)

    def __call__(self, kind):
        if kind not in {"images", "folder", "video", "output"}:
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
            if kind in {"folder", "output"}:
                value = QFileDialog.getExistingDirectory(self.parent_window, "選擇匯出位置" if kind == "output" else "匯入整份影像資料夾")
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


class MainWindow(QMainWindow):
    def __init__(self, service, bridge):
        super().__init__()
        self.service, self.bridge = service, bridge
        bridge.parent_window = self
        self.closing = self.allow_close = False
        self.update_pending = False
        self.source_baseline = source_snapshot(APP_ROOT)
        self.pending_source_changes = []
        self.camera_shutdown = None
        self.setWindowTitle("Vision Workbench｜影像資料工作台")
        self.resize(1520, 960)
        self.setMinimumSize(1000, 680)
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
        self.view.setPage(self.page)
        self.file_drop_bridge = FileDropBridge(self.view)
        self.channel = QWebChannel(self.page)
        self.native_bridge = WorkbenchBridge(self)
        self.channel.registerObject("workbenchNative", self.native_bridge)
        self.page.setWebChannel(self.channel)
        self.stack = QStackedWidget(self)
        self.stack.addWidget(self.view)
        self.setCentralWidget(self.stack)
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
        self.view.setUrl(QUrl(service.entry_url))
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
        self.update_timer.setInterval(10000)
        self.update_timer.timeout.connect(self.check_update_indicator)
        if __import__('os').environ.get('VISION_WORKBENCH_DEVELOPMENT') == '1':
            self.update_timer.start()

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
            QMessageBox.warning(self,'標註尚未同步',error);return
        self.external_mode=None
        self.stack.setCurrentWidget(self.view);self.cvat_toolbar.hide()
        if transition['source']=='cvat':self.cvat_view.setUrl(QUrl('about:blank'))
        if transition['target']=='close':QTimer.singleShot(0,self.close)
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
                    "message":"目前執行中的程式已是最新狀態。"}
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
                self.notify_update_progress("current","目前執行中的程式已是最新狀態。");return
            validate_sources(APP_ROOT);schedule_restart(APP_ROOT)
            self.notify_update_progress("restarting","更新已驗證，工作台正在重新啟動…")
            self.allow_close=True;QTimer.singleShot(250,QApplication.closeAllWindows)
        except (OSError,SyntaxError,ValueError) as error:
            self.update_pending=False
            self.notify_update_progress("error","更新未完成；目前視窗會繼續保留。"+str(error))

    def save_download(self, download):
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
            QMessageBox.critical(self, "介面載入失敗", "請檢查 data/logs/workbench.log 後重新啟動。專案資料仍保留在本機。")
        else:
            self.check_update_indicator()

    def closeEvent(self, event):
        if self.allow_close:
            event.accept()
            return
        event.ignore()
        if self.external_mode:
            self.transition_editor('close');return
        if self.closing:
            return
        if self.service.jobs.active():
            QMessageBox.information(self,"工作仍在執行","請等待匯入、AI 或匯出完成後再關閉，確保資料完整保存。")
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
