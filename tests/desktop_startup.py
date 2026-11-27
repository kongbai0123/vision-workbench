"""Exercise the real desktop entry, authenticated UI and orderly close in isolation."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import sys
import tempfile
import threading
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('QTWEBENGINE_CHROMIUM_FLAGS', '--disable-gpu --disable-background-timer-throttling')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

import main as entry
from workbench import desktop
from workbench.server import Handler
from workbench.store import ProjectStore


def main():
    output = ROOT / 'qa-output' / 'startup'
    output.mkdir(parents=True, exist_ok=True)
    result = {
        'passed': False, 'dark_page_background': False, 'cover_initially_shown': False,
        'cover_during_initialization': False, 'cover_dark': False, 'cover_released': False,
        'cover_progress_indeterminate': False, 'ui_ready': False, 'authenticated_api': False,
        'project_progress_indeterminate': False, 'project_ready': False, 'closed': False,
    }
    windows = []
    system_requested = threading.Event()
    release_system = threading.Event()
    project_requested = threading.Event()
    release_project = threading.Event()
    original_get = Handler.do_GET

    def get_with_startup_gate(handler):
        if handler.path == '/api/system' and not release_system.is_set():
            system_requested.set()
            release_system.wait(15)
        if handler.path == f'/api/projects/{project_id}' and not release_project.is_set():
            project_requested.set()
            release_project.wait(15)
        return original_get(handler)

    class StartupWindow(desktop.MainWindow):
        def __init__(self, service, bridge):
            super().__init__(service, bridge)
            windows.append(self)
            result['dark_page_background'] = self.page.backgroundColor().name() == '#10171e'
            result['cover_initially_shown'] = self.startup_pending and not self.startup_cover.isHidden()
            QTimer.singleShot(75, self.inspect_initial_cover)
            QTimer.singleShot(30000, self.timeout)

        def inspect_initial_cover(self):
            if not system_requested.is_set():
                QTimer.singleShot(75, self.inspect_initial_cover)
                return
            result['cover_during_initialization'] = self.startup_pending and self.startup_cover.isVisible()
            progress = self.startup_cover.progress
            result['cover_progress_indeterminate'] = progress.isVisible() and progress.minimum() == progress.maximum() == 0
            picture = self.startup_cover.grab()
            if not picture.isNull():
                color = picture.toImage().pixelColor(1, 1)
                result['cover_dark'] = max(color.red(), color.green(), color.blue()) < 64
                picture.save(str(output / 'loading.png'))
            release_system.set()
            self.inspect()

        def inspect(self):
            if result['ui_ready']:
                return
            self.page.runJavaScript("""JSON.stringify((()=>{
                const ready=document.getElementById('workspaceLoading')?.dataset.startup==='settled' &&
                    typeof window.workbenchState==='function' && typeof window.workbenchFlush==='function';
                if(ready && !window.__startupApi){
                    window.__startupApi='pending';
                    fetch('/api/system').then(async r=>r.ok ? r.json() : null)
                        .then(v=>window.__startupApi=v?.name==='Vision Workbench' ? 'ok' : 'failed')
                        .catch(()=>window.__startupApi='failed');
                }
                return {ready,api:window.__startupApi};
            })())""", self.inspected)

        def inspected(self, value):
            state = json.loads(value or '{}')
            if state.get('ready') and state.get('api') == 'ok' and not self.startup_pending and self.startup_cover.isHidden():
                result['cover_released'] = True
                result['ui_ready'] = True
                result['authenticated_api'] = True
                self.view.grab().save(str(output / 'desktop.png'))
                self.page.runJavaScript("document.querySelector('.project-card-open').click()")
                self.inspect_project_loading()
            else:
                QTimer.singleShot(150, self.inspect)

        def inspect_project_loading(self):
            if not project_requested.is_set():
                QTimer.singleShot(75, self.inspect_project_loading)
                return
            self.page.runJavaScript("""(() => {
                const loading = document.getElementById('workspaceLoading');
                const detail = document.getElementById('workspaceLoadingDetail');
                const bar = document.getElementById('workspaceLoadingBar');
                return !!loading?.getClientRects().length && detail?.textContent.includes('讀取專案') &&
                    !!bar?.getClientRects().length && bar.getAttribute('role') === 'progressbar' &&
                    !bar.hasAttribute('aria-valuenow');
            })()""", self.inspected_project_loading)

        def inspected_project_loading(self, visible):
            if visible is not True:
                QTimer.singleShot(75, self.inspect_project_loading)
                return
            result['project_progress_indeterminate'] = True
            release_project.set()
            self.inspect_project_ready()

        def inspect_project_ready(self):
            self.page.runJavaScript("""(() => {
                const state = window.workbenchState();
                return state.stage === 'acquire' && !state.transitioning &&
                    !document.getElementById('workspaceLoading').getClientRects().length &&
                    !!document.getElementById('acquire').getClientRects().length;
            })()""", self.inspected_project_ready)

        def inspected_project_ready(self, ready):
            if ready is True:
                result['project_ready'] = True
                self.close()  # Use the production save/close handshake.
            else:
                QTimer.singleShot(75, self.inspect_project_ready)

        def timeout(self):
            if not result['closed']:
                result['error'] = 'Desktop startup or orderly close timed out.'
                release_system.set()
                release_project.set()
                self.allow_close = True
                self.close()
                QApplication.instance().exit(1)

    with tempfile.TemporaryDirectory(prefix='vision-startup-') as folder:
        data = Path(folder) / 'data'
        project_id = ProjectStore(data / 'projects').create_project('大型專案載入驗收')['id']
        with (
            patch.object(desktop, 'MainWindow', StartupWindow),
            patch.object(StartupWindow, 'start_remote_version_check', return_value=None),
            patch.object(Handler, 'do_GET', get_with_startup_gate),
            patch.object(sys, 'argv', [str(ROOT / 'main.py'), '--data-root', str(data)]),
        ):
            try:
                code = entry.main()
            finally:
                logging.shutdown()
        result['closed'] = bool(windows) and not windows[0].service.thread.is_alive() and not (data / 'desktop.lock').exists()
        result['passed'] = code == 0 and all(value is True for key, value in result.items() if key != 'passed')
    (output / 'report.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print('DESKTOP_STARTUP', json.dumps(result))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
