"""Exercise the real desktop entry, authenticated UI and orderly close in isolation."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('QTWEBENGINE_CHROMIUM_FLAGS', '--disable-gpu --disable-background-timer-throttling')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

import main as entry
from workbench import desktop


def main():
    output = ROOT / 'qa-output' / 'startup'
    output.mkdir(parents=True, exist_ok=True)
    result = {'passed': False, 'ui_ready': False, 'authenticated_api': False, 'closed': False}
    windows = []

    class StartupWindow(desktop.MainWindow):
        def __init__(self, service, bridge):
            super().__init__(service, bridge)
            windows.append(self)
            QTimer.singleShot(200, self.inspect)
            QTimer.singleShot(30000, self.timeout)

        def inspect(self):
            if result['ui_ready']:
                return
            self.page.runJavaScript("""JSON.stringify((()=>{
                const ready=typeof window.workbenchState==='function' && typeof window.workbenchFlush==='function';
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
            if state.get('ready') and state.get('api') == 'ok':
                result['ui_ready'] = True
                result['authenticated_api'] = True
                self.view.grab().save(str(output / 'desktop.png'))
                self.close()  # Use the production save/close handshake.
            else:
                QTimer.singleShot(150, self.inspect)

        def timeout(self):
            if not result['closed']:
                result['error'] = 'Desktop startup or orderly close timed out.'
                self.allow_close = True
                self.close()
                QApplication.instance().exit(1)

    with tempfile.TemporaryDirectory(prefix='vision-startup-') as folder:
        data = Path(folder) / 'data'
        with patch.object(desktop, 'MainWindow', StartupWindow), patch.object(
            StartupWindow, 'start_remote_version_check', return_value=None
        ), patch.object(sys, 'argv', [str(ROOT / 'main.py'), '--data-root', str(data)]):
            try:
                code = entry.main()
            finally:
                logging.shutdown()
        result['closed'] = bool(windows) and not windows[0].service.thread.is_alive() and not (data / 'desktop.lock').exists()
        result['passed'] = code == 0 and result['ui_ready'] and result['authenticated_api'] and result['closed']
    (output / 'report.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print('DESKTOP_STARTUP', json.dumps(result))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
