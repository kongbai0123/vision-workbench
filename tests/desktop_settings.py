"""Desktop Chromium smoke test for the settings and model catalog."""
import json
import os
from pathlib import Path
import sys
import tempfile
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--disable-gpu --disable-background-timer-throttling")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtCore import QEventLoop, QTimer, QUrl
from PySide6.QtTest import QTest
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import QApplication

from workbench.server import WorkbenchService


def main():
    app = QApplication.instance() or QApplication([])
    output = Path(__file__).resolve().parents[1] / "qa-output" / "desktop"
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="vision-settings-") as folder:
        service = WorkbenchService(Path(folder) / "data").start()
        service.store.create_project("模型目錄互動測試")
        view = QWebEngineView(); view.resize(1440, 900); view.setUrl(QUrl(service.url)); view.show()
        page = view.page()

        def js(script):
            result = []; loop = QEventLoop()
            page.runJavaScript(script, lambda value: (result.append(value), loop.quit()))
            QTimer.singleShot(10000, loop.quit); loop.exec()
            if not result: raise AssertionError("JavaScript timed out")
            return result[0]

        def wait(expression, seconds=30):
            deadline = time.monotonic() + seconds
            while time.monotonic() < deadline:
                if js(expression): return
                QTest.qWait(80)
            raise AssertionError("Timeout: " + expression)

        try:
            wait("typeof window.workbenchState==='function'")
            js("document.querySelector('#settings').click()")
            wait("!document.querySelector('#settingsShell').hidden")
            js("document.querySelector('[data-settings-page=models]').click()")
            wait("document.querySelectorAll('#settingsModelList .settings-model-row').length>=15")
            count = js("document.querySelectorAll('#settingsModelList .settings-model-row').length")
            assert count >= 15, count
            js("[...document.querySelectorAll('#settingsModelList .settings-model-row')].find(e=>e.innerText.includes('EfficientAD')).click()")
            wait("document.querySelector('#settingsModelDetail').innerText.includes('整合開發中')")
            QTest.qWait(200); view.grab().save(str(output / "10-settings-model-center.png"))
            js("document.querySelector('[data-settings-page=updates]').click()")
            wait("!document.querySelector('[data-settings-panel=updates]').hidden")
            wait("document.querySelector('#settingsVersion').textContent.includes('Vision Workbench')")
            QTest.qWait(150); view.grab().save(str(output / "11-settings-updates.png"))
            js("document.querySelector('#closeSettings').click()")
            js("document.querySelector('.project-card-open').click()")
            wait("!document.querySelector('[data-stage=train]').disabled")
            js("document.querySelector('[data-stage=train]').click()")
            wait("document.querySelectorAll('#trainingEngine option').length>=15")
            assert js("[...document.querySelectorAll('#trainingEngine option')].every(option=>!option.disabled)")
            js("document.querySelector('#trainingEngine').value='yolo26n_seg';document.querySelector('#trainingEngine').dispatchEvent(new Event('change'))")
            wait("!document.querySelector('#trainingModelNotice').hidden")
            assert js("document.querySelector('#trainingModelNotice').innerText.includes('尚未準備')")
            assert not js("document.documentElement.scrollWidth>innerWidth+2")
            print("DESKTOP_SETTINGS_OK", json.dumps({"models": count}, ensure_ascii=False))
        finally:
            view.close(); service.close()


if __name__ == "__main__": main()
