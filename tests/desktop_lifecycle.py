"""Explicit native shell startup/save-before-close check, isolated from user data."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
os.environ.setdefault('QTWEBENGINE_CHROMIUM_FLAGS','--disable-gpu --disable-background-timer-throttling --disable-renderer-backgrounding')
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import tempfile
import time
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QEventLoop,QTimer
from PySide6.QtTest import QTest
from workbench.desktop import DialogBridge,MainWindow
from workbench.server import WorkbenchService


def main():
    app=QApplication([])
    with tempfile.TemporaryDirectory(prefix='vision-shell-') as folder:
        bridge=DialogBridge();service=WorkbenchService(Path(folder),dialog=bridge).start()
        window=MainWindow(service,bridge);window.show()
        def js(script):
            result=[];loop=QEventLoop()
            window.page.runJavaScript(script,lambda value:(result.append(value),loop.quit()))
            QTimer.singleShot(5000,loop.quit);loop.exec();return result[0] if result else None
        try:
            end=time.monotonic()+25
            while time.monotonic()<end:
                if js("typeof window.workbenchFlush==='function'"):break
                QTest.qWait(100)
            else:raise AssertionError('Native shell failed to load')
            assert window.update_timer.isActive(),'Update indicator polling did not start after the UI loaded'
            assert window.update_timer.interval() <= 3000,'Update indicator refresh is too slow'
            original_hash=window.source_baseline['web/app.mjs']
            window.source_baseline['web/app.mjs']='outdated'
            window.check_update_indicator();QTest.qWait(120)
            assert js("!document.querySelector('#updateDot').hidden"),'Changed source did not show the update indicator'
            window.source_baseline['web/app.mjs']=original_hash
            window.check_update_indicator();QTest.qWait(120)
            assert js("document.querySelector('#updateDot').hidden"),'Current source did not clear the update indicator'
            assert window.fullscreen_action.shortcut().toString() == 'F11'
            window.toggle_fullscreen();QTest.qWait(120)
            assert window.isFullScreen(),'F11 action did not enter full-screen mode'
            window.toggle_fullscreen();QTest.qWait(120)
            assert not window.isFullScreen(),'F11 action did not restore windowed mode'
            # Make saving visibly asynchronous; the real close handler must lock
            # editing and wait before allowing the native window to disappear.
            js("window.workbenchFlush=()=>new Promise(r=>setTimeout(()=>{window.__saved=true;r(true)},500))")
            window.close()
            assert window.isVisible() and not window.view.isEnabled()
            end=time.monotonic()+5
            while time.monotonic()<end and window.isVisible():QTest.qWait(100)
            assert not window.isVisible(),'Close did not wait for and finish the save'
            assert js("window.__saved===true"),'Window closed before the save completed'
            print('NATIVE_SHELL_STARTUP_AND_CLOSE_OK')
        finally:
            window.allow_close=True;window.close();service.close()


if __name__=='__main__':main()
