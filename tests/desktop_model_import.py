"""Real desktop/API import workflow with a synthetic checkpoint inspector."""
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('QTWEBENGINE_CHROMIUM_FLAGS', '--disable-gpu --disable-background-timer-throttling --disable-renderer-backgrounding')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtCore import QEventLoop, QTimer, QUrl
from PySide6.QtTest import QTest
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import QApplication
from workbench.server import WorkbenchService
from workbench.training_engine import atomic_json
from desktop_training import Page


def main():
    app = QApplication.instance() or QApplication([])
    output = Path(__file__).parents[1] / 'qa-output/model-import'
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as folder:
        folder = Path(folder)
        service = WorkbenchService(folder / 'data').start()
        project = service.store.create_project('外部模型匯入示範')
        registry = service.training.registry
        registry.component_status = lambda *args, **kwargs: {key: {'state': 'ready', 'message': 'synthetic runtime'} for key in ('builtin', 'torchvision', 'anomalib', 'ultralytics')}
        def inspect(command, *_args, **_kwargs):
            atomic_json(Path(command[command.index('--output')+1]), {'engine': 'external_yolo_detect', 'task': 'object_detection', 'classes': ['part'], 'architecture': 'DetectionModel'})
        view = QWebEngineView();page = Page(view);view.setPage(page)
        view.resize(1460, 1000);view.show();view.setUrl(QUrl(service.entry_url))
        def js(script):
            values=[];loop=QEventLoop();page.runJavaScript(script,lambda value:(values.append(value),loop.quit()))
            QTimer.singleShot(8000,loop.quit);loop.exec();assert values,'JavaScript timeout';return values[0]
        def wait(expression):
            deadline=time.monotonic()+20
            while time.monotonic()<deadline:
                if js(expression):return
                QTest.qWait(50)
            raise AssertionError(expression+'\n'+str(js('document.body.innerText')))
        def click(selector):
            wait(f"document.querySelector({json.dumps(selector)}) && !document.querySelector({json.dumps(selector)}).disabled && !window.workbenchState().busy && !window.workbenchState().transitioning")
            js(f'document.querySelector({json.dumps(selector)}).click()')
        def settle():
            loop=QEventLoop();QTimer.singleShot(500,loop.quit);loop.exec()
        try:
            wait("document.querySelector('.project-card')!==null")
            click('.project-card');wait("typeof window.workbenchState==='function' && !window.workbenchState().busy")
            click('[data-stage="models"]')
            wait("!document.querySelector('#models').hidden && !window.workbenchState().busy")
            wait("document.querySelector('#modelDetail').textContent.includes('尚無可用模型')")
            for width in (1460,1088):
                view.resize(width,1000);QTest.qWait(200)
                assert js("Math.abs(document.querySelector('.model-browser').getBoundingClientRect().bottom-document.querySelector('#modelDetail').getBoundingClientRect().bottom)<2")
            settle();view.grab().save(str(output/'aligned-empty.png'))
            click('#importModel');wait("document.querySelector('#modelImportFile')?.type==='file'")
            assert js("document.querySelector('#modelImportPath')===null && document.querySelector('#modelImportFile').accept==='.pt'")
            js("""(()=>{const transfer=new DataTransfer();transfer.items.add(new File([new TextEncoder().encode('synthetic; never loaded by torch')],'external.pt',{type:'application/octet-stream'}));document.querySelector('.model-import-dropzone').dispatchEvent(new DragEvent('drop',{bubbles:true,cancelable:true,dataTransfer:transfer}));return true})()""")
            wait("document.querySelector('.model-import-selection').textContent.includes('external.pt')")
            settle();view.grab().save(str(output/'selected-file.png'))
            click('#confirmDialog')
            wait("document.querySelector('#toast').textContent.includes('可信任')")
            assert not service.training.list_models(project['id'])
            js("document.querySelector('#modelImportName').value='零件模型';document.querySelector('#modelImportTrusted').checked=true")
            with patch('workbench.process_control.run_controlled', inspect):
                click('#confirmDialog')
                wait("!document.querySelector('#formDialog').open && document.querySelector('#modelDetail').textContent.includes('零件模型')")
            text=js("document.querySelector('#modelDetail').textContent")
            assert '尚未評估' in text and '0.000' not in text
            assert list(service.model_uploads.root.iterdir())==[]
            assert js("!document.querySelector('#trialImages').disabled && document.querySelector('#runModelComparison').disabled")
            assert js("Math.abs(document.querySelector('.model-browser').getBoundingClientRect().bottom-document.querySelector('#modelDetail').getBoundingClientRect().bottom)<2")
            wait("!window.workbenchState().busy && !window.workbenchState().transitioning")
            js("document.querySelector('#toast').textContent=''")
            settle();view.grab().save(str(output/'imported.png'))
            assert service.training.catalog_issues[project['id']]==[]
            assert not page.errors,page.errors
            print('MODEL_IMPORT_DESKTOP_OK',flush=True)
        finally:
            view.close();service.close()
    app.processEvents()


if __name__=='__main__':main()
