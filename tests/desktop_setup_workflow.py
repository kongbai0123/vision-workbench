"""Exercise review, preparation and setup through real Qt Chromium.

Run explicitly: .venv/Scripts/python.exe tests/desktop_setup_workflow.py
Projects are isolated fixtures. Training admission is patched to fail immediately.
"""
import json
import os
from pathlib import Path
import sys
import tempfile
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--disable-gpu --disable-background-timer-throttling --disable-renderer-backgrounding")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image
import numpy as np
from PySide6.QtCore import QEventLoop, QTimer, QUrl
from PySide6.QtTest import QTest
from PySide6.QtWebEngineCore import QWebEnginePage
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import QApplication

from workbench.server import WorkbenchService
from workbench.training_engine import atomic_json
from composer_core.geometry import encode_rle


class Page(QWebEnginePage):
    def __init__(self, *args):
        super().__init__(*args)
        self.errors = []

    def javaScriptConsoleMessage(self, level, message, line, source):
        if level == QWebEnginePage.JavaScriptConsoleMessageLevel.ErrorMessageLevel:
            self.errors.append(f"{source}:{line}: {message}")


def main():
    from unittest.mock import patch
    from workbench.training import TrainingWorkspace
    from workbench.workflow_drafts import WorkflowDrafts
    app = QApplication.instance() or QApplication([])
    output = Path(__file__).resolve().parents[1] / 'qa-output' / 'setup-audit'
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='setup-no-training-') as directory, patch.object(TrainingWorkspace, 'start_run', side_effect=AssertionError('TRAINING FORBIDDEN')) as start:
        root = Path(directory)
        service = WorkbenchService(root / 'data').start()
        pid = service.store.create_project('Setup A')['id']
        other = service.store.create_project('Setup B')['id']
        for i in range(12):
            path = root / f'image{i}.png'
            Image.new('RGB', (160,120), (20+i*10,80,120)).save(path)
            service.store.add_assets(pid,[{'path':str(path),'batch_id':f'source-{i}', 'shapes':[
                {'id':f'shape-{i}','type':'rectangle','label':'part','x':10,'y':10,'width':30,'height':30}]}])
        view=QWebEngineView();page=Page(view);view.setPage(page)
        view.resize(1440,1000);view.show();page.setVisible(True);view.setUrl(QUrl(service.entry_url))
        def js(script):
            results=[];loop=QEventLoop()
            page.runJavaScript(script,lambda value:(results.append(value),loop.quit()))
            QTimer.singleShot(10000,loop.quit);loop.exec()
            assert results, 'JavaScript timeout'
            return results[0]
        def wait(expression):
            deadline=time.monotonic()+25
            while time.monotonic()<deadline:
                if js(expression):return
                QTest.qWait(80)
            raise AssertionError(expression+'\n'+str(page.errors)+'\n'+str(js('document.body.innerText.slice(-3500)')))
        def click(selector):
            print('CLICK',selector,flush=True)
            wait('!window.workbenchState().busy&&!window.workbenchState().transitioning')
            wait(f"document.querySelector({json.dumps(selector)})&&!document.querySelector({json.dumps(selector)}).disabled")
            js(f'document.querySelector({json.dumps(selector)}).click()')
        def fill(selector,value):
            js(f"(()=>{{const e=document.querySelector({json.dumps(selector)});e.value={json.dumps(str(value))};e.dispatchEvent(new Event('input',{{bubbles:true}}));e.dispatchEvent(new Event('change',{{bubbles:true}}));}})()")
        def open_project(name):
            wait("window.workbenchState().stage==='library'&&!window.workbenchState().transitioning")
            wait("document.querySelector('.project-card-open')!==null")
            js(f"[...document.querySelectorAll('.project-card-open')].find(e=>e.textContent.includes({json.dumps(name)})).click()")
            expected=pid if name=='Setup A' else other
            wait(f'window.workbenchState().projectId==={json.dumps(expected)}&&!window.workbenchState().transitioning')
        def capture(name):
            QTest.qWait(200);view.grab().save(str(output / f'{name}.png'))
            assert not js('document.documentElement.scrollWidth>innerWidth+2')
        try:
            wait("typeof window.workbenchState==='function'")
            open_project('Setup A')
            click('[data-stage=review]');wait("document.querySelectorAll('.review-card').length===12")
            fill('#reviewAnnotationFilter','unannotated');wait("document.querySelectorAll('.review-card').length===0")
            fill('#reviewAnnotationFilter','annotated');wait("document.querySelectorAll('.review-card').length===12")
            click('#reviewSelectAll');click('#reviewApprove');wait("document.querySelectorAll('.review-card').length===0")
            click('[data-stage=split]');wait("document.querySelector('#splitFlowStats').textContent.includes('12')")
            click('#openSplitFlowManager');wait("!document.querySelector('#previewSmartSplit').disabled")
            fill('#smartRatio-train',60);fill('#smartRatio-val',20);fill('#smartRatio-test',20);fill('#smartSplitSeed',123)
            click('#previewSmartSplit');wait("!document.querySelector('#applySmartSplit').disabled")
            click('#applySmartSplit');wait("!document.querySelector('#smartSplitDialog').open")
            click('#openSplitFlowManager');wait("!document.querySelector('#previewSmartSplit').disabled")
            assert js("document.querySelector('#smartSplitSeed').value")=='123'
            assert js("document.querySelector('#smartRatio-train').value")=='60'
            click('#smartSplitDialog .smart-split-header button')
            fill('#augmentationPreset','custom');fill('#augmentationExpansion',2);fill('#augmentationBrightness',.25)
            wait("document.querySelector('#augmentationSummary').textContent.includes('每輪')")
            fill('#augmentationExpansion','');assert js("document.querySelector('#createPreparationDatasetVersion').disabled")
            fill('#augmentationExpansion',2)
            click('#createPreparationDatasetVersion');wait("document.querySelector('#augmentationVersionState').textContent.includes('D001')")
            capture('05-preparation')
            dataset=service.training.dataset(pid,'D001');count=dataset['splits']['train']
            assert dataset['training_events']['events']==count*3
            old=(service.training.datasets_dir(pid)/'D001'/'manifest.json').read_bytes()
            click('#continueToTraining');wait("window.workbenchState().stage==='train'")
            fill('#trainingEngine','yolo26n_detect');fill('[data-training-param=batch_size]',4)
            fill('[data-training-param=gradient_accumulation]',3)
            click('#estimateTrainingTime');wait("document.querySelector('#trainingSetupReport').textContent.includes('首批實測')")
            assert service.training.list_runs(pid)==[]
            click('#validateTrainingSetup');wait("document.querySelector('#trainingSetupReport').textContent.includes('設定驗證通過')")
            assert js("document.querySelector('#trainingSummary').textContent.includes('12 張')")
            capture('06-preflight')
            fill('[data-training-param=image_size]',641);assert js("document.querySelector('#validateTrainingSetup').disabled")
            fill('[data-training-param=image_size]',640)
            fill('#trainingEngine','pixel_prototype_v1');click('#validateTrainingSetup')
            wait("document.querySelector('#trainingSetupReport').textContent.includes('設定驗證通過')")
            assert js("document.querySelector('#trainingSetupReport').textContent.includes('不套用擴增')")
            fill('#trainingEngine','yolo26n_detect');assert js("document.querySelector('[data-training-param=batch_size]').value")=='4'
            click('[data-stage=split]');fill('#augmentationExpansion',3)
            click('#continueToTraining');wait("document.querySelector('#trainingSetupReport').textContent.includes('草稿與 D001 不同')")
            assert (service.training.datasets_dir(pid)/'D001'/'manifest.json').read_bytes()==old
            click('#home');open_project('Setup B');click('[data-stage=split]')
            wait("window.workbenchState().stage==='split'&&!window.workbenchState().transitioning")
            assert js("document.querySelector('#augmentationExpansion').value")=='0', js("JSON.stringify({state:window.workbenchState(),value:document.querySelector('#augmentationExpansion').value,text:document.body.innerText.slice(-4000)})")
            click('#home');open_project('Setup A');click('[data-stage=split]')
            wait("window.workbenchState().stage==='split'&&!window.workbenchState().transitioning")
            assert js("document.querySelector('#augmentationExpansion').value")=='3'
            assert js("document.querySelector('#augmentationBrightness').value")=='0.25'
            click('#continueToTraining');wait("document.querySelector('#trainingEngine').value==='yolo26n_detect'")
            assert js("document.querySelector('[data-training-param=batch_size]').value")=='4'
            click('#home')
            loaded=QEventLoop()
            view.loadFinished.connect(loaded.quit)
            view.setUrl(QUrl(service.entry_url));QTimer.singleShot(15000,loaded.quit);loaded.exec()
            view.loadFinished.disconnect(loaded.quit)
            wait("typeof window.workbenchState==='function'&&document.querySelector('.project-card-open')!==null")
            open_project('Setup A');click('[data-stage=split]')
            wait("window.workbenchState().stage==='split'&&!window.workbenchState().transitioning")
            assert js("document.querySelector('#augmentationExpansion').value")=='3'
            for width in (1440,1024,760):
                view.resize(width,1000);QTest.qWait(180);capture(f'05-width-{width}')
                small=js("JSON.stringify([...document.querySelectorAll('#split *')].filter(e=>e.getClientRects().length&&[...e.childNodes].some(n=>n.nodeType===3&&n.textContent.trim())&&parseFloat(getComputedStyle(e).fontSize)<12).map(e=>e.outerHTML.slice(0,100)))")
                assert json.loads(small)==[],small
            assert WorkflowDrafts(service.store).read(pid)['payload']['augmentation']['expansion_count']=='3'
            assert service.training.list_runs(pid)==[]
            assert service.training.list_runs(other)==[]
            start.assert_not_called()
            assert not page.errors,page.errors
            print('SETUP_WORKFLOW_OK: review, split restore, augmentation, immutable version, preflight, engine switching, validation, project isolation, reload, responsive text. ZERO training starts.',flush=True)
        finally:
            view.close();service.close();QTest.qWait(100)

if __name__=='__main__':
    main()
