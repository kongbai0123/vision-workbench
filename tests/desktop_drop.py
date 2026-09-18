"""Native Qt drag/drop through the real desktop shell; isolated synthetic data."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('QTWEBENGINE_CHROMIUM_FLAGS', '--disable-gpu --disable-background-timer-throttling')
import json
from pathlib import Path
import sys
import tempfile
import time
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image
from PySide6.QtCore import QEventLoop, QTimer, QMimeData, QUrl, QPoint, QPointF, Qt
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QDragLeaveEvent
from PySide6.QtWidgets import QApplication
from PySide6.QtTest import QTest
from workbench.desktop import MainWindow, DialogBridge
from workbench.server import WorkbenchService
from workbench.training_engine import atomic_json


def main():
    app = QApplication([])
    output = Path(__file__).resolve().parents[1] / 'qa-output' / 'drop-import'
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='vision-drop-') as temp:
        root = Path(temp)
        labelme = root / '圖片與標註 空白'; labelme.mkdir()
        Image.new('RGB', (100, 80), 'white').save(labelme / '工件.png')
        annotation = labelme / '工件.json'
        annotation.write_text(json.dumps({'imagePath':'工件.png','imageWidth':100,'imageHeight':80,
            'shapes':[{'label':'標籤','shape_type':'rectangle','points':[[10,10],[60,50]]}]}), encoding='utf-8')
        coco = root / 'COCO'; coco.mkdir()
        Image.new('RGB', (100, 80), 'blue').save(coco / 'image.png')
        (coco / 'instances.json').write_text(json.dumps({'images':[{'id':1,'file_name':'image.png','width':100,'height':80}],
            'categories':[{'id':1,'name':'COCO物件'}], 'annotations':[{'id':1,'image_id':1,'category_id':1,'bbox':[10,10,30,20],'area':600,'iscrowd':0}]}),encoding='utf-8')
        yolo = root / 'YOLO'; (yolo / 'images/train').mkdir(parents=True); (yolo / 'labels/train').mkdir(parents=True)
        Image.new('RGB', (100, 80), 'red').save(yolo / 'images/train/frame.png')
        (yolo / 'data.yaml').write_text('names: [YOLO-object]\n', encoding='utf-8')
        (yolo / 'labels/train/frame.txt').write_text('0 0.5 0.5 0.4 0.4\n', encoding='utf-8')
        bad = root / 'unsupported.xyz'; bad.write_text('unsupported')
        checkpoint = root / '外部 權重.pt'; checkpoint.write_bytes(b'synthetic; never loaded by torch')
        def inspect(command, *_args, **_kwargs):
            atomic_json(Path(command[command.index('--output')+1]), {'engine':'external_yolo_detect','task':'object_detection',
                                                                    'classes':['part'],'architecture':'DetectionModel'})
        bridge = DialogBridge(); service = WorkbenchService(root / 'data', dialog=bridge).start()
        window = MainWindow(service, bridge); window.resize(1440,900); window.show()
        def js(script):
            result=[]; loop=QEventLoop()
            window.page.runJavaScript(script,lambda value:(result.append(value),loop.quit()))
            QTimer.singleShot(8000,loop.quit);loop.exec()
            assert result, 'JS timed out'
            return result[0]
        def wait(expression):
            end=time.monotonic()+35
            while time.monotonic()<end:
                if js(expression):return
                QTest.qWait(80)
            raise AssertionError(expression+'\n'+str(js('document.body.innerText')))
        def native_drag(paths, outside=False, drop=True, selector='#fileDropZone', highlight='drop-active'):
            rect=json.loads(js(f"JSON.stringify(document.querySelector({json.dumps(selector)}).getBoundingClientRect().toJSON())"))
            zoom=window.view.zoomFactor()
            p=QPoint(5,5) if outside else QPoint(round((rect['x']+rect['width']/2)*zoom),round((rect['y']+rect['height']/2)*zoom))
            target=window.view.focusProxy() or window.view
            p=target.mapFrom(window.view,p)
            mime=QMimeData();mime.setUrls([QUrl.fromLocalFile(str(path)) for path in paths])
            enter=QDragEnterEvent(p,Qt.DropAction.CopyAction,mime,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier)
            QApplication.sendEvent(target,enter);QTest.qWait(120)
            contains=f"document.querySelector({json.dumps(selector)}).classList.contains({json.dumps(highlight)})"
            if not outside:
                wait(contains)
            if not drop:
                QTest.qWait(400)
                window.view.grab().save(str(output/'drag-highlight.png'))
                QApplication.sendEvent(target,QDragLeaveEvent());QTest.qWait(100)
                assert not js(contains)
                return
            event=QDropEvent(QPointF(p),Qt.DropAction.CopyAction,mime,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier)
            QApplication.sendEvent(target,event);QTest.qWait(150)
        def confirm_import():
            wait("document.querySelector('#formDialog').open && document.querySelector('#dialogTitle').textContent.includes('選擇要匯入')")
            js("document.querySelector('#dialogForm').requestSubmit()")
        try:
            wait("typeof window.workbenchState==='function' && !document.querySelector('#newProject').disabled")
            js("document.querySelector('#newProject').click()")
            wait("document.querySelector('#formDialog').open")
            js("document.querySelector('#dialogBody input').value='拖曳匯入驗收';document.querySelector('#dialogForm').requestSubmit()")
            wait("window.workbenchState().projectId && !window.workbenchState().transitioning")
            js("document.querySelector('[data-source=files]').click()")
            native_drag([labelme], drop=False)
            native_drag([labelme], outside=True)
            assert service.store.list_projects()[0]['stats']['total']==0
            native_drag([labelme]);confirm_import()
            wait("document.querySelector('#acquireTotal').textContent==='1' && !window.workbenchState().busy")
            # Deliberately send image before COCO JSON to catch label loss due to ordering.
            window.view.setZoomFactor(1.25);QTest.qWait(150)
            native_drag([coco/'image.png',coco/'instances.json']);confirm_import()
            wait("document.querySelector('#acquireTotal').textContent==='2' && !window.workbenchState().busy")
            window.view.setZoomFactor(1);QTest.qWait(100)
            native_drag([yolo/'labels/train/frame.txt']);confirm_import()
            wait("document.querySelector('#acquireTotal').textContent==='3' && !window.workbenchState().busy")
            pid=service.store.list_projects()[0]['id']
            snapshot=service.store.snapshot(pid)
            assert sorted(s['label'] for a in snapshot['assets'] for s in a['shapes'])==['COCO物件','YOLO-object','標籤']
            native_drag([annotation]);confirm_import()
            wait("!window.workbenchState().busy && document.querySelector('#importReport').innerText.includes('重複')")
            assert service.store.get_project(pid)['stats']['total']==3
            native_drag([bad])
            # Unsupported sources are reported inside the preview, so nothing is imported on cancel.
            wait("document.querySelector('#formDialog').open && document.querySelector('#dialogBody').textContent.includes('格式不支援')")
            js("document.querySelector('#cancelDialog').click()")
            wait("!document.querySelector('#formDialog').open && !window.workbenchState().busy")
            assert window.page.url().toString().rstrip('/')==service.url.rstrip('/')
            assert service.store.get_project(pid)['stats']['total']==3
            js("document.querySelector('#toast').hidden=true")
            QTest.qWait(100);window.view.grab().save(str(output/'imported.png'))
            # The model import dialog takes the drop from the acquisition zone below it.
            service.training.registry.component_status=lambda *args,**kwargs:{key:{'state':'ready','message':'synthetic runtime'}
                                                                             for key in ('builtin','torchvision','anomalib','ultralytics')}
            js("document.querySelector('[data-stage=models]').click()")
            wait("!document.querySelector('#models').hidden && !window.workbenchState().busy && !window.workbenchState().transitioning")
            js("document.querySelector('#importModel').click()")
            wait("document.querySelector('.model-import-dropzone')!==null")
            native_drag([bad],selector='.model-import-dropzone',highlight='drag-active')
            wait("document.querySelector('#toast').classList.contains('error') && !document.querySelector('.model-import-selection').classList.contains('selected')")
            # The toast joins the top layer, so the dialog opened before it cannot cover it.
            assert js("document.querySelector('#formDialog').open && document.querySelector('#toast').matches(':popover-open')")
            QTest.qWait(300);window.view.grab().save(str(output/'dialog-toast.png'))
            native_drag([checkpoint],selector='.model-import-dropzone',highlight='drag-active')
            wait("document.querySelector('.model-import-selection').textContent.includes('外部 權重.pt')")
            assert not js("document.querySelector('#fileDropZone').classList.contains('drop-active')")
            js("document.querySelector('#modelImportTrusted').checked=true")
            QTest.qWait(400);window.view.grab().save(str(output/'model-drop.png'))
            with patch('workbench.process_control.run_controlled',inspect):
                js("document.querySelector('#dialogForm').requestSubmit()")
                wait("!document.querySelector('#formDialog').open && document.querySelector('#modelDetail').textContent.includes('外部匯入')")
            models=service.training.list_models(pid)
            assert len(models)==1 and models[0]['source']['filename']=='外部 權重.pt',models
            # A dropped checkpoint is copied, never moved out of the user's folder.
            assert checkpoint.is_file()
            (output/'report.json').write_text(json.dumps({'success':True,'images':3,'labels':3,
                'checks':['native folder drop','Chinese/space paths','drag highlight/leave','outside ignored',
                    'multi-file COCO with image first','125% zoom mapping','YOLO TXT pairing','LabelMe JSON duplicate',
                    'unsupported format reported','no file navigation','model import dialog drop','toast above modal dialog']},ensure_ascii=False,indent=2),encoding='utf-8')
            print('NATIVE_DROP_IMPORT_OK',flush=True)
        finally:
            window.allow_close=True;window.close();service.close()

if __name__=='__main__':main()
