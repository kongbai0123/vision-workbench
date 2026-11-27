"""Real Qt Chromium UI workflow in an isolated store. No user's window or camera.

Run explicitly: .venv/Scripts/python.exe tests/desktop_workflow.py
"""
import os
os.environ.setdefault("QT_QPA_PLATFORM","offscreen")
os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS","--disable-gpu --disable-background-timer-throttling --disable-renderer-backgrounding")
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import tempfile
import time

from PIL import Image,ImageDraw
from PySide6.QtWidgets import QApplication
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWebEngineCore import QWebEnginePage
from PySide6.QtCore import QEventLoop,QTimer,QUrl,QPoint,Qt
from PySide6.QtTest import QTest

from workbench.server import WorkbenchService


class Page(QWebEnginePage):
    def __init__(self,*args):
        super().__init__(*args);self.errors=[]
    def javaScriptConsoleMessage(self,level,message,line,source):
        if "Error" in message or "error" in message or "Refused" in message:
            self.errors.append(f"{source}:{line}: {message}")
            print("WEB:",message,flush=True)


def main():
    app=QApplication.instance() or QApplication([])
    output=Path(__file__).resolve().parents[1]/"qa-output"/"desktop"
    output.mkdir(parents=True,exist_ok=True)
    steps=[]
    with tempfile.TemporaryDirectory(prefix="vision-ui-") as directory:
        root=Path(directory)
        inputs=[]
        for batch in range(2):
            source=root/f"拍攝批次{batch+1}";source.mkdir();inputs.append(str(source))
            for number in range(2):
                image=Image.new("RGB",(640,480),(230-batch*12,233-number*16,235))
                draw=ImageDraw.Draw(image)
                draw.rounded_rectangle((125+number*20,90,510,390),radius=25,fill=(50+batch*25,90,125))
                draw.ellipse((260,180,370,290),fill=(230-batch*12,233-number*16,235))
                image.save(source/f"工件-{number+1}.png")
                (source/f"工件-{number+1}.json").write_text(json.dumps({"imagePath":f"工件-{number+1}.png", "imageWidth":640,"imageHeight":480,
                    "shapes":[{"label":"工件","shape_type":"rectangle","points":[[125+number*20,90],[510,390]]}]}),encoding="utf-8")
        service=WorkbenchService(root/"data").start()
        view=QWebEngineView();page=Page(view);view.setPage(page);view.resize(1440,900);view.show()
        view.setUrl(QUrl(service.entry_url));page.setVisible(True)
        def js(script):
            result=[];loop=QEventLoop()
            page.runJavaScript(script,lambda value:(result.append(value),loop.quit()))
            QTimer.singleShot(10000,loop.quit);loop.exec()
            if not result:raise AssertionError("JavaScript timed out")
            return result[0]
        def wait(expression,seconds=40):
            end=time.monotonic()+seconds
            while time.monotonic()<end:
                if js(expression):return
                QTest.qWait(80)
            raise AssertionError("Timeout: "+expression+"\n"+str(js("document.body.innerText.slice(-3500)")))
        def click(selector):
            assert js(f"!!document.querySelector({json.dumps(selector)})"),selector
            wait("!document.querySelector('#home').disabled")
            wait(f"(()=>{{const e=document.querySelector({json.dumps(selector)});if(!e||e.disabled)return false;e.click();return true}})()")
        def fill(selector,value):
            js(f"{{const e=document.querySelector({json.dumps(selector)});e.value={json.dumps(value)};e.dispatchEvent(new Event('input',{{bubbles:true}}));e.dispatchEvent(new Event('change',{{bubbles:true}}));}}")
        def capture(name):
            QTest.qWait(200);view.grab().save(str(output/f"{name}.png"))
            overflow=js("document.documentElement.scrollWidth>innerWidth+2")
            if overflow:raise AssertionError("Horizontal page overflow: "+name)
            steps.append(name);print("PASS",name,flush=True)
        try:
            wait("typeof window.workbenchState==='function'")
            assert js("!!document.querySelector('#settings') && document.querySelector('#updateDot').hidden")
            wait("!document.querySelector('#libraryEmpty').hidden")
            capture("01-project-library")
            click("#newProject")
            wait("document.querySelector('#formDialog').open")
            assert js("document.activeElement.id") == "createProjectName"
            fill("#createProjectName","整合流程驗收")
            fill("#createProjectClasses"," 工件；L_grip;工件\n\n表面 瑕疵；")
            assert js("document.querySelector('#createProjectClassCount').textContent") == "已輸入 3 個類別"
            js("document.querySelector('#dialogForm').requestSubmit()")
            wait("!document.querySelector('#acquire').hidden")
            initial_classes = ['工件', 'L_grip', '表面 瑕疵']
            created = service.store.list_projects()[0]
            assert service.store.get_project(created['id'])['classes'] == initial_classes
            click("[data-source='files']")
            click("#showManualImport")
            fill("#importPaths","\n".join(inputs));click("#importFiles")
            wait("document.querySelector('#formDialog').open && document.querySelectorAll('#dialogBody input[type=checkbox]').length===4")
            js("document.querySelector('#confirmDialog').click()")
            wait("document.querySelector('#importReport').innerText.includes('4')",seconds=60)
            wait("!document.querySelector('#importFiles').disabled")
            project=service.store.list_projects()[0];pid=project['id']
            assert project['stats']['total']==4,project
            capture("02-shared-import")
            click("[data-stage='annotate']")
            wait("!document.querySelector('#imageStage').hidden && document.querySelector('#assetImage').complete")
            displayed_classes = js("JSON.stringify(Array.from(document.querySelectorAll('#annotationClassChoices [data-class-name]'), button => button.dataset.className))")
            assert json.loads(displayed_classes) == initial_classes, displayed_classes
            # Rebuilding the list after selection must retain the user's place.
            js("{const e=document.querySelector('#assetList');e.style.cssText='flex:0 0 120px;height:120px';e.scrollTop=e.scrollHeight}")
            js("document.querySelector('#assetList .asset-row:last-child').click()")
            wait("document.querySelector('#assetList .asset-row:last-child').classList.contains('active') && !document.querySelector('#home').disabled")
            QTest.qWait(120)
            assert js("(()=>{const l=document.querySelector('#assetList'),a=l.querySelector('.asset-row.active'),lr=l.getBoundingClientRect(),ar=a.getBoundingClientRect();return l.scrollTop>0&&ar.top>=lr.top-1&&ar.bottom<=lr.bottom+1})()"),'Selected asset did not remain visible at the restored scroll position'
            js("document.querySelector('#assetList .asset-row:first-child').click()")
            wait("document.querySelector('#assetList .asset-row:first-child').classList.contains('active') && !document.querySelector('#home').disabled")
            js("document.querySelector('#assetList').removeAttribute('style')")
            capture("03-annotation-workspace")
            assert js("document.querySelector('#annotationClassChoices button[data-class-name=工件]').getClientRects().length > 0")
            # Category management must produce an immediate, persisted UI change.
            js("""window.classManagerFocus=[];
                document.addEventListener('focusin',event=>{
                    if(event.target.closest('#formDialog'))window.classManagerFocus.push(event.target.id);
                });""")
            click("#manageClasses");wait("document.querySelector('#formDialog').open")
            QTest.qWait(90)
            assert js("window.classManagerFocus.join()") == 'classManagementInput'
            assert js("getComputedStyle(document.querySelector('#formDialog'),'::backdrop').backdropFilter") == 'none'
            js("""window.originalClassRow=document.querySelector('.class-management-row');
                window.classEditorTop=document.querySelector('#classManagementInput').getBoundingClientRect().top;""")
            fill("#dialogBody textarea","工件\n已修正工件\n暫存類別")
            assert js("""window.originalClassRow===document.querySelector('.class-management-row') &&
                Math.abs(window.classEditorTop-document.querySelector('#classManagementInput').getBoundingClientRect().top)<1""")
            js("document.querySelector('#dialogForm').requestSubmit()")
            wait("!document.querySelector('#formDialog').open && !!document.querySelector('#annotationClassChoices button[data-class-name=暫存類別]')")
            assert service.store.get_project(pid)['classes']==['工件','已修正工件','暫存類別']
            click("#manageClasses");wait("document.querySelector('#formDialog').open")
            fill("#dialogBody textarea","工件\n已修正工件")
            js("document.querySelector('#dialogForm').requestSubmit()")
            wait("!document.querySelector('#formDialog').open && !document.querySelector('#annotationClassChoices button[data-class-name=暫存類別]')")
            assert service.store.get_project(pid)['classes']==['工件','已修正工件']
            class_choice = '#annotationClassChoices button[data-class-name="已修正工件"]'
            class_selected = f"document.querySelector({json.dumps(class_choice)}).getAttribute('aria-pressed') === 'true'"
            click(class_choice)
            click("#shapeList .shape-row")
            assert js(f"{class_selected} && document.querySelector('#selectedShapeLabel').value==='工件'")
            fill("#selectedShapeLabel","已修正工件")
            wait("window.workbenchState().dirty")
            click("#saveNow")
            wait("document.querySelector('#shapeList .shape-name').innerText==='已修正工件'")
            wait("!window.workbenchState().dirty")
            snapshot=service.store.snapshot(pid)
            changed=[a for a in snapshot['assets'] if any(s['label']=='已修正工件' for s in a['shapes'])]
            assert len(changed)==1,"UI label edit was not saved"
            changed_id=changed[0]['id']
            click("#nextAsset")
            wait(f"window.workbenchState().assetId!=={json.dumps(changed_id)} && !document.querySelector('#home').disabled")
            click("#previousAsset")
            wait(f"window.workbenchState().assetId==={json.dumps(changed_id)} && !document.querySelector('#home').disabled")
            js("document.querySelector('#canvasArea').dispatchEvent(new KeyboardEvent('keydown',{key:'d',bubbles:true}))")
            wait(f"window.workbenchState().assetId!=={json.dumps(changed_id)} && !document.querySelector('#home').disabled")
            js("document.querySelector('#canvasArea').dispatchEvent(new KeyboardEvent('keydown',{key:'a',bubbles:true}))")
            wait(f"window.workbenchState().assetId==={json.dumps(changed_id)} && !document.querySelector('#home').disabled")
            wait("document.querySelector('#shapeList .shape-name').innerText==='已修正工件'")
            # Real editor command -> undo -> redo, all persisted via same API.
            click("#selectAllShapes");click("#copyShapes");click("#saveNow")
            wait("!window.workbenchState().dirty")
            assert service.store.get_asset(pid,changed_id)['shape_count']==2
            click("#undo");click("#saveNow");wait("!window.workbenchState().dirty")
            assert service.store.get_asset(pid,changed_id)['shape_count']==1
            click("#redo");click("#saveNow");wait("!window.workbenchState().dirty")
            assert service.store.get_asset(pid,changed_id)['shape_count']==2
            click("#shapeList .shape-row")
            fill("#selectedShapeLabel","工件")
            assert js(class_selected),"Existing-object class changed the new-object default"
            wait("window.workbenchState().dirty")
            click("#saveNow");wait("!window.workbenchState().dirty")
            labels=[shape['label'] for shape in service.store.get_asset(pid,changed_id)['shapes']]
            assert sorted(labels)==sorted(['已修正工件','工件']),labels
            # Removing a used class through the real dialog must relabel its
            # objects and remove the class from the persisted project.
            click("#manageClasses");wait("document.querySelector('#formDialog').open")
            fill("#dialogBody textarea","已修正工件")
            wait("!!document.querySelector('#dialogBody .requires-replacement select')")
            fill("#dialogBody .requires-replacement select","已修正工件")
            fill("#dialogBody textarea","已修正工件\n待檢查類別")
            assert js("document.querySelector('#dialogBody .requires-replacement select').value") == '已修正工件'
            fill("#dialogBody textarea","已修正工件")
            js("document.querySelector('#dialogForm').requestSubmit()")
            wait("!document.querySelector('#formDialog').open && !document.querySelector('#annotationClassChoices button[data-class-name=工件]')")
            snapshot=service.store.snapshot(pid)
            assert snapshot['classes']==['已修正工件']
            assert not any(shape['label']=='工件' for asset in snapshot['assets'] for shape in asset['shapes'])
            # Request an actual local segmentation through the UI and accept it.
            click("#selectAllShapes");click("#annotationAiTab")
            click("#annotationSegmentationTab")
            wait("!document.querySelector('#segmentationPanel').hidden")
            assert js("!document.querySelector('#segmentationPanel #aiMessage')")
            js("""{
                const fetchOriginal=window.fetch.bind(window);
                window.testAiRequests=[];
                window.fetch=(url,options)=>{
                    if(options?.method==='POST' && /\\/(ai|predictions)$/.test(new URL(url,location.href).pathname)){
                        window.testAiRequests.push({source:document.querySelector('#annotate').dataset.messageSource,asset:window.workbenchState().assetId});
                    }
                    return fetchOriginal(url,options);
                };
                window.testAiCanvas=document.querySelector('#canvasArea');
            }""")
            fill("#aiEngine","grabcut");click("#runAI")
            wait("document.querySelector('#annotate').dataset.messageSource==='segmentation'")
            assert js("document.querySelector('#canvasArea')===window.testAiCanvas && !document.querySelector('#propertiesSidebar').getClientRects().length")
            wait("!document.querySelector('#acceptAI').disabled")
            assert json.loads(js("JSON.stringify(window.testAiRequests[0])")) == {"source": "segmentation", "asset": changed_id}
            js("document.querySelector('#canvasArea').dispatchEvent(new KeyboardEvent('keydown',{key:'Delete',bubbles:true}))")
            assert js("document.querySelectorAll('#shapeList .shape-row').length===2")
            assert service.store.get_asset(pid,changed_id)['shape_count']==2,"AI silently committed its candidate"
            capture("03-ai-messages-segmentation")
            click("#acceptAI");click("#saveNow");wait("!window.workbenchState().dirty")
            assert service.store.get_asset(pid,changed_id)['shape_count']==3
            click("#returnAnnotationEditor")
            assert js(f"document.querySelector('#annotationMessagesPanel').hidden && window.workbenchState().assetId==={json.dumps(changed_id)} && document.querySelector('#aiEngine').value==='grabcut'")
            assert js(class_selected),"Default class must persist after creation"
            # Native Qt pointer events draw a rectangle without leaving AI assistance.
            click(class_choice)
            click("#addBoundingBox")
            coords=json.loads(js("JSON.stringify((()=>{const r=document.querySelector('#overlay').getBoundingClientRect();return [r.left+45/640*r.width,r.top+40/480*r.height,r.left+100/640*r.width,r.top+75/480*r.height]})())"))
            start=QPoint(round(coords[0]),round(coords[1]));end=QPoint(round(coords[2]),round(coords[3]))
            target=view.focusProxy() or view
            start=target.mapFrom(view,start);end=target.mapFrom(view,end)
            QTest.mousePress(target,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,start)
            QTest.mouseMove(target,end,50)
            QTest.mouseRelease(target,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,end)
            click("#saveNow");wait("!window.workbenchState().dirty")
            assert service.store.get_asset(pid,changed_id)['shape_count']==4,"Canvas rectangle gesture did not persist"
            click("#editBoundingBox")
            assert js("document.querySelector('#editBoundingBox').classList.contains('active') && !document.querySelector('#annotationAiPanel').hidden")
            click("#annotationObjectsTab")
            for width,height in ((1280,720),(1920,1080)):
                view.resize(width,height);QTest.qWait(200)
                capture(f"annotation-{width}x{height}")
                box=js("JSON.stringify(document.querySelector('#canvasArea').getBoundingClientRect().toJSON())")
                bounds=json.loads(box);assert bounds['height']>height*.4 and bounds['width']>350,bounds
            view.resize(1440,900)
            click("[data-stage='review']");wait("!document.querySelector('#review').hidden")
            fill("#reviewAnnotationFilter","unannotated")
            wait("document.querySelector('#reviewPageLabel').innerText.includes('0 張')")
            assert js("document.querySelector('#reviewGrid').innerText.includes('沒有符合篩選條件')")
            fill("#reviewAnnotationFilter","annotated")
            wait("document.querySelectorAll('#reviewGrid .review-card').length===4")
            assert js("[...document.querySelectorAll('#reviewGrid .review-card-body>small:first-of-type')].every(e=>e.innerText.includes('有標註'))")
            click("#reviewSelectAll");click("#reviewApprove")
            wait("document.querySelector('#reviewStats').innerText.includes('4')")
            wait("!window.workbenchState().busy")
            assert service.store.get_project(pid)['stats']['approved']==4
            capture("04-review-approved")
            # Build and run a real immutable training version from the same UI.
            click("#prepareTraining")
            wait("window.workbenchState().stage==='split'&&!window.workbenchState().transitioning")
            assert js("document.querySelector('[data-stage=split]').classList.contains('active')")
            capture("05-data-split")
            click("#openSplitFlowManager")
            wait("document.querySelector('#smartSplitDialog').open")
            # This fixture contains independent imported drawings, so use the
            # explicit class-balanced option rather than splitting camera footage.
            fill("#splitPurpose","reviewed_independent")
            fill("#splitStrategy","hybrid")
            click("#previewSmartSplit")
            wait("!document.querySelector('#applySmartSplit').disabled")
            click("#applySmartSplit")
            wait("!document.querySelector('#smartSplitDialog').open")
            wait("document.querySelector('#splitFlowStatus').innerText.includes('符合建立固定資料版本')")
            split_assets=service.store.snapshot(pid)['assets']
            changed_split=next(asset['split'] for asset in split_assets if asset['id']==changed_id)
            if changed_split=='train':
                replacement=next(asset for asset in split_assets if asset['split'] in {'val','test'})
                service.store.assign(pid,[changed_id],split=replacement['split'])
                service.store.assign(pid,[replacement['id']],split='train')
                click("#refreshSplitPage")
                wait("document.querySelector('#splitFlowStatus').innerText.includes('符合建立固定資料版本')")
            click("#continueToTraining")
            wait("window.workbenchState().stage==='train'&&!window.workbenchState().transitioning")
            click("#createDatasetVersion")
            wait("document.querySelector('#trainingDatasetCurrent').innerText==='D001'")
            fill("#trainingEngine","pixel_prototype_v1");fill("#trainingEpochs","4");click("#startTraining")
            wait("document.querySelector('#trainingRunDetail').innerText.includes('已完成')",seconds=60)
            click('[data-chart-group="all"]')
            wait("document.querySelectorAll('#trainingPlots circle[data-run-id]').length>=8")
            assert len(service.training.run_metrics(pid,'R001')['metrics'])==4
            assert js("!!document.querySelector('#trainingPlots [data-metric-key=threshold]')")
            assert js("document.querySelector('#trainingRunDetail .metric-chart-note').innerText.includes('X 軸涵蓋完整 Run')")
            assert js("[...document.querySelectorAll('#trainingPlots svg')].every(svg=>svg.dataset.xMax==='4')")
            assert js("new Set([...document.querySelectorAll('#trainingPlots circle[data-epoch]')].map(e=>e.dataset.epoch)).size===4")
            assert (service.training.datasets_dir(pid)/'D001'/'manifest.json').is_file()
            assert (service.training.models_dir(pid)/'M001'/'model.json').is_file()
            capture("05-training-completed")
            js("document.querySelector('#trainingRunDetail').scrollIntoView({block:'start'})");QTest.qWait(150);capture("05b-training-plot")
            training_manifest=json.loads((service.training.datasets_dir(pid)/'D001'/'manifest.json').read_text(encoding='utf-8'))
            training_asset=next(asset['asset_id'] for asset in training_manifest['assets'] if asset['split']=='train')
            click("[data-stage='annotate']")
            click(f"#assetList [data-asset-id='{training_asset}']")
            wait(f"window.workbenchState().assetId==={json.dumps(training_asset)}")
            click("[data-stage='models']")
            wait("document.querySelector('#modelDetail').innerText.includes('M001')")
            assert js("document.querySelector('#modelDetail').innerText.includes('匯出模型封裝')")
            assert js("[...document.querySelector('#modelDatasetFilter').options].some(option=>option.value==='D001')")
            assert js("Math.abs(document.querySelector('.model-browser .model-section-heading').getBoundingClientRect().top-document.querySelector('#modelDetail .model-section-heading').getBoundingClientRect().top)<2")
            js("document.querySelector('.model-parameters').open=true")
            assert js("document.querySelector('.model-parameters').innerText.includes('訓練輪數')")
            capture("05c-model-version-filters")
            click("[data-stage='annotate']")
            wait("document.querySelector('#annotationModel').value==='M001'")
            click("#annotationAiTab")
            click("#annotationPredictionTab")
            wait("!document.querySelector('#annotationPredictionPanel').hidden")
            assert js("!document.querySelector('#annotationPredictionPanel #annotationPredictionList')")
            click("#generateCurrentPrediction")
            wait("document.querySelector('#annotate').dataset.messageSource==='prediction'")
            wait("document.querySelector('#annotationPredictionList').textContent.includes('M001')",seconds=60)
            assert json.loads(js("JSON.stringify(window.testAiRequests.at(-1))")) == {"source": "prediction", "asset": training_asset}
            generated_candidate=service.training.list_predictions(pid)[0]
            assert generated_candidate['assets'][0]['status'] in {'candidate','empty'}
            if generated_candidate['assets'][0]['status']=='candidate':
                click("#annotationPredictionList button:first-of-type")
                click("#annotationPredictionList button:last-of-type")
                wait("!document.querySelector('#annotationPredictionList').innerText.includes('接受目前圖片')")
                pending_after_prediction=service.store.get_project(pid)['stats']['pending']
                assert pending_after_prediction==1,pending_after_prediction
            capture("06-model-and-prediction")
            click("#returnAnnotationEditor")
            assert js(f"document.querySelector('#annotationMessagesPanel').hidden && window.workbenchState().assetId==={json.dumps(training_asset)}")
            if service.store.get_project(pid)['stats']['pending']:
                click("[data-stage='review']");click("#reviewSelectAll");click("#reviewApprove")
                wait("[...document.querySelectorAll('#reviewStats .stat-chip')].some(x=>x.querySelector('span').textContent==='已核准' && x.querySelector('b').textContent==='4')")
                wait("!window.workbenchState().busy")
            assert service.store.get_project(pid)['stats']['approved']==4
            click("[data-stage='models']");click("#openExchange");click("#validateProject")
            wait("document.querySelector('#validationReport').innerText.includes('4')")
            wait("!document.querySelector('#validateProject').disabled")
            click("#exportProject")
            wait("document.querySelector('#exportHistory').innerText.includes('v1')",seconds=60)
            project=service.store.get_project(pid)
            assert len(project['exports'])==1,project['exports']
            exported=project['exports'][0]
            assert Path(exported['path']).is_dir() and Path(exported['zip_path']).is_file()
            capture("07-validated-export")
            click("[data-stage='annotate']")
            click('#dataExport')
            wait("!document.querySelector('#export').hidden && document.querySelector('#exportFormat').value==='yolo_detection'")
            assert js("document.querySelector('#backFromExport').innerText.includes('標註編輯')")
            assert js("!document.querySelector('#portableExportHint').hidden")
            assert js("document.querySelector('#exportHistory').innerText.includes('ZIP 所在資料夾')")
            click('#backFromExport')
            wait("!document.querySelector('#annotate').hidden")
            # Reload entire page; choose the saved project, all data must remain.
            js("document.documentElement.dataset.workflowReload='old'")
            page.triggerAction(QWebEnginePage.WebAction.Reload)
            wait("document.readyState==='complete' && !document.documentElement.dataset.workflowReload && typeof window.workbenchState==='function'")
            wait("document.querySelector('#projectGrid').innerText.includes('整合流程驗收')")
            assert service.store.get_asset(pid,changed_id)['shape_count']==4
            assert service.store.get_project(pid)['stats']['approved']==4
            # The destructive action is separate from opening the card, gives
            # a complete warning, supports cancel, and clears an open project.
            click('.project-card-open');wait(f"window.workbenchState().projectId==={json.dumps(pid)}")
            click('#home');wait("!document.querySelector('#library').hidden")
            click('.project-delete');wait("document.querySelector('#formDialog').open")
            assert js("document.querySelector('#dialogBody').innerText.includes('4 張影像')")
            assert js("document.querySelector('#dialogBody').innerText.includes('已匯出的資料集會保留')")
            capture('08-delete-confirmation')
            click('#cancelDialog');wait("!document.querySelector('#formDialog').open")
            assert service.store.get_project(pid)['id']==pid
            click('.project-delete');wait("document.querySelector('#formDialog').open")
            js("document.querySelector('#dialogForm').requestSubmit()")
            wait("window.workbenchState().projectId===null && !document.querySelector('.project-card')")
            assert Path(exported['path']).is_dir() and Path(exported['zip_path']).is_file()
            capture('09-project-deleted')
            assert not page.errors,page.errors
            (output/'report.json').write_text(json.dumps({'success':True,'steps':steps,'project_images':4,
                'approved':4,'exported':4,'page_errors':page.errors},ensure_ascii=False,indent=2),encoding='utf-8')
            print("DESKTOP_WORKFLOW_OK",flush=True)
        except Exception:
            view.grab().save(str(output/'failure.png'))
            (output/'failure-state.txt').write_text(str(js("document.body.innerText")),encoding='utf-8')
            raise
        finally:
            view.close();page.deleteLater();QTest.qWait(50);service.close()


if __name__=='__main__':main()
