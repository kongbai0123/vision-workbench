"""Desktop Chromium smoke test for the settings and model catalog."""
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--disable-gpu --disable-background-timer-throttling")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtCore import QEventLoop, QTimer, QUrl
from PySide6.QtTest import QTest
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import QApplication

from workbench.model_registry import ModelRegistry
from workbench.server import WorkbenchService
from workbench.training import TrainingWorkspace


def main():
    app = QApplication.instance() or QApplication([])
    output = Path(__file__).resolve().parents[1] / "qa-output" / "desktop"
    output.mkdir(parents=True, exist_ok=True)
    runtime = {"state": "ready", "message": "合成測試環境", "python": sys.executable}
    with (
        tempfile.TemporaryDirectory(prefix="vision-settings-") as folder,
        patch.object(TrainingWorkspace, "_probe_maskrcnn", return_value=False),
        patch.object(ModelRegistry, "_probe_torchvision", return_value=runtime),
        patch.object(ModelRegistry, "_probe_python_component", return_value=runtime),
    ):
        service = WorkbenchService(Path(folder) / "data").start()
        service.store.create_project("模型目錄互動測試")
        view = QWebEngineView(); view.resize(1440, 900); view.setUrl(QUrl(service.entry_url)); view.show()
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

        def assert_catalog_layout(*, equal_height=True):
            geometry = json.loads(js("""JSON.stringify((()=>{
                const rect=selector=>document.querySelector(selector).getBoundingClientRect().toJSON();
                const list=document.querySelector('#settingsModelList');
                const content=document.querySelector('#settingsContent');
                return {
                    sidebar:rect('.model-catalog-sidebar'),detail:rect('#settingsModelDetail'),
                    action:rect('#settingsModelActions button'),content:rect('#settingsContent'),drawer:rect('.settings-drawer'),
                    listOverflow:getComputedStyle(list).overflowY,
                    listScrollHeight:list.scrollHeight,listHeight:list.clientHeight,
                    outerOverflow:content.scrollHeight-content.clientHeight,
                    horizontalOverflow:content.scrollWidth-content.clientWidth,
                    viewportWidth:innerWidth,
                    filtersInSidebar:document.querySelector('.model-catalog-sidebar').contains(document.querySelector('#modelCatalogSearch'))
                       && document.querySelector('.model-catalog-sidebar').contains(document.querySelector('#modelCatalogFilter')),
                };
            })())"""))
            sidebar, detail, action, content = (geometry[k] for k in ("sidebar", "detail", "action", "content"))
            assert geometry["filtersInSidebar"], geometry
            assert geometry["horizontalOverflow"] <= 2, geometry
            assert geometry["drawer"]["left"] >= -1 and geometry["drawer"]["right"] <= geometry["viewportWidth"] + 1, geometry
            assert geometry["outerOverflow"] <= 2, geometry
            assert geometry["listOverflow"] in ("auto", "scroll"), geometry
            assert geometry["listScrollHeight"] > geometry["listHeight"] > 0, geometry
            assert action["height"] > 0 and content["top"] <= action["top"] and action["bottom"] <= content["bottom"] + 1, geometry
            assert 0 <= detail["bottom"] - action["bottom"] <= 28, geometry
            if equal_height:
                assert sidebar["right"] <= detail["left"] + 1, geometry
                assert abs(sidebar["top"] - detail["top"]) <= 1 and abs(sidebar["bottom"] - detail["bottom"]) <= 1, geometry
            else:
                assert sidebar["bottom"] <= detail["top"] + 1, geometry

        def reveal_selected_model():
            assert js("""(()=>{
                const list=document.querySelector('#settingsModelList');
                const selection=list.querySelector('.settings-model-row.active');
                const listBounds=list.getBoundingClientRect();
                list.scrollTop+=selection.getBoundingClientRect().top-listBounds.top-6;
                const selectedBounds=selection.getBoundingClientRect();
                return selectedBounds.top>=listBounds.top && selectedBounds.bottom<=listBounds.bottom;
            })()""")

        try:
            wait("typeof window.workbenchState==='function'")
            # Exercise hardware and annotation states independently of the local machine.
            js("""{
                const originalFetch=window.fetch.bind(window);
                window.testAnnotationInstalled=false;
                window.testHardwareRequests=0;
                window.fetch=async (url, options)=>{
                    const pathname=new URL(url,location.href).pathname;
                    const json=(data,status=200)=>new Response(JSON.stringify(data),{status,headers:{'Content-Type':'application/json'}});
                    if(pathname==='/api/system/hardware'){
                        window.testHardwareRequests++;
                        if(window.testHardwareError)return json({error:'測試硬體暫時無法讀取'},503);
                        const GiB=1024**3;
                        if(window.testAlternateHardware)return json({
                            cpu:{name:'AMD Ryzen 7 7840U',logical_cores:16},memory:{total_bytes:16*GiB},gpus:[],
                        });
                        return json({cpu:{name:'Intel Core i7-13700',logical_cores:24},memory:{total_bytes:32*GiB},gpus:[
                            {name:'NVIDIA GeForce RTX 4070',memory_bytes:12*GiB},
                            {name:'Intel UHD Graphics 770',memory_bytes:null},
                        ]});
                    }
                    if(pathname==='/api/model-components/annotation-assist/install'){
                        window.testAnnotationInstallSource=JSON.parse(options.body).source;
                        return json({id:'test-annotation-install',state:'running',progress:40});
                    }
                    if(pathname==='/api/jobs/test-annotation-install'){
                        window.testAnnotationInstalled=true;
                        return json({id:'test-annotation-install',state:'succeeded',progress:100});
                    }
                    const response=await originalFetch(url,options);
                    if(pathname!=='/api/model-catalog')return response;
                    const catalog=await response.json();
                    for(const model of catalog.engines){
                        if(!['pixel_prototype_v1','yolo26s_seg'].includes(model.key))model.description+=' 版面測試。';
                        if(model.key==='yolo26n_seg')model.description+=' 模型說明需要捲動才能完整查看。'.repeat(80);
                    }
                    const customLocation=Boolean(window.testAnnotationCustomLocation);
                    catalog.annotation_models={install_path:'models/annotation-assist',custom_location:customLocation,installable:!customLocation,components:[
                        {id:'runtime',name:'批量標註 CUDA Python',install_path:'models/annotation-assist/runtime',state:'available'},
                        {id:'mm-tiny',name:'MM Grounding DINO Tiny',install_path:'models/annotation-assist/MM-Grounding-DINO-Tiny',state:'available'},
                        {id:'qwen-4b',name:'Qwen3-VL 4B',install_path:'models/annotation-assist/Qwen3-VL-4B-Instruct',state:window.testAnnotationInstalled?'available':'missing'},
                        {id:'batch-sam',name:'SAM 2.1 批次框界檢查',install_path:'models/annotation-assist/sam2.1-hiera-tiny',state:'incomplete'},
                        {id:'gemma-4',name:'Gemma 4 觀察覆核',install_path:'models/annotation-assist/Gemma-4-E4B-it-QAT-Q4',state:'missing'},
                        {id:'gemma-runtime',name:'Gemma 4 CUDA 執行庫',install_path:'models/annotation-assist/_runtimes/llama.cpp',state:'missing'},
                        {id:'interactive-sam',name:'SAM 2.1 互動分割權重',install_path:'models/sam2.1-hiera-tiny',state:'missing'},
                    ]};
                    return json(catalog);
                };
            }""")
            js("document.querySelector('#settings').click()")
            wait("!document.querySelector('#settingsShell').hidden")
            assert js("document.querySelectorAll('.settings-nav [data-settings-page]').length===4")
            assert js("document.querySelector('[data-settings-page=general]').getAttribute('aria-current')==='page'")
            assert js("""(()=>{
                const compute=document.querySelector('[data-settings-panel=general] #settingsComputeSection');
                return compute.contains(document.querySelector('#settingsHardwareSummary')) &&
                    compute.contains(document.querySelector('#settingsHardwareRefresh')) &&
                    !document.querySelector('[data-settings-page=hardware], [data-settings-panel=hardware]');
            })()""")
            wait("window.testHardwareRequests===1 && !document.querySelector('#settingsHardwareSummary').hidden && document.querySelector('#settingsHardwareStatus').hidden")
            wait("document.querySelector('[data-settings-panel=general] #computeSummary').children.length===4")
            assert js("document.querySelector('[data-settings-panel=general] #settingScale') && !!document.querySelector('[data-settings-panel=general] #settingDevice')")
            assert js("document.querySelectorAll('#settingsHardwareSummary > .settings-fact').length===3")
            assert js("document.querySelector('#settingsHardwareCpuName').textContent==='Intel Core i7-13700' && document.querySelector('#settingsHardwareCpuCores').textContent.includes('24') && document.querySelector('#settingsHardwareMemory').textContent==='32 GiB'")
            assert js("""(()=>{
                const [dedicated,integrated,...rest]=document.querySelectorAll('#settingsHardwareGpus li');
                return !rest.length && dedicated.textContent.includes('RTX 4070') && dedicated.textContent.includes('12 GiB')
                    && integrated.textContent.includes('Intel UHD') && integrated.textContent.includes('未讀取');
            })()""")
            QTest.qWait(150)
            assert view.grab().save(str(output / "10f-settings-hardware.png"))
            js("document.querySelector('[data-settings-page=models]').click()")
            wait("document.querySelectorAll('#settingsModelList .settings-model-row').length>=15")
            count = js("document.querySelectorAll('#settingsModelList .settings-model-row').length")
            assert count >= 15, count
            assert js("document.querySelector('#settingsModelTabTraining').getAttribute('aria-selected')==='true' && !document.querySelector('#settingsTrainingModelsPanel').hidden && document.querySelector('#settingsAnnotationModelsPanel').hidden")
            assert_catalog_layout()
            QTest.qWait(150)
            assert view.grab().save(str(output / "10-settings-model-center.png"))
            js("""{
                const search=document.querySelector('#modelCatalogSearch');
                search.value='版面測試';
                search.dispatchEvent(new Event('input',{bubbles:true}));
                const filter=document.querySelector('#modelCatalogFilter');
                filter.value='ready';
                filter.dispatchEvent(new Event('change',{bubbles:true}));
                [...document.querySelectorAll('#settingsModelList .settings-model-row')].find(row=>row.textContent.includes('YOLO26n Seg')).click();
            }""")
            assert js("document.querySelector('#settingsModelDetail .catalog-facts code')?.textContent==='.venv-models/ultralytics'")
            assert js("document.querySelector('#settingsModelActions button').textContent.includes('回到訓練')")
            assert_catalog_layout()
            scroll_positions = json.loads(js("""JSON.stringify((()=>{
                const list=document.querySelector('#settingsModelList'),detail=document.querySelector('#settingsModelDetailBody');
                list.scrollTop=100; detail.scrollTop=120;
                return [list.scrollTop,detail.scrollTop];
            })())"""))
            assert min(scroll_positions) > 0, scroll_positions
            assert_catalog_layout()
            js("""{
                const tab=document.querySelector('#settingsModelTabTraining');tab.focus();
                tab.dispatchEvent(new KeyboardEvent('keydown',{key:'ArrowRight',bubbles:true}));
            }""")
            wait("!document.querySelector('#settingsAnnotationModels').hidden && !document.querySelector('#startAnnotationModelInstall').disabled")
            assert js("document.querySelector('#settingsModelTabAnnotation').getAttribute('aria-selected')==='true' && !document.querySelector('#settingsAnnotationModelsPanel').hidden && document.querySelector('#settingsTrainingModelsPanel').hidden")
            assert js("document.activeElement.id==='settingsModelTabAnnotation'")
            assert js("document.querySelector('#settingsAnnotationModelLocation code').textContent==='models/annotation-assist'")
            assert js("document.querySelectorAll('#settingsAnnotationModelList .settings-state').length===6 && document.querySelector('#settingsInteractiveModelList').textContent.includes('SAM 2.1 互動分割權重')")
            assert js("document.querySelector('#settingsInteractiveModelList code')?.textContent==='models/sam2.1-hiera-tiny'")
            assert js("""(()=>{
                const states=[...document.querySelectorAll('#settingsAnnotationModelList .settings-state')].map(node=>node.textContent);
                const cards=[...document.querySelector('#settingsAnnotationModelList').children].map(card=>card.getBoundingClientRect());
                return ['已找到檔案','未找到','檔案不完整'].every(state=>states.includes(state)) &&
                    cards[0].right<=cards[1].left+1 && Math.abs(cards[0].top-cards[1].top)<=1;
            })()""")
            QTest.qWait(150)
            assert view.grab().save(str(output / "10b-settings-annotation-models.png"))
            js("document.querySelector('#settingsModelTabAnnotation').dispatchEvent(new KeyboardEvent('keydown',{key:'ArrowLeft',bubbles:true}))")
            wait("!document.querySelector('#settingsTrainingModelsPanel').hidden")
            assert js("document.activeElement.id==='settingsModelTabTraining'")
            assert js("document.querySelector('#modelCatalogSearch').value==='版面測試' && document.querySelector('#modelCatalogFilter').value==='ready'")
            assert js("document.querySelector('#settingsModelList .settings-model-row.active').textContent.includes('YOLO26n Seg') && document.querySelector('#settingsModelDetail h3').textContent.includes('YOLO26n Seg')")
            assert js("!document.querySelector('#settingsModelList').textContent.includes('EfficientAD') && !document.querySelector('#settingsModelList').textContent.includes('內建基準')")
            assert json.loads(js("JSON.stringify([document.querySelector('#settingsModelList').scrollTop,document.querySelector('#settingsModelDetailBody').scrollTop])")) == scroll_positions
            assert_catalog_layout()
            assert js("window.testHardwareRequests===1 && !document.querySelector('#settingsHardwareCpuName').getClientRects().length")
            js("document.activeElement.blur();document.querySelector('[data-settings-page=general]').click()")
            hardware_return = json.loads(js("""JSON.stringify({
                requests:window.testHardwareRequests,
                generalSelected:document.querySelector('[data-settings-page=general]').getAttribute('aria-current')==='page',
                hasHardwareRows:document.querySelector('#settingsHardwareCpuName').getClientRects().length>0,
                modelsHidden:document.querySelector('[data-settings-panel=models]').hidden,
                trainingTabSelected:document.querySelector('#settingsModelTabTraining').getAttribute('aria-selected')==='true',
            })"""))
            assert hardware_return == {
                "requests": 1, "generalSelected": True, "hasHardwareRows": True,
                "modelsHidden": True, "trainingTabSelected": True,
            }, hardware_return
            js("window.testHardwareError=true;document.querySelector('#settingsHardwareRefresh').click()")
            wait("window.testHardwareRequests===2 && document.querySelector('#settingsHardwareStatus').textContent.includes('重新檢查失敗')")
            assert js("!document.querySelector('#settingsHardwareSummary').hidden && document.querySelector('#settingsHardwareCpuName').textContent==='Intel Core i7-13700'")
            js("window.testHardwareError=false;window.testAlternateHardware=true;document.querySelector('#settingsHardwareRefresh').click()")
            wait("window.testHardwareRequests===3 && document.querySelector('#settingsHardwareCpuName').textContent==='AMD Ryzen 7 7840U'")
            assert js("document.querySelector('#settingsHardwareStatus').hidden && document.querySelector('#settingsHardwareMemory').textContent==='16 GiB' && document.querySelector('#settingsHardwareGpus').textContent.includes('未取得顯示卡資訊')")
            view.resize(415,690);QTest.qWait(150)
            assert js("""(()=>{
                const content=document.querySelector('#settingsContent');
                const hardware=document.querySelector('#settingsHardwareSummary');
                const gpuRow=hardware.lastElementChild;
                const bounds=content.getBoundingClientRect();
                content.scrollTop=0;
                const originalEnd=gpuRow.getBoundingClientRect().bottom;
                gpuRow.scrollIntoView({block:'end'});
                const row=gpuRow.getBoundingClientRect();
                return content.scrollWidth-content.clientWidth<=2 && row.bottom<=bounds.bottom+1 &&
                    (originalEnd<=bounds.bottom+1 || content.scrollTop>0) &&
                    row.left>=bounds.left-1 && row.right<=bounds.right+1 &&
                    !/^(auto|scroll)$/.test(getComputedStyle(hardware).overflowY);
            })()""")
            view.resize(1440,900);QTest.qWait(150)
            js("""{
                const device=document.querySelector('#settingDevice');device.value='cuda';
                device.dispatchEvent(new Event('change',{bubbles:true}));
            }""")
            assert js("document.querySelector('#settingDevice').value==='cuda' && localStorage.getItem('vision-workbench.device')==='cuda' && window.testHardwareRequests===3")
            js("document.querySelector('[data-settings-page=models]').click()")
            assert js("!document.querySelector('#settingsHardwareCpuName').getClientRects().length && document.querySelector('#settingsModelTabTraining').getAttribute('aria-selected')==='true' && document.querySelector('#settingsTrainingModelsPanel').getClientRects().length>0")
            assert js("document.querySelector('#modelCatalogSearch').value==='版面測試' && document.querySelector('#modelCatalogFilter').value==='ready' && document.querySelector('#settingsModelList .active').textContent.includes('YOLO26n Seg')")
            assert json.loads(js("JSON.stringify([document.querySelector('#settingsModelList').scrollTop,document.querySelector('#settingsModelDetailBody').scrollTop])")) == scroll_positions
            js("document.querySelector('#settingsModelTabAnnotation').click()")
            js("document.querySelector('#startAnnotationModelInstall').click()")
            wait("document.querySelector('#formDialog').open && !!document.querySelector('#annotationModelSource')")
            assert js("!document.querySelector('.annotation-model-source-row button')")
            model_source = str(Path(folder) / "備份模型資料夾")
            js(f"document.querySelector('#annotationModelSource').value={json.dumps(model_source)};document.querySelector('#dialogForm').requestSubmit()")
            wait("window.testAnnotationInstalled && document.querySelector('#settingsJobProgress').hidden && document.querySelector('#settingsJobText').textContent==='標註輔助模型檔案已安裝'")
            assert js("window.testAnnotationInstallSource") == model_source
            wait("document.querySelectorAll('#settingsAnnotationModelList .settings-state.ready').length===3")
            js("window.testAnnotationCustomLocation=true;document.querySelector('#refreshModelCatalog').click()")
            wait("document.querySelector('#startAnnotationModelInstall').disabled && document.querySelector('#settingsAnnotationModelLocation').textContent.includes('自訂位置')")
            js("""{
                document.querySelector('#settingsModelTabTraining').click();
                const search=document.querySelector('#modelCatalogSearch');search.value='';search.dispatchEvent(new Event('input',{bubbles:true}));
                const filter=document.querySelector('#modelCatalogFilter');filter.value='all';filter.dispatchEvent(new Event('change',{bubbles:true}));
                [...document.querySelectorAll('#settingsModelList .settings-model-row')].find(row=>row.textContent.includes('YOLO26s Seg')).click();
            }""")
            view.resize(1024,680); QTest.qWait(150)
            assert_catalog_layout()
            reveal_selected_model()
            wait("document.querySelector('#toast').hidden")
            QTest.qWait(150)
            assert view.grab().save(str(output / "10c-settings-model-center-short.png"))
            view.resize(465,650); QTest.qWait(150)
            assert_catalog_layout()
            reveal_selected_model()
            assert view.grab().save(str(output / "10d-settings-model-center-narrow.png"))
            view.resize(415,690); QTest.qWait(150)
            assert_catalog_layout(equal_height=False)
            reveal_selected_model()
            assert view.grab().save(str(output / "10e-settings-model-center-stacked.png"))
            js("document.querySelector('#settingsModelTabAnnotation').click()")
            assert js("""(()=>{
                const [first,second]=[...document.querySelector('#settingsAnnotationModelList').children].map(card=>card.getBoundingClientRect());
                return Math.abs(first.left-second.left)<=1 && first.bottom<=second.top+1;
            })()""")
            annotation_scroll = js("""(()=>{
                const panel=document.querySelector('#settingsAnnotationModelsPanel');panel.scrollTop=80;return panel.scrollTop;
            })()""")
            assert annotation_scroll > 0, annotation_scroll
            js("document.querySelector('#settingsModelTabTraining').click();document.querySelector('#settingsModelTabAnnotation').click()")
            assert js("document.querySelector('#settingsAnnotationModelsPanel').scrollTop") == annotation_scroll
            js("document.querySelector('#settingsModelTabTraining').click()")
            view.resize(1440,900); QTest.qWait(150)
            js("[...document.querySelectorAll('#settingsModelList .settings-model-row')].find(e=>e.innerText.includes('EfficientAD')).click()")
            wait("document.querySelector('#settingsModelDetail').innerText.includes('整合開發中')")
            js("document.querySelector('[data-settings-page=updates]').click()")
            wait("!document.querySelector('[data-settings-panel=updates]').hidden")
            wait("document.querySelector('#settingsVersion').textContent.includes('Vision Workbench')")
            wait("document.querySelector('[data-settings-panel=updates] #diagnosticSummary').textContent.includes('Vision Workbench')")
            js("window.workbenchUpdateStatus(3)")
            wait("!document.querySelector('#settingsUpdateNavDot').hidden && !document.querySelector('#settingsUpdateCardDot').hidden")
            assert js("document.querySelector('#desktopUpdateStatus').innerText.includes('3 個程式檔案')")
            assert js("document.querySelector('#openDesktopUpdater').innerText==='套用本機變更並重新啟動'")
            assert js("document.querySelector('#settingsUpdateNavDot').getBoundingClientRect().width > 0")
            assert not js("document.documentElement.scrollWidth>innerWidth+2")
            QTest.qWait(150); view.grab().save(str(output / "11-settings-updates.png"))
            js("document.querySelector('#goSettingsDiagnostics').click()")
            assert js("document.querySelector('#settingsContent').scrollTop>0 && document.activeElement.id==='settingsDiagnosticsTitle'")
            js("navigator.clipboard.writeText=async text=>{window.copiedSettingsSummary=text};document.querySelector('#copyDiagnostics').click()")
            wait("window.copiedSettingsSummary===document.querySelector('#diagnosticSummary').textContent")
            js("delete navigator.clipboard.writeText;document.querySelector('#goModelUpdates').click()")
            assert js("document.querySelector('[data-settings-page=models]').getAttribute('aria-current')==='page'")
            js("document.querySelector('#settingsModelTabAnnotation').click()")
            js("document.querySelector('#closeSettings').click()")
            js("document.querySelector('.project-card-open').click()")
            wait("!document.querySelector('[data-stage=train]').disabled")
            js("document.querySelector('[data-stage=train]').click()")
            wait("document.querySelectorAll('#trainingEngine option').length>=15")
            assert js("[...document.querySelectorAll('#trainingEngine option')].every(option=>!option.disabled)")
            js("document.querySelector('#trainingEngine').value='efficientad';document.querySelector('#trainingEngine').dispatchEvent(new Event('change'))")
            wait("!document.querySelector('#trainingModelNotice').hidden")
            assert js("document.querySelector('#trainingModelNotice').innerText.includes('開發待辦')")
            js("document.querySelector('#openSettingsModels').click()")
            wait("!document.querySelector('[data-settings-panel=models]').hidden && document.querySelector('#settingsModelDetail h3').textContent.includes('EfficientAD')")
            assert js("document.querySelector('#settingsModelTabTraining').getAttribute('aria-selected')==='true'")
            assert not js("document.documentElement.scrollWidth>innerWidth+2")
            js("[...document.querySelectorAll('#settingsModelList .settings-model-row')].find(row=>row.textContent.includes('YOLO26s Seg')).click()")
            wait("document.querySelector('#settingsModelActions button').textContent.includes('回到訓練')")
            js("document.querySelector('#settingsModelActions button').click()")
            wait("document.querySelector('#settingsShell').hidden && window.workbenchState().stage==='train' && document.querySelector('#trainingEngine').value==='yolo26s_seg'")
            print("DESKTOP_SETTINGS_OK", json.dumps({"models": count, "layout_sizes": ["1440x900", "1024x680", "465x650", "415x690"]}, ensure_ascii=False))
        finally:
            view.close(); service.close()


if __name__ == "__main__": main()
