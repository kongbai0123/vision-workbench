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

        try:
            wait("typeof window.workbenchState==='function'")
            # Document the UI contract without touching model files in the real checkout.
            js("""{
                const originalFetch=window.fetch.bind(window);
                window.testAnnotationInstalled=false;
                window.fetch=async (url, options)=>{
                    const pathname=new URL(url,location.href).pathname;
                    const json=data=>new Response(JSON.stringify(data),{headers:{'Content-Type':'application/json'}});
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
                    const customLocation=Boolean(window.testAnnotationCustomLocation);
                    catalog.annotation_models={install_path:'models/annotation-assist',custom_location:customLocation,installable:!customLocation,components:[
                        {id:'mm',name:'MM-Grounding-DINO-Tiny',install_path:'models/annotation-assist/MM-Grounding-DINO-Tiny',state:'available'},
                        {id:'qwen',name:'Qwen3-VL-4B-Instruct',install_path:'models/annotation-assist/Qwen3-VL-4B-Instruct',state:window.testAnnotationInstalled?'available':'missing'},
                        {id:'sam',name:'SAM 2.1',install_path:'models/annotation-assist/sam2.1-hiera-tiny',state:'incomplete'},
                    ]};
                    return json(catalog);
                };
            }""")
            js("document.querySelector('#settings').click()")
            wait("!document.querySelector('#settingsShell').hidden")
            assert js("document.querySelectorAll('.settings-nav [data-settings-page]').length===4")
            assert js("document.querySelector('[data-settings-page=general]').getAttribute('aria-current')==='page'")
            wait("document.querySelector('[data-settings-panel=general] #computeSummary').children.length===4")
            assert js("document.querySelector('[data-settings-panel=general] #settingScale') && !!document.querySelector('[data-settings-panel=general] #settingDevice')")
            js("document.querySelector('[data-settings-page=models]').click()")
            wait("document.querySelectorAll('#settingsModelList .settings-model-row').length>=15")
            count = js("document.querySelectorAll('#settingsModelList .settings-model-row').length")
            assert count >= 15, count
            wait("!document.querySelector('#settingsAnnotationModels').hidden && !document.querySelector('#startAnnotationModelInstall').disabled")
            assert js("document.querySelector('#settingsAnnotationModelLocation code').textContent==='models/annotation-assist'")
            assert js("[...document.querySelectorAll('#settingsAnnotationModelList .settings-state')].map(node=>node.textContent).join(',')==='已找到檔案,未找到,檔案不完整'")
            js("document.querySelector('#startAnnotationModelInstall').click()")
            wait("document.querySelector('#formDialog').open && !!document.querySelector('#annotationModelSource')")
            assert js("!document.querySelector('.annotation-model-source-row button')")
            model_source = str(Path(folder) / "備份模型資料夾")
            js(f"document.querySelector('#annotationModelSource').value={json.dumps(model_source)};document.querySelector('#dialogForm').requestSubmit()")
            wait("window.testAnnotationInstalled && document.querySelector('#settingsJobProgress').hidden && document.querySelector('#settingsJobText').textContent==='標註輔助模型檔案已安裝'")
            assert js("window.testAnnotationInstallSource") == model_source
            wait("document.querySelectorAll('#settingsAnnotationModelList .settings-state.ready').length===2")
            js("window.testAnnotationCustomLocation=true;document.querySelector('#refreshModelCatalog').click()")
            wait("document.querySelector('#startAnnotationModelInstall').disabled && document.querySelector('#settingsAnnotationModelLocation').textContent.includes('自訂位置')")
            js("[...document.querySelectorAll('#settingsModelList .settings-model-row')].find(e=>e.innerText.includes('YOLO26n Seg')).click()")
            assert js("document.querySelector('#settingsModelDetail .catalog-facts code')?.textContent==='.venv-models/ultralytics'")
            js("[...document.querySelectorAll('#settingsModelList .settings-model-row')].find(e=>e.innerText.includes('EfficientAD')).click()")
            wait("document.querySelector('#settingsModelDetail').innerText.includes('整合開發中')")
            QTest.qWait(200); view.grab().save(str(output / "10-settings-model-center.png"))
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
            wait("document.querySelector('#settingsContent').scrollTop>0")
            assert not js("document.documentElement.scrollWidth>innerWidth+2")
            print("DESKTOP_SETTINGS_OK", json.dumps({"models": count}, ensure_ascii=False))
        finally:
            view.close(); service.close()


if __name__ == "__main__": main()
