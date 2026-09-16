"""Verify multilabel split controls and preview/apply using an isolated project."""
import json
import tempfile
import time
from pathlib import Path
from desktop_workflow import QApplication, QWebEngineView, Page, QUrl, QEventLoop, QTimer, QTest, Image, WorkbenchService


def main():
    app=QApplication.instance() or QApplication([])
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory);service=WorkbenchService(root/'data').start()
        pid=service.store.create_project('Multilabel UI')['id']
        records=[]
        for i in range(12):
            image=root/f'{i}.png';Image.new('RGB',(100,80),(20+i*10,90,100)).save(image)
            labels=['common']+(['rare'] if i<3 else [])
            shapes=[{'id':str(j),'type':'rectangle','label':label,'x':10+j*20,'y':10,'width':15,'height':15} for j,label in enumerate(labels)]
            records.append({'path':image,'shapes':shapes,'batch_id':'one-session','review_state':'approved'})
        service.store.add_assets(pid,records)
        view=QWebEngineView();page=Page(view);view.setPage(page);view.resize(1440,1000);view.show();view.setUrl(QUrl(service.url))
        def js(script):
            result=[];loop=QEventLoop();page.runJavaScript(script,lambda value:(result.append(value),loop.quit()))
            QTimer.singleShot(10000,loop.quit);loop.exec();return result[0] if result else None
        def wait(expression):
            end=time.monotonic()+30
            while time.monotonic()<end:
                if js(f'Boolean({expression})'):return
                QTest.qWait(80)
            raise AssertionError(expression+'\n'+str(js('document.body.innerText.slice(-2000)')))
        def click(selector):
            wait(f"document.querySelector({json.dumps(selector)}) && !document.querySelector({json.dumps(selector)}).disabled")
            js(f'document.querySelector({json.dumps(selector)}).click()')
        try:
            wait("document.querySelector('.project-card-open')")
            click('.project-card-open');wait("!document.querySelector('#home').disabled")
            click('[data-stage="review"]');wait("document.querySelector('#confirmIndependentAssets')")
            click('#confirmIndependentAssets');wait("document.querySelector('#formDialog').open")
            click('#confirmDialog');wait("!document.querySelector('#formDialog').open && document.querySelector('#confirmIndependentAssets').checked")
            click('[data-stage="split"]');wait("!document.querySelector('#home').disabled")
            click('#openSplitFlowManager');wait("document.querySelector('#smartSplitDialog').open && !document.querySelector('#previewSmartSplit').disabled")
            assert js("document.querySelector('#splitPurpose').value==='reviewed_independent' && document.querySelector('#splitStrategy').value==='hybrid' && !document.querySelector('#splitSourceIsolation').checked")
            js("document.querySelector('#splitPurpose').value='formal';document.querySelector('#splitPurpose').dispatchEvent(new Event('change'))")
            click('#previewSmartSplit');wait("document.querySelector('.split-manager-message').textContent.includes('只有 1')")
            assert js("document.querySelector('#applySmartSplit').disabled")
            js("document.querySelector('#splitPurpose').value='reviewed_independent';document.querySelector('#splitPurpose').dispatchEvent(new Event('change'))")
            click('#previewSmartSplit');wait("!document.querySelector('#applySmartSplit').disabled")
            assert js("document.querySelector('#smartSplitDialog').innerText.includes('總圖片／物件')")
            assert js("document.querySelector('#smartSplitDialog').innerText.includes('來源群組不足')")
            assert all(not a['split'] for a in service.store.get_project(pid)['assets'])
            page.setVisible(True);view.resize(1438,998);QTest.qWait(500)
            output=Path(__file__).resolve().parents[1]/'qa-output'/'desktop';output.mkdir(parents=True,exist_ok=True)
            view.grab().save(str(output/'multilabel-preview.png'))
            for width in (1024,760):
                view.resize(width,1000);QTest.qWait(150)
                assert not js('document.documentElement.scrollWidth>innerWidth+2')
            click('#applySmartSplit');wait("!document.querySelector('#smartSplitDialog').open")
            project=service.store.get_project(pid)
            assert project['split_plan']['purpose']=='reviewed_independent'
            assert project['split_plan']['source_isolation'] is False
            assert all(a['split'] for a in project['assets'])
            click('#openSplitFlowManager');wait("document.querySelector('#smartSplitDialog').open && !document.querySelector('#previewSmartSplit').disabled")
            js("document.querySelector('#splitPurpose').value='experimental';document.querySelector('#splitPurpose').dispatchEvent(new Event('change'))")
            assert js("document.querySelector('#splitSourceIsolation').disabled && !document.querySelector('#splitSourceIsolation').checked")
            click('#previewSmartSplit');wait("!document.querySelector('#applySmartSplit').disabled")
            assert js("document.querySelector('.split-manager-message').textContent.includes('寬鬆分割')")
            click('#applySmartSplit');wait("!document.querySelector('#smartSplitDialog').open")
            assert service.store.get_project(pid)['split_plan']['strategy']=='random_loose'
            assert not page.errors,page.errors
            print('PASS: source isolation, multiclass preview metrics, read-only preview, responsive layout and apply',flush=True)
        finally:
            view.close();service.close();QTest.qWait(100)


if __name__=='__main__':main()
