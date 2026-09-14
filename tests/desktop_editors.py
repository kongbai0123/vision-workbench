"""Opt-in real Qt + local CVAT workflow; only edits a new isolated QA project."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
os.environ.setdefault('QTWEBENGINE_CHROMIUM_FLAGS','--disable-gpu --disable-background-timer-throttling --disable-renderer-backgrounding')
import sys,json,time,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from PIL import Image
from PySide6.QtWidgets import QApplication,QMessageBox
from PySide6.QtCore import QEventLoop,QTimer
from PySide6.QtTest import QTest
from workbench.desktop import MainWindow,DialogBridge
from workbench.server import WorkbenchService,APP_ROOT
from workbench.cvat_bridge import CvatProjectBridge
from cvat_setup import CvatSetup


def main():
    app=QApplication([])
    output=APP_ROOT/'qa-output'/'editors';output.mkdir(parents=True,exist_ok=True)
    setup=CvatSetup()
    script="from django.test import Client; from django.contrib.auth import get_user_model; from django.middleware.csrf import _get_new_csrf_string; import json; u=get_user_model().objects.get(username='hub_f8b8851da070'); c=Client(); c.force_login(u,backend='django.contrib.auth.backends.ModelBackend'); print('SESSION_JSON:'+json.dumps([{'name':'sessionid','value':c.cookies['sessionid'].value},{'name':'csrftoken','value':_get_new_csrf_string()}]))"
    response=setup._compose('exec','--no-TTY','cvat_server','python','manage.py','shell','-c',script,timeout=30,private=True)
    cookies=json.loads(next(line.split('SESSION_JSON:',1)[1] for line in response.stdout.splitlines() if line.startswith('SESSION_JSON:')))
    class ReadyCvat:
        def launch(self):return {'auth_cookies':cookies,'url':'http://cvat.localhost:8768/'}
        def status(self):return {'ready':True,'busy':False,'steps':[]}
        def stop(self):pass
    with tempfile.TemporaryDirectory(prefix='vision-editor-qa-',ignore_cleanup_errors=True) as tmp:
        root=Path(tmp);service=WorkbenchService(root/'data').start();service._cvat=ReadyCvat()
        pid=service.store.create_project('Workbench editor QA '+root.name)['id']
        image=root/'one.png';Image.new('RGB',(160,120),'#b3c7d1').save(image)
        aid=service.store.add_assets(pid,[dict(path=image,shapes=[dict(id='original',type='rectangle',label='part',x=20,y=20,width=50,height=50)])])['asset_ids'][0]
        service.store.review(pid,[aid],'approved',{aid:1})
        window=MainWindow(service,DialogBridge());window.show()
        failures=[]
        original_warning=QMessageBox.warning
        QMessageBox.warning=lambda parent,title,message:failures.append(title+': '+message)
        def js(script,page=None):
            result=[];loop=QEventLoop();(page or window.page).runJavaScript(script,lambda x:(result.append(x),loop.quit()))
            QTimer.singleShot(10000,loop.quit);loop.exec()
            if not result:raise AssertionError('JS timeout')
            return result[0]
        def wait(check,seconds=60):
            end=time.monotonic()+seconds
            while time.monotonic()<end:
                if failures:raise AssertionError(failures)
                if check():return
                QTest.qWait(100)
            raise AssertionError('Timed out: '+str(js('document.body.innerText.slice(-1500)')))
        def select(mode):
            wait(lambda:js("!document.querySelector('#editorSelector').disabled"))
            js(f"{{const e=document.querySelector('#editorSelector');e.value={json.dumps(mode)};e.dispatchEvent(new Event('change'));}}")
        def annotate():
            wait(lambda:js("!document.querySelector('[data-stage=annotate]').disabled"))
            js("document.querySelector('[data-stage=annotate]').click()")
            wait(lambda:js("!document.querySelector('#annotate').hidden && !document.querySelector('#editorSelector').disabled"))
        try:
            wait(lambda:js("!!document.querySelector('.project-card-open')"))
            js("document.querySelector('.project-card-open').click()")
            wait(lambda:js(f"window.workbenchState?.().projectId==={json.dumps(pid)}"))
            annotate()
            wait(lambda:js("!document.querySelector('#annotate').hidden && document.querySelector('#assetImage').complete"))
            select('labelme');wait(lambda:window.external_mode=='labelme')
            assert window.stack.currentWidget() is window.labelme_editor
            assert window.labelme_editor._canvas_widgets.canvas.shapes[0].points[0,0]==20
            window.grab().save(str(output/'01-labelme-embedded.png'))
            shape=window.labelme_editor._canvas_widgets.canvas.shapes[0];shape.points[0,0]=24;window.labelme_editor.mark_dirty()
            window.transition_editor('review');wait(lambda:window.external_mode is None and window.editor_transition is None)
            wait(lambda:js("!document.querySelector('#review').hidden"))
            asset=service.store.get_asset(pid,aid);assert asset['shapes'][0]['x']==24 and asset['review_state']=='pending'
            print('PASS Labelme -> review',flush=True)
            annotate();select('cvat')
            wait(lambda:window.external_mode=='cvat',120)
            cp=window.cvat_view.page()
            wait(lambda:js("window.workbenchCvatHost?.state==='ready' && !!document.querySelector('.cvat-annotation-header-save-button')",cp),90)
            bridge=CvatProjectBridge(service.data_root)
            assert bridge.read_annotations(service.store.snapshot(pid),cookies)==[]
            window.grab().save(str(output/'02-cvat-embedded.png'))
            print('PASS Labelme -> CVAT annotations',flush=True)
            # Use the same public CVAT plugin/core editing API the canvas saves to.
            js("""window.__qa='working';window.cvatUI.registerComponent(({store})=>{
                (async()=>{try {const a=store.getState().annotation,j=a.job.instance;
                    const objects=await j.annotations.get(a.player.frame.number,false,[]);
                    objects[0].points=[28,20,70,70];await objects[0].save();window.__qa='changed';
                }catch(e){window.__qa=String(e)}})();return {name:'workbench-qa-edit',destructor(){}};});""",cp)
            wait(lambda:js("window.__qa==='changed'",cp))
            window.transition_editor('labelme');wait(lambda:window.external_mode=='labelme' and window.editor_transition is None)
            assert service.store.get_asset(pid,aid)['shapes'][0]['x']==28
            assert window.labelme_editor._canvas_widgets.canvas.shapes[0].points[0,0]==28
            print('PASS CVAT save -> shared store -> Labelme',flush=True)
            window.transition_editor('builtin');wait(lambda:window.external_mode is None)
            wait(lambda:js("window.workbenchState().assetId!==null"))
            # Use the built-in editor's shared API, then refresh its record.
            project=service.store.get_project(pid)
            service.store.manage_classes(pid,project['classes']+['new-part'],{},project['revision'])
            a=service.store.get_asset(pid,aid);a['shapes'][0]['x']=30;a['shapes'][0]['label']='new-part'
            service.store.save_asset(pid,aid,a['shapes'],a['revision'])
            js("window.__qaReload=false;window.workbenchExternalSync({source:'labelme'}).then(()=>window.__qaReload=true)")
            wait(lambda:js('window.__qaReload===true'))
            select('cvat');wait(lambda:window.external_mode=='cvat',120)
            wait(lambda:js("window.workbenchCvatHost?.state==='ready' && !!document.querySelector('.cvat-annotation-header-save-button')",cp),90)
            assert bridge.read_annotations(service.store.snapshot(pid),cookies)==[]
            window.transition_editor('review');wait(lambda:window.external_mode is None and window.editor_transition is None)
            wait(lambda:js("!document.querySelector('#review').hidden"))
            assert service.store.get_asset(pid,aid)['shapes'][0]['x']==30
            window.grab().save(str(output/'03-unified-review.png'))
            print('PASS built-in -> CVAT -> review',flush=True)
            (output/'report.json').write_text(json.dumps({'success':True,'workflow':['Labelme -> review','Labelme -> CVAT','CVAT -> Labelme','built-in -> CVAT -> review'],'warnings':failures},indent=2),'utf-8')
        except Exception:
            window.grab().save(str(output/'failure.png'))
            (output/'failure-state.txt').write_text(str(js('document.body.innerText')),'utf-8')
            if window.cvat_view:
                (output/'failure-cvat.txt').write_text(str(js("JSON.stringify({host:window.workbenchCvatHost,text:document.body.innerText})",window.cvat_view.page())),'utf-8')
            raise
        finally:
            QMessageBox.warning=original_warning
            # Remove only this test's newly created remote project and task.
            linked=CvatProjectBridge(service.data_root)._load().get(pid)
            if linked:CvatProjectBridge(service.data_root)._request('DELETE',f'/api/projects/{linked["project_id"]}',cookies,expected=(204,))
            window.allow_close=True;window.close();window.page.deleteLater()
            if window.cvat_view:window.cvat_view.page().deleteLater()
            QTest.qWait(100);service.close()
    print('DESKTOP_EDITORS_OK',flush=True)

if __name__=='__main__':main()
