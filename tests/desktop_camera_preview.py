"""Real camera worker + HTTP + Qt UI, with a synthetic delayed video device."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
os.environ.setdefault('QTWEBENGINE_CHROMIUM_FLAGS','--disable-gpu --disable-background-timer-throttling --disable-renderer-backgrounding')
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import tempfile
import threading
import time
import numpy as np
import cv2
from PySide6.QtWidgets import QApplication
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtCore import QEventLoop,QTimer,QUrl
from PySide6.QtTest import QTest
from workbench.acquisition import AcquisitionError,AcquisitionCancelled
from workbench.server import WorkbenchService
from workbench.camera_controls import CameraControls
from test_camera_controls import Driver


class SyntheticControls(CameraControls):
    def __init__(self):
        self.interfaces = {'camera': Driver()}
        self.error = None
        self.threads = []

    def read(self):
        self.threads.append(threading.current_thread().name)
        return super().read()

    def close(self):
        self.threads.append(threading.current_thread().name)


class SyntheticDevice:
    def __init__(self):self.frames=0;self.disconnect=False;self.config={'width':640,'height':480,'fps':25}
    def get(self,prop):return cv2.VideoWriter_fourcc(*'MJPG') if prop==cv2.CAP_PROP_FOURCC else self.config['fps']
    def read(self):
        time.sleep(1/self.config['fps']);self.frames+=1
        if self.disconnect:return False,None
        frame=np.full((self.config['height'],self.config['width'],3),(50,80,30),np.uint8)
        frame[90:390,150:490]=(100,180,230)
        frame[200:280,280:360]=(50,80,30)
        frame[20:40,20:20+self.frames%580]=(220,240,240)
        return True,frame
    def release(self):pass


def main():
    app=QApplication([])
    output=Path(__file__).resolve().parents[1]/'qa-output/camera-preview'
    output.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='vision-camera-ui-') as folder:
        service=WorkbenchService(Path(folder)).start()
        service.store.create_project('相機參數設定 · 合成影像示範')
        camera=service.camera;gate=threading.Event();control={'fail':False,'frame_error':False,'requests':0}
        device=SyntheticDevice()
        controls=SyntheticControls()
        camera._open_controls=lambda index:controls
        def open_device(config):
            device.config=dict(config)
            while not gate.wait(.02):
                if camera._stop_event.is_set():raise AcquisitionCancelled('cancelled')
            if control['fail']:raise AcquisitionError('模擬裝置啟動失敗')
            camera._state['backend']='DSHOW'
            return device
        camera._open_capture=open_device
        camera.devices=lambda:[{'index':0,'name':'Synthetic delayed camera'}]
        camera.capabilities=lambda index:{'index':index,'modes':[
            {'width':1280,'height':720,'min_fps':15,'max_fps':30,'fps_options':[15,30],'pixel_format':'MJPG'},
            {'width':640,'height':480,'min_fps':5,'max_fps':5,'fps_options':[5],'pixel_format':'YUY2'},
            {'width':640,'height':480,'min_fps':15,'max_fps':15,'fps_options':[15],'pixel_format':'MJPG'}]}
        original_frame=camera.frame_jpeg
        def get_frame(processed=False):
            control['requests']+=1
            if control['frame_error']:raise AcquisitionError('模擬暫時無影格')
            return original_frame(processed)
        camera.frame_jpeg=get_frame
        view=QWebEngineView();view.resize(1440,900);view.show();view.setUrl(QUrl(service.entry_url))
        def js(script):
            result=[];loop=QEventLoop()
            view.page().runJavaScript(script,lambda value:(result.append(value),loop.quit()))
            QTimer.singleShot(8000,loop.quit);loop.exec()
            assert result,'JS timeout'
            return result[0]
        def wait(expression):
            deadline=time.monotonic()+15
            while time.monotonic()<deadline:
                if js(expression):return
                QTest.qWait(80)
            raise AssertionError(expression+'\n'+str(js('document.body.innerText')))
        def click(selector):
            wait(f"!document.querySelector({json.dumps(selector)}).disabled && !window.workbenchState().busy && !window.workbenchState().transitioning")
            js(f"document.querySelector({json.dumps(selector)}).click()")
        def running():
            wait("!document.querySelector('#cameraFrame').hidden && document.querySelector('#cameraFrame').naturalWidth===640")
            assert js("document.querySelector('#cameraEmpty').hidden")
            wait("!document.querySelector('#takeSnapshot').disabled")
        try:
            wait("typeof window.workbenchState==='function' && document.querySelector('.project-card')!==null")
            click('.project-card');click('#findCameras')
            wait("document.querySelector('#cameraResolution').value==='1280x720' && !window.workbenchState().busy")
            js("document.querySelector('#cameraResolution').value='640x480';document.querySelector('#cameraResolution').dispatchEvent(new Event('change'))")
            assert js("document.querySelector('#cameraFPS').value")=='15'
            js("document.querySelector('#cameraFPS').value='60'")
            click('#startCamera')
            wait("document.querySelector('#toast').innerText.includes('不支援所填 FPS')")
            assert camera.status()['state']=='stopped'
            js("document.querySelector('#cameraFPS').value='15';document.querySelector('#toast').hidden=true")
            click('#startCamera')
            wait("document.querySelector('#cameraStatus').innerText.includes('等待第一張') && !window.workbenchState().busy")
            assert camera.status()['state']=='starting'
            assert js("document.querySelector('#startCamera').disabled && !document.querySelector('#stopCamera').disabled")
            QTest.qWait(800);assert control['requests']==0
            gate.set();running()
            assert device.config['width']==640 and device.config['height']==480 and device.config['fps']==15 and device.config['pixel_format']=='MJPG'
            assert js("!document.querySelector('#cameraResolution').disabled && !document.querySelector('#cameraFPS').disabled")
            assert '15 FPS' in js("document.querySelector('#cameraReadout').textContent")
            click('#toggleSourceInspector')
            wait("document.querySelector('#cameraParam-exposure')!==null")
            assert js("document.querySelector('#cameraParam-exposure').disabled")
            js("document.querySelector('[data-parameter=exposure] input[type=checkbox]').click()")
            wait("!document.querySelector('#cameraParam-exposure').disabled")
            js("let input=document.querySelector('#cameraParam-exposure');input.value=-8;input.dispatchEvent(new Event('change'))")
            wait("document.querySelector('#cameraParam-exposure')?.value==='-8' && !document.querySelector('#cameraParam-exposure').disabled")
            assert controls.interfaces['camera'].value==-8
            js("document.querySelector('#cameraProfileName').value='固定光源'")
            click('#saveCameraProfile')
            wait("document.querySelector('#cameraProfileStatus').textContent.includes('已儲存')")
            js("document.querySelector('#cameraPixelFormat').value='YUY2';document.querySelector('#cameraPixelFormat').dispatchEvent(new Event('change'))")
            assert js("document.querySelector('#cameraFPS').value")=='5'
            js("document.querySelector('#cameraPixelFormat').value='MJPG';document.querySelector('#cameraPixelFormat').dispatchEvent(new Event('change'))")
            click('#applyCameraSettings');running()
            wait("document.querySelector('#cameraParam-exposure')?.value==='-8' && !document.querySelector('#cameraParam-exposure').disabled")
            view.resize(1440,1120)
            wait("!document.querySelector('#cameraReadout').textContent.includes('預覽 0 FPS')")
            js("document.querySelector('#toast').textContent=''")
            QTest.qWait(200)
            view.grab().save(str(output/'settings.png'))
            assert js("(()=>{const a=document.querySelector('.camera-actions-footer').getBoundingClientRect(),p=document.querySelector('#cameraSettingsInspector').getBoundingClientRect();return a.bottom<=p.bottom+1&&a.top>=p.top})()")
            view.resize(1088,650);QTest.qWait(300)
            assert js("(()=>{const a=document.querySelector('.camera-actions-footer').getBoundingClientRect(),p=document.querySelector('#cameraSettingsInspector').getBoundingClientRect();return a.bottom<=p.bottom+1&&a.top>=p.top})()")
            view.resize(1440,900);QTest.qWait(200)
            click('#stopCamera');wait("!document.querySelector('#startCamera').disabled")
            js("document.querySelector('#cameraFPS').value='5'")
            click('#loadCameraProfile')
            wait("document.querySelector('#cameraFPS').value==='15'")
            click('#startCamera');running()
            assert set(controls.threads)=={'WorkbenchCamera'}
            click('#closeSourceInspector')
            first=js("document.querySelector('#cameraFrame').src")
            wait(f"document.querySelector('#cameraFrame').src!=={json.dumps(first)}")
            js("document.querySelector('#processingMode').value='classical';document.querySelector('#processingMode').dispatchEvent(new Event('change'))")
            click('#applyProcessing')
            wait("!window.workbenchState().busy && document.querySelector('#processingStatus').innerText.includes('顯示原圖')")
            running()
            js("document.querySelector('#processingMode').value='canny';document.querySelector('#processingMode').dispatchEvent(new Event('change'))")
            click('#applyProcessing')
            js("document.querySelector('#previewLayout').value='compare';document.querySelector('#previewLayout').dispatchEvent(new Event('change'))")
            wait("!document.querySelector('#cameraRawFrame').hidden && document.querySelector('#cameraRawFrame').naturalWidth===640 && !document.querySelector('#cameraFrame').hidden")
            assert js("document.querySelector('#cameraPreview').classList.contains('compare')")
            js("document.querySelector('#autoCaptureInterval').value='0.5';document.querySelector('#autoCaptureLimit').value='2'")
            click('#startAutoCapture')
            wait("!document.querySelector('#startAutoCapture').hidden && document.querySelector('#autoCaptureStatus').innerText.includes('執行 2 次')")
            wait("document.querySelectorAll('#acquiredAssets .acquired-item').length===2")
            click('#selectAllAssets')
            wait("document.querySelector('#deleteSelectedAssets').innerText.includes('2')")
            click('#deleteSelectedAssets')
            wait("document.querySelector('#confirmDialog').getClientRects().length>0")
            js("document.querySelector('#confirmDialog').click()")
            wait("document.querySelectorAll('#acquiredAssets .acquired-item').length===0 && !document.querySelector('#formDialog').open")
            QTest.qWait(350);view.grab().save(str(output/'running.png'))
            control['frame_error']=True
            wait("document.querySelector('#cameraStatus').innerText.includes('正在重試')")
            control['frame_error']=False;running()
            click("[data-source='files']");QTest.qWait(300)
            count=control['requests'];QTest.qWait(850);assert control['requests']==count
            device.disconnect=True
            wait("document.querySelector('#cameraStatus').innerText.includes('相機已中斷')")
            click("[data-source='camera']")
            assert not js("document.querySelector('#startCamera').disabled")
            device.disconnect=False;control['fail']=True;click('#startCamera')
            wait("document.querySelector('#cameraEmpty').innerText.includes('模擬裝置啟動失敗')")
            control['fail']=False;click('#startCamera');running()
            click('#stopCamera');wait("!document.querySelector('#startCamera').disabled")
            assert camera.status()['state']=='stopped'
            gate.clear();click('#startCamera')
            wait("document.querySelector('#stopCamera').innerText==='取消啟動' && !window.workbenchState().busy")
            click('#stopCamera');wait("!document.querySelector('#startCamera').disabled")
            assert camera.status()['state']=='stopped'
            assert not js("document.querySelector('#cameraFrame').hasAttribute('src')")
            result={'success':True,'synthetic_device':True,'frame_requests':control['requests'],
                'checks':['delayed startup automatically displays frames','prevent duplicate start','continuous preview',
                          'frame error retry','no hidden preview requests','disconnect without frame callback',
                          'async startup error','restart','stop','cancel startup','device modes',
                          'resolution-linked fps','reject unsupported fps','selected settings reach capture worker',
                          'raw fallback while processing needs calibration','raw/processed split comparison','custom interval auto capture','automatic capture limit',
                          'select visible assets','atomic batch deletion','manual exposure readback',
                          'pixel format linked fps','apply settings and restart','profile persistence and reload',
                          'persistent footer at 1440 and 1088 widths','native controls owned by capture worker']}
            (output/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
            print('CAMERA_PREVIEW_LIFECYCLE_OK',flush=True)
        except Exception:
            view.grab().save(str(output/'failure.png'));raise
        finally:
            gate.set();view.close();service.close()

if __name__=='__main__':main()
