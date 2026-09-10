"""Explicit short hardware acceptance. Images live only in a temporary folder."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import json
import tempfile
import time
from workbench.acquisition import CameraService


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--camera-index',type=int,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    report={'success':False,'camera_index':args.camera_index}
    with tempfile.TemporaryDirectory(prefix='vision-hardware-') as directory:
        camera=CameraService(Path(directory))
        try:
            report['devices']=camera.devices()
            camera.start(args.camera_index,width=640,height=480,fps=15)
            deadline=time.monotonic()+15
            while time.monotonic()<deadline:
                state=camera.status()
                if state.get('has_frame'):break
                if state.get('error'):raise RuntimeError(state['error'])
                time.sleep(.1)
            else:raise RuntimeError('Camera did not deliver a frame within 15 seconds')
            snapshot=camera.snapshot()
            report['snapshot_bytes']=Path(snapshot['path']).stat().st_size
            camera.set_processing('canny')
            deadline=time.monotonic()+5
            while time.monotonic()<deadline and not camera.status().get('has_processed_frame'):time.sleep(.05)
            report['processed_jpeg_bytes']=len(camera.frame_jpeg(processed=True))
            camera.start_recording('hardware-acceptance')
            time.sleep(2)
            stopped=camera.stop_recording()
            recording=stopped.get('last_recording')
            if not recording:raise RuntimeError('No finalized recording returned')
            report['recording']={k:v for k,v in recording.items() if k not in {'path','name','batch_id'}}
            report['status']={k:camera.status().get(k) for k in ('width','height','fps','measured_fps','processing_ms')}
            final=camera.stop()
            if final.get('state') not in ('stopped','idle') or final.get('recording'):
                raise RuntimeError('Camera failed to stop cleanly')
            report['final_state']=final.get('state')
            report['success']=True
        except Exception as exc:
            report['error']=str(exc)
        finally:
            camera.close()
    target=root/'verification'/'hardware-camera-20260909.json'
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=True))
    return 0 if report['success'] else 1


if __name__=='__main__':raise SystemExit(main())
