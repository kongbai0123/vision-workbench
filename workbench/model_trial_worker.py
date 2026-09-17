"""Run one trained model over external images or every frame of one video."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import cv2
import numpy as np
from PIL import Image
from .training_engine import atomic_json, read_json

IMAGE_SUFFIXES={'.png','.jpg','.jpeg','.webp','.bmp','.tif','.tiff'}
VIDEO_SUFFIXES={'.mp4','.avi','.mov','.mkv','.webm'}

def predictor(request):
    model_path=Path(request['model_path']);record=read_json(model_path);record['score_threshold']=0.05;kind=request['predict_module']
    if kind=='workbench.ultralytics_predict_worker':
        from .ultralytics_engine import Predictor; dtype=np.uint8
    elif kind=='workbench.torchvision_predict_worker':
        from .torchvision_engines import Predictor; dtype=np.float32
    elif kind=='workbench.maskrcnn_predict_worker':
        from .maskrcnn_engine import Predictor; dtype=np.float32
    else: raise ValueError('此模型尚未提供圖片／影片試跑介面')
    return Predictor(record,model_path.parent,request.get('device','auto')),dtype

def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--request',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args(argv);request=read_json(Path(args.request));output=Path(request['output_dir']);output.mkdir(parents=True,exist_ok=True)
    model,dtype=predictor(request);frames=[]
    def infer(rgb,name,source_index,frame_index=None,time_seconds=None):
        height,width=rgb.shape[:2];image_name=f'{len(frames):08d}.jpg';Image.fromarray(rgb).save(output/image_name,'JPEG',quality=90)
        shapes=model.predict(rgb.astype(dtype,copy=False),width,height)
        frames.append({'index':len(frames),'name':name,'image':image_name,'width':width,'height':height,'shapes':shapes,
                       'source_index':source_index,'frame_index':frame_index,'time_seconds':time_seconds})
    for source_index,raw in enumerate(request['paths']):
        path=Path(raw);suffix=path.suffix.lower()
        if suffix in IMAGE_SUFFIXES:
            with Image.open(path) as image: infer(np.asarray(image.convert('RGB')),path.name,source_index)
            print('圖片模型推論中',flush=True)
        elif suffix in VIDEO_SUFFIXES:
            capture=cv2.VideoCapture(str(path));fps=float(capture.get(cv2.CAP_PROP_FPS) or 0);index=0
            if not capture.isOpened() or fps<=0: raise ValueError(f'無法解碼影片：{path.name}')
            try:
                while True:
                    ok,bgr=capture.read()
                    if not ok: break
                    infer(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),path.name,source_index,index,index/fps);index+=1
                    print('影片逐幀推論中',flush=True)
            finally: capture.release()
            if not index: raise ValueError(f'影片沒有可解碼影格：{path.name}')
        else: raise ValueError(f'不支援的試跑格式：{path.name}')
    atomic_json(Path(args.output),{'frames':frames})
    return 0
if __name__=='__main__': raise SystemExit(main())
