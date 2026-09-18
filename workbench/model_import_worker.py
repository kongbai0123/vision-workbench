"""Inspect explicitly trusted Ultralytics weights in the optional model runtime."""
import argparse
import os
from pathlib import Path

from .training_engine import atomic_json


def inspect_checkpoint(path):
    os.environ.setdefault('YOLO_OFFLINE', 'true')
    os.environ['YOLO_AUTOINSTALL'] = 'false'
    import numpy as np
    from ultralytics import YOLO, RTDETR
    from ultralytics.nn.tasks import RTDETRDetectionModel
    model = YOLO(str(path))
    rtdetr = isinstance(model.model, RTDETRDetectionModel)
    if rtdetr:
        model = RTDETR(str(path))
    task = model.task
    if task not in {'detect', 'segment'} or (rtdetr and task != 'detect'):
        raise ValueError('目前僅支援 YOLO 偵測／實例分割與 RT-DETR 偵測；不支援分類、姿態或 OBB 權重')
    names = model.names
    if not isinstance(names, dict) or sorted(names) != list(range(len(names))):
        raise ValueError('模型類別索引必須從 0 連續排列')
    classes = [names[index] for index in range(len(names))]
    if not classes or not all(isinstance(name, str) and name.strip() for name in classes) or len(set(classes)) != len(classes):
        raise ValueError('模型類別名稱不得為空或重複')
    print('已讀取架構與類別，正在驗證 CPU 推論', flush=True)
    model.predict(source=np.zeros((640, 640, 3), dtype=np.uint8), imgsz=640, device='cpu', verbose=False)
    return {'engine': 'rt_detr_external' if rtdetr else f'external_yolo_{task}',
            'task': 'instance_segmentation' if task == 'segment' else 'object_detection',
            'classes': classes, 'architecture': type(model.model).__name__}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    path = Path(args.checkpoint)
    if not path.is_file() or path.suffix.lower() != '.pt':
        raise ValueError('請選擇本機 .pt 權重')
    atomic_json(Path(args.output), inspect_checkpoint(path))


if __name__ == '__main__':
    main()
