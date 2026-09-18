"""The dispatch contract shared by launch, training workers and inference."""
from dataclasses import dataclass
from importlib import import_module


@dataclass(frozen=True)
class EngineSpec:
    definition: dict
    train_module: str
    predict_module: str | None

    def parameter_schema(self):
        from .training_parameters import parameter_schema
        return parameter_schema(self.definition)

    def validate(self, config):
        from .training_parameters import validate_config
        return validate_config(self.definition, config)

    def train(self, dataset, run_dir, model_dir):
        return import_module(self.train_module).train(dataset, run_dir, model_dir)

    def validate_dataset(self, manifest, config):
        from .split_quality import loose_split_applies
        if self.definition['task'] == 'image_classification':
            if len(manifest['classes']) < 2:
                raise ValueError('影像分類至少需要兩個類別')
            ambiguous = [asset for asset in manifest['assets']
                         if len({shape.get('label') for shape in asset.get('shapes', [])
                                 if shape.get('label') in manifest['classes']}) != 1]
            if ambiguous:
                raise ValueError(f'影像分類要求每張圖片只有一個標註類別；請修正 {len(ambiguous)} 張圖片')
            train_labels = {shape.get('label') for asset in manifest['assets'] if asset['split'] == 'train'
                            for shape in asset.get('shapes', [])}
            missing = [name for name in manifest['classes'] if name not in train_labels]
            if missing and not loose_split_applies(manifest.get('split_plan'), manifest['assets']):
                raise ValueError(f"Train 缺少分類樣本：{'、'.join(missing)}")
        if self.definition['component'] == 'ultralytics' and self.definition['task'] == 'instance_segmentation':
            from .yolo_compatibility import analyze_manifest, blocker_message
            report = analyze_manifest(manifest, config)
            if not report['compatible']:
                raise ValueError(blocker_message(report))
            return report
        return None


FAMILY_ADAPTERS = {
    'YOLO External': ('workbench.ultralytics_engine', 'workbench.ultralytics_predict_worker'),
    '內建基準': ('workbench.training_engine', None),
    'Mask R-CNN': ('workbench.maskrcnn_engine', 'workbench.maskrcnn_predict_worker'),
    'Faster R-CNN': ('workbench.torchvision_engines', 'workbench.torchvision_predict_worker'),
    'DeepLabV3': ('workbench.torchvision_engines', 'workbench.torchvision_predict_worker'),
    'TorchVision Classification': ('workbench.classification_engine', None),
    'RT-DETR': ('workbench.ultralytics_engine', 'workbench.ultralytics_predict_worker'),
    'YOLO26 Seg': ('workbench.ultralytics_engine', 'workbench.ultralytics_predict_worker'),
    'YOLO26 Detect': ('workbench.ultralytics_engine', 'workbench.ultralytics_predict_worker'),
}


def engine_spec(key):
    from .model_registry import MODELS
    definition = next((item for item in MODELS if item['key'] == key), None)
    if definition is None or definition['family'] not in FAMILY_ADAPTERS:
        raise ValueError(f'未知或未實作的訓練引擎：{key}')
    return EngineSpec(definition, *FAMILY_ADAPTERS[definition['family']])
