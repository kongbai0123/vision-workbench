"""Effective setup counts and adapter support, available before any training."""
import math
from .augmentation import normalize_augmentation, augmentation_event_layout


def effective_setup(manifest, definition, config):
    profile = normalize_augmentation(config.get('augmentation'))
    counts = {split: sum(a['split'] == split for a in manifest['assets']) for split in ('train', 'val', 'test')}
    builtin = definition['component'] == 'builtin'
    supported = set() if builtin else {'brightness', 'contrast', 'fliplr', 'flipud'}
    if definition['component'] == 'ultralytics':
        supported = {'brightness', 'fliplr', 'flipud', 'degrees', 'translate', 'scale', 'mosaic', 'mixup', 'copy_paste', 'close_mosaic'}
    ignored = [key for key in ('brightness', 'contrast', 'fliplr', 'flipud', 'degrees', 'translate', 'scale', 'mosaic', 'mixup', 'copy_paste', 'close_mosaic') if profile[key] and key not in supported]
    layout = augmentation_event_layout(counts['train'], None if builtin else profile)
    notes = []
    if builtin:
        notes.append('內建像素基準只讀取 Train 原圖一次建立統計；不使用增強、擴充、Batch 或梯度累積。輪數是 Validation 門檻搜尋次數。')
    if definition.get('family') == 'RT-DETR':
        notes.append('RT-DETR 的最終變換組合由已安裝 Ultralytics adapter 決定；此檢查驗證傳入參數，不宣稱已執行所有變換。')
    if ignored:
        notes.append('此引擎不使用配方欄位：' + '、'.join(ignored))
    if definition['component'] == 'ultralytics':
        notes.append('有效批次為設定目標；實際批次受資料量影響，暖身期梯度累積可能調整，未驗證 GPU 記憶體或執行效能。')
    if definition['component'] == 'ultralytics' and profile['expansion_count']:
        notes.append('Ultralytics 會對原始與額外載入項目都套用線上增強；不保證原始載入項目保持未變換。')
    batch = config.get('batch_size') if not builtin else None
    return {'original_images': len(manifest['assets']), 'splits': counts, 'training_events': layout,
            'batches_per_epoch': math.ceil(layout['events'] / batch) if batch else None,
            'effective_batch_size': batch * config.get('gradient_accumulation', 1) if batch else None,
            'augmentation_applied': not builtin and profile['preset'] != 'off',
            'ignored_augmentation_fields': ignored, 'notes': notes}
