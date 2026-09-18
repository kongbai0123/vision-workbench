"""Publish validated external weights as project-owned model versions."""
from hashlib import sha256
from pathlib import Path
import shutil
import time
import uuid

from .training_engine import atomic_json, read_json


def import_model(workspace, project_id, path, name='', trusted=False, progress=lambda *_args: None, source_filename=None):
    if trusted is not True:
        raise ValueError('請確認權重來自可信任的來源；.pt 載入可能執行其中的 Python 程式碼')
    workspace.store.get_project(project_id, include_assets=False)
    if not isinstance(path, (str, Path)) or not str(path).strip():
        raise ValueError('請選擇 .pt 權重檔案')
    source = Path(path).expanduser().resolve()
    if not source.is_file() or source.suffix.lower() != '.pt' or source.stat().st_size == 0:
        raise ValueError('請選擇非空白的本機 .pt 權重檔案')
    if not isinstance(name, str) or len(name.strip()) > 100:
        raise ValueError('模型名稱上限為 100 字元')
    original_name = Path(str(source_filename or source.name).replace('\\', '/')).name
    if not original_name or Path(original_name).suffix.lower() != '.pt':
        raise ValueError('模型來源檔名無效')
    # Always refresh here: an installation can finish after the catalog cache was built.
    component = workspace.registry.component_status(refresh=True).get('ultralytics', {})
    if component.get('state') != 'ready':
        raise ValueError('請先至「設定 → 模型與元件」安裝或修復 Ultralytics 執行環境')
    parent = workspace.models_dir(project_id, create=True)
    temporary = parent / f'.import-{uuid.uuid4().hex}'
    temporary.mkdir()
    try:
        checkpoint = temporary / 'checkpoint.pt'
        progress('複製權重至專案模型區', 10)
        shutil.copyfile(source, checkpoint)
        with checkpoint.open('rb') as handle:
            digest_state = sha256()
            for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                digest_state.update(chunk)
            digest = digest_state.hexdigest()
        output = temporary / 'inspection.json'
        from .process_control import run_controlled
        progress('驗證模型架構、類別與推論相容性', 25)
        run_controlled([str(workspace.registry.component_python('ultralytics')), '-m', 'workbench.model_import_worker',
                        '--checkpoint', str(checkpoint), '--output', str(output)], progress,
                       cwd=Path(__file__).resolve().parents[1], timeout=300)
        info = read_json(output)
        definition = workspace.registry.model(info.get('engine'))
        if not definition or not definition.get('inference_only') or info.get('task') != definition['task']:
            raise ValueError('權重未通過模型相容性驗證')
        classes = info.get('classes')
        if not isinstance(classes, list) or not classes or not all(isinstance(value, str) and value.strip() for value in classes) or len(set(classes)) != len(classes):
            raise ValueError('模型類別資訊無效')
        from .training import _next_id
        with workspace.lock:
            model_id = _next_id(parent, 'M')
            record = {'schema_version': 1, 'model_version_id': model_id, 'run_id': None, 'dataset_version_id': None,
                      'engine': info['engine'], 'engine_name': name.strip() or f"{definition['name']} · {Path(original_name).stem}",
                      'task': info['task'], 'classes': classes, 'checkpoint': 'checkpoint.pt',
                      'image_size': 640, 'score_threshold': .5, 'created_at': time.time(),
                      'source': {'kind': 'external_import', 'filename': original_name, 'sha256': digest,
                                 'bytes': checkpoint.stat().st_size, 'architecture': info.get('architecture'),
                                 'trusted_by_user': True, 'validation': 'cpu_inference_smoke_test'},
                      'validation': None, 'test': None}
            output.unlink()
            atomic_json(temporary / 'model.json', record)
            target = parent / model_id
            temporary.replace(target)
            try:
                workspace._sync_catalog(project_id)
            except BaseException:
                # Only this newly published version is rolled back.
                target.replace(temporary)
                raise
        progress('模型匯入完成，可進行試跑與預標註', 100)
        return record
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
