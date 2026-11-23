"""Install a verified, local batch-annotation bundle beside the application."""

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import stat
from tempfile import TemporaryDirectory
from uuid import uuid4

from .annotation_assist_gemma import MODEL_FILES, RUNTIME_FILES
from .annotation_assist_grounding import FILES as DINO_FILES, HEALTH_EXAMPLES
from .annotation_assist_runtime import model_home
from .annotation_assist_sam import SAM_FILES
from .annotation_assist_worker import QWEN_PROVENANCE_SHA256
from .file_lock import exclusive_file_lock


INSTALL_PATH = 'models/annotation-assist'
INTERACTIVE_SAM_PATH = 'models/sam2.1-hiera-tiny'
QWEN_MANIFEST = 'weights-provenance.json'
COPY_CHUNK = 8 * 1024 * 1024


@dataclass(frozen=True)
class Component:
    id: str
    name: str
    directory: str
    files: dict
    required: bool = False


RUNTIME = Component('runtime', '批量標註 CUDA Python', 'runtime', {
    'python.exe': None,
    'python313.dll': None,
    'Lib/encodings/__init__.py': None,
    'Lib/site-packages/PIL/__init__.py': None,
    'Lib/site-packages/numpy/__init__.py': None,
    'Lib/site-packages/torch/__init__.py': None,
    'Lib/site-packages/torch/lib/torch_cuda.dll': None,
    'Lib/site-packages/torchvision/__init__.py': None,
    'Lib/site-packages/transformers/__init__.py': None,
    'Lib/site-packages/safetensors/__init__.py': None,
}, required=True)
DINO = Component('mm-tiny', 'MM Grounding DINO Tiny', 'MM-Grounding-DINO-Tiny', {
    **DINO_FILES, **{f"health/{row['file']}": row['sha256'] for row in HEALTH_EXAMPLES},
}, required=True)
QWEN = Component('qwen-4b', 'Qwen3-VL 4B', 'Qwen3-VL-4B-Instruct', {}, required=True)
BATCH_SAM = Component('batch-sam', 'SAM 2.1 批次框界檢查', 'sam2.1-hiera-tiny', SAM_FILES)
GEMMA = Component('gemma-4', 'Gemma 4 觀察覆核', 'Gemma-4-E4B-it-QAT-Q4', MODEL_FILES)
LLAMA = Component('gemma-runtime', 'Gemma 4 CUDA 執行庫', '_runtimes/llama.cpp-b12541-cuda13.5', RUNTIME_FILES)
BATCH_COMPONENTS = (RUNTIME, DINO, QWEN, BATCH_SAM, GEMMA, LLAMA)
INTERACTIVE_SAM = Component('interactive-sam', 'SAM 2.1 互動分割權重', INTERACTIVE_SAM_PATH, SAM_FILES)


def _entry(path):
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
        raise ValueError(f'模型需使用實體本機檔案：{path.name}')
    return info


def _directory(path):
    if not stat.S_ISDIR(_entry(path).st_mode):
        raise ValueError(f'模型目錄無效：{path.name}')


def _file(path, size=None, *, allow_empty=False):
    info = _entry(path)
    if not stat.S_ISREG(info.st_mode) or (not allow_empty and info.st_size == 0) or (size is not None and info.st_size != size):
        raise ValueError(f'模型檔案或大小不符：{path.name}')


def _files(component, directory):
    if component.id != QWEN.id:
        return {name: pin if isinstance(pin, tuple) else (None, pin) for name, pin in component.files.items()}
    manifest = directory / QWEN_MANIFEST
    _file(manifest)
    # The worker pins the manifest itself, not only the weights it lists.
    raw = manifest.read_bytes()
    if sha256(raw).hexdigest() != QWEN_PROVENANCE_SHA256:
        raise ValueError('Qwen 權重來源清冊與核准版本不符。')
    files = json.loads(raw)['files']
    return {
        QWEN_MANIFEST: (len(raw), QWEN_PROVENANCE_SHA256),
        **{name: (entry['bytes'], entry['sha256']) for name, entry in files.items()},
    }


def _checked_directory(home, component):
    directory = home
    for part in Path(component.directory).parts:
        directory /= part
        _directory(directory)
    return directory


def _inventory(home, component):
    directory = _checked_directory(home, component)
    files = _files(component, directory)
    for name, (size, _) in files.items():
        relative = Path(name)
        if relative.anchor or not relative.parts or '..' in relative.parts:
            raise ValueError('模型檔案清冊含有無效的相對路徑。')
        for parent in reversed(relative.parents):
            _directory(directory / parent)
        _file(directory / relative, size)
    return files


def _status(home, component):
    try:
        _checked_directory(home, component)
    except FileNotFoundError:
        return 'missing', '尚未安裝'
    except (OSError, ValueError) as exc:
        return 'incomplete', str(exc)
    try:
        _inventory(home, component)
    except (OSError, ValueError) as exc:
        missing = Path(exc.filename).name if isinstance(exc, FileNotFoundError) and exc.filename else None
        return 'incomplete', f'缺少模型檔案：{missing}' if missing else str(exc)
    message = '已找到必要檔案'
    if component.id == INTERACTIVE_SAM.id:
        message = '已找到權重。互動分割另外需要桌面 SAM 2.1 執行環境。'
    return 'available', message


def annotation_model_catalog(app_root):
    app_root = Path(app_root).resolve()
    try:
        home = model_home(app_root)
    except (OSError, RuntimeError, ValueError):
        home = None
    custom_location = home != app_root / INSTALL_PATH
    rows = []
    for component in (*BATCH_COMPONENTS, INTERACTIVE_SAM):
        interactive = component.id == INTERACTIVE_SAM.id
        if interactive or home is not None:
            state, message = _status(app_root if interactive else home, component)
        else:
            state, message = 'incomplete', '無法讀取設定的批量標註模型位置。'
        if interactive and state == 'missing':
            message = '互動分割權重由 bootstrap.ps1 -AI 準備。'
        elif not interactive and state != 'incomplete':
            message = ''
        rows.append({'id': component.id, 'name': component.name, 'state': state, 'message': message,
                     'install_path': INTERACTIVE_SAM_PATH if interactive else f'{INSTALL_PATH}/{component.directory}'})
    return {'install_path': INSTALL_PATH, 'custom_location': custom_location,
            'installable': not custom_location, 'components': rows}


def _source_home(app_root, source):
    if not isinstance(source, (str, Path)) or isinstance(source, str) and not source.strip():
        raise ValueError('請選擇工作台資料夾或標註模型資料夾。')
    root = (app_root / Path(source)).resolve(strict=True)
    _directory(root)
    nested = root / INSTALL_PATH
    if nested.exists():
        _directory(nested.parent)
        _directory(nested)
        return nested
    if not any((root / item.directory).exists() for item in BATCH_COMPONENTS):
        raise ValueError('所選資料夾未找到標註模型套件。')
    return root


def _plan(home, source_home):
    plan = []
    for component in BATCH_COMPONENTS:
        if _status(home, component)[0] == 'available':
            continue
        state, message = _status(source_home, component)
        if state == 'available':
            plan.append(component)
        elif component.required or state == 'incomplete':
            raise ValueError(f'來源的 {component.name} 無法安裝。{message}')
    return plan


def _copy_file(source, destination, digest, update):
    _file(source, allow_empty=digest is None)
    checksum = sha256() if digest else None
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open('rb') as reader, destination.open('xb') as writer:
        while block := reader.read(COPY_CHUNK):
            writer.write(block)
            if checksum:
                checksum.update(block)
            update()
    if checksum and checksum.hexdigest() != digest:
        raise ValueError(f'模型檔案雜湊不符：{source.name}')


def _copy_runtime(source, destination, update, *, skip=()):
    for path in source.iterdir():
        if path.name in skip or path.name == '__pycache__' or path.suffix in {'.pyc', '.pyo'}:
            continue
        entry = _entry(path)
        target = destination / path.name
        if stat.S_ISDIR(entry.st_mode):
            target.mkdir()
            _copy_runtime(path, target, update)
        else:
            _copy_file(path, target, None, update)


def _copy_component(component, source_home, staging, update):
    files = _inventory(source_home, component)
    source = source_home / component.directory
    destination = staging / component.directory
    destination.mkdir(parents=True)
    if component.id == RUNTIME.id:
        # Windows console launchers can embed the interpreter's original path.
        _copy_runtime(source, destination, update, skip={'Scripts'})
    else:
        for name, (_, digest) in files.items():
            _copy_file(source / name, destination / name, digest, update)
    _inventory(staging, component)


def _publish(home, staging, component, backup_root):
    target = home / component.directory
    if target.parent.resolve() != target.parent or not target.is_relative_to(home):
        raise ValueError('模型安裝路徑不是工作台內的實體目錄。')
    target.parent.mkdir(parents=True, exist_ok=True)
    staged = staging / component.directory
    if staged.resolve() != staged or not staged.is_relative_to(staging):
        raise ValueError('模型安裝暫存路徑無效。')
    backup = None
    try:
        _entry(target)
    except FileNotFoundError:
        pass
    else:
        backup = backup_root / component.directory
        if backup.parent.resolve() != backup.parent or not backup.is_relative_to(home):
            raise ValueError('原模型備份路徑無效。')
        backup.parent.mkdir(parents=True, exist_ok=True)
        target.rename(backup)
    try:
        staged.rename(target)
    except OSError:
        if backup is not None:
            backup.rename(target)
        raise
    return backup


def install_annotation_models(app_root, source, progress):
    app_root = Path(app_root).resolve()
    home = app_root / INSTALL_PATH
    if not annotation_model_catalog(app_root)['installable']:
        raise ValueError('目前使用自訂模型位置。請以預設位置安裝，或手動管理自訂目錄。')
    try:
        source_home = _source_home(app_root, source)
    except FileNotFoundError as exc:
        raise ValueError('找不到所選的模型來源。') from exc
    installed, backups = [], []
    with exclusive_file_lock(home.parent / '.annotation-model-install.lock'):
        progress('檢查標註模型來源', 5)
        plan = _plan(home, source_home)
        if plan:
            if home.resolve() != home:
                raise ValueError('模型安裝路徑不是工作台內的實體目錄。')
            home.mkdir(parents=True, exist_ok=True)
            backup_root = home / f'.before-import-{uuid4().hex}'
            with TemporaryDirectory(prefix='.annotation-install-', dir=home) as directory:
                staging = Path(directory).resolve()
                for index, component in enumerate(plan):
                    percent = 10 + 75 * index / len(plan)
                    message = f'複製並核對 {component.name}'

                    def update():
                        progress(message, percent)

                    update()
                    _copy_component(component, source_home, staging, update)
                progress('保存已核對的標註模型', 95)
                for component in plan:
                    backup = _publish(home, staging, component, backup_root)
                    if backup is not None:
                        backups.append(backup.relative_to(app_root).as_posix())
                    installed.append(component.id)
    return {**annotation_model_catalog(app_root), 'installed': installed, 'backups': backups}
