"""Run local annotation proposals in a verified Windows AppContainer."""

from hashlib import file_digest
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from tempfile import TemporaryDirectory
import time

from .resources import accelerator_lease


ROOT = Path(__file__).resolve().parents[1]
PROFILE = 'VisionWorkbenchAnnotationAssistV1'
MODEL_ENV = 'VISION_WORKBENCH_ANNOTATION_ASSIST_HOME'
PHASES = {
    'task_preparation': ('整理類別文字', 2, 3),
    'scene_scan': ('Qwen 掃描原圖', 5, 10),
    'dino': ('MM 尋找候選框', 16, 25),
    'qwen': ('Qwen 裁決候選框', 43, 28),
    'dino_retry': ('MM 重查局部', 73, 11),
    'qwen_retry': ('Qwen 覆核重查', 85, 11),
}


def model_home():
    return Path(os.environ.get(MODEL_ENV, ROOT / 'models' / 'annotation-assist')).resolve()


def available():
    home = model_home()
    return os.name == 'nt' and all((home / item).is_file() for item in (
        'runtime/python.exe',
        'MM-Grounding-DINO-Tiny/model.safetensors',
        'Qwen3-VL-4B-Instruct/weights-provenance.json',
        'Qwen3-VL-4B-Instruct/model-00001-of-00002.safetensors',
        'Qwen3-VL-4B-Instruct/model-00002-of-00002.safetensors',
    ))


def _digest(path):
    with path.open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def _validated_proof(path, manifest):
    proof = json.loads(path.read_bytes())
    if (proof.get('schema') != 'vision_workbench.annotation_assist_isolation.v1'
            or proof.get('profile') != PROFILE
            or proof.get('manifest_sha256') != _digest(manifest)
            or proof.get('appcontainer_token') is not True
            or type(proof.get('os_network_capabilities')) is not int
            or proof['os_network_capabilities'] != 0
            or proof.get('network_denied') is not True
            or proof.get('host_files_denied') is not True
            or proof.get('cuda') is not True):
        raise RuntimeError('無法驗證背景作業的斷網 AppContainer 與 CUDA 裝置。')
    return proof


class WorkerProgress:
    """Tail complete JSON events while leaving model diagnostics on disk."""

    def __init__(self, work):
        self.work = work
        self.path = None
        self.position = 0
        self.state = ('載入斷網推論環境與本機模型…', 1, 'initializing')

    def read(self):
        if self.path is None:
            self.path = self._locate_log()
        if self.path is None:
            return self.state
        try:
            with self.path.open('rb') as stream:
                stream.seek(self.position)
                while line := stream.readline():
                    if not line.endswith(b'\n'):
                        break
                    self.position = stream.tell()
                    try:
                        event = json.loads(line)
                    except (ValueError, UnicodeError):
                        continue
                    if isinstance(event, dict) and event.get('event') == 'progress':
                        self._set(event)
        except FileNotFoundError:
            # The launcher removes its isolated directory at the end of the run.
            pass
        return self.state

    def _locate_log(self):
        try:
            local = os.environ['LOCALAPPDATA']
            folder = json.loads((self.work / 'live.json').read_bytes())['run_folder']
        except (OSError, ValueError, KeyError):
            return None
        if not isinstance(folder, str) or not re.fullmatch(r'run-[a-f0-9]{32}', folder):
            return None
        return Path(local) / 'Packages' / PROFILE / 'AC' / 'Temp' / folder / 'worker.log'

    def _set(self, event):
        phase = event.get('stage')
        if phase not in PHASES:
            return
        message, initial, span = PHASES[phase]
        complete, total = event.get('completed'), event.get('total')
        if type(complete) is int and type(total) is int and 0 <= complete <= total and total:
            message = f'{message} {complete}／{total} 張'
            initial += int(span * complete / total)
        self.state = message, max(initial, self.state[1]), phase


def _last_diagnostic(path):
    if not path.is_file():
        return ''
    with path.open('rb') as stream:
        stream.seek(max(0, path.stat().st_size - 8192))
        lines = stream.read().decode('utf-8', errors='replace').splitlines()
    return '\n'.join(lines[-5:])[:1500]


def _execute(command, work, checkpoint, cancel_file):
    with (work / 'launcher.log').open('wb') as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            while process.poll() is None:
                checkpoint()
                time.sleep(.2)
            checkpoint()
            return process.returncode
        finally:
            if process.poll() is None:
                # The launcher requests cooperative cleanup and owns a kill-on-close Job Object.
                cancel_file.touch()
                try:
                    process.wait(timeout=18)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)


def run_batch(root, manifest_path, output_path, progress):
    """Read a staged manifest and atomically return suggestions and isolation proof."""
    if not available():
        raise RuntimeError(f'需要 Windows、獨立 CUDA Python 環境及已核對的 MM／Qwen 本機模型：{model_home()}')
    manifest = Path(manifest_path).resolve(strict=True)
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(f'標註結果已存在：{output}')
    work_parent = output.parent
    work_parent.mkdir(parents=True, exist_ok=True)

    def waiting():
        progress('等待 GPU…', 0, phase='gpu_queue')

    with accelerator_lease(root, checkpoint=waiting), TemporaryDirectory(
            prefix='.annotation-assist-', dir=work_parent) as temporary:
        work = Path(temporary)
        cancel_file = work / 'cancel.request'
        command = ['powershell.exe', '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass',
                   '-File', str(ROOT / 'scripts' / 'run-annotation-assist.ps1'),
                   '-ManifestPath', str(manifest), '-Destination', str(work),
                   '-ModelHome', str(model_home()), '-CancelFile', str(cancel_file)]
        status = WorkerProgress(work)

        def checkpoint():
            message, percent, phase = status.read()
            progress(message, percent, phase=phase)

        try:
            code = _execute(command, work, checkpoint, cancel_file)
            if code != 0:
                detail = _last_diagnostic(work / 'worker.log') or _last_diagnostic(work / 'launcher.log')
                raise RuntimeError(f'本機斷網推論失敗（{code}）。\n{detail}')
            proof = _validated_proof(work / 'isolation.json', manifest)
            checkpoint()
            (work / 'inference.json').replace(output)
            return proof
        finally:
            for name in ('worker.log', 'launcher.log', 'isolation.json'):
                if (work / name).is_file():
                    shutil.copyfile(work / name, output.with_name(f'{output.stem}-{name}'))
