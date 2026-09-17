"""Streaming subprocesses with cancellation and independent idle/total limits."""
import codecs
import math
import os
import signal
import subprocess
import threading
import time


def run_controlled(command, progress, *, cwd, timeout=1800, idle_timeout=None, env=None):
    for limit in (timeout, idle_timeout):
        if limit is not None and (not math.isfinite(limit) or limit <= 0):
            raise ValueError('逾時必須為正數或 None')
    process = subprocess.Popen(command, cwd=str(cwd), stdin=subprocess.DEVNULL,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               env={**os.environ, **(env or {})}, start_new_session=os.name != 'nt',
                               creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    started = time.monotonic()
    state = {'tail': '', 'last_output': started, 'message': '子程序執行中…'}
    lock = threading.Lock()

    def consume():
        decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
        while True:
            chunk = os.read(process.stdout.fileno(), 4096)
            text = decoder.decode(chunk, final=not chunk)
            with lock:
                state['tail'] = (state['tail'] + text)[-16384:]
                if chunk:
                    state['last_output'] = time.monotonic()
                lines = state['tail'].replace('\r', '\n').splitlines()
                meaningful = next((line for line in reversed(lines) if any(c.isalnum() for c in line)), None)
                if meaningful:
                    state['message'] = meaningful[-1000:]
            if not chunk:
                break

    reader = threading.Thread(target=consume, name='subprocess-output', daemon=True)
    reader.start()
    try:
        while process.poll() is None:
            with lock:
                current = dict(state)
            progress(current['message'], None)
            now = time.monotonic()
            if timeout is not None and now - started >= timeout:
                raise TimeoutError(f'子程序超過總時間 {timeout} 秒\n{current["tail"]}')
            if idle_timeout is not None and now - current['last_output'] >= idle_timeout:
                raise TimeoutError(f'子程序連續 {idle_timeout} 秒沒有新輸出\n{current["tail"]}')
            time.sleep(.1)
        reader.join(timeout=2)
        with lock:
            detail, message = state['tail'], state['message']
        progress(message, None)
        if process.returncode:
            raise RuntimeError(detail.strip() or f'子程序結束碼 {process.returncode}')
        return detail
    finally:
        if process.poll() is None:
            if os.name == 'nt':
                try:
                    subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                                   capture_output=True, timeout=15,
                                   creationflags=subprocess.CREATE_NO_WINDOW)
                except (OSError, subprocess.TimeoutExpired):
                    pass
            else:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            if process.poll() is None:
                process.kill()
            process.wait(timeout=15)
        reader.join(timeout=2)
        if not reader.is_alive():
            process.stdout.close()


def installation_options():
    """Inherited by pip itself and pip launched through bootstrap.ps1."""
    def seconds(key, default):
        value = float(os.environ.get(key, default))
        if not math.isfinite(value) or value < 0:
            raise ValueError(f'{key} 不可小於 0')
        return value or None
    return {'timeout': seconds('VISION_WORKBENCH_INSTALL_TIMEOUT', '0'),
            'idle_timeout': seconds('VISION_WORKBENCH_INSTALL_IDLE_TIMEOUT', '1800'),
            'env': {'PIP_PROGRESS_BAR': 'raw', 'PYTHONUNBUFFERED': '1'}}
