"""Bounded subprocess execution with cancellation even when output is silent."""
import os
import subprocess
import tempfile
import time
import signal


def run_controlled(command, progress, *, cwd, timeout=1800):
    with tempfile.TemporaryFile(mode='w+b') as output:
        process = subprocess.Popen(command, cwd=str(cwd), stdin=subprocess.DEVNULL,
                                   stdout=output, stderr=subprocess.STDOUT,
                                   start_new_session=os.name != 'nt',
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        deadline = time.monotonic() + timeout
        try:
            while process.poll() is None:
                progress('子程序執行中…', None)
                if time.monotonic() >= deadline:
                    raise TimeoutError(f'子程序超過 {timeout} 秒')
                time.sleep(.2)
            progress('子程序執行完成', None)
            output.seek(0, os.SEEK_END)
            size = output.tell()
            output.seek(max(0, size - 16384))
            detail = output.read().decode('utf-8', errors='replace')
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
