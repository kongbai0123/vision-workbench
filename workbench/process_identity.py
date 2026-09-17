"""PID identity probes. Unknown access is never evidence that a worker died."""
import os
from pathlib import Path


def identity(pid):
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return {'alive': False, 'created': None}
    try:
        if pid <= 0:
            return {'alive': False, 'created': None}
        if os.name == 'nt':
            import ctypes
            from ctypes import wintypes
            kernel = ctypes.WinDLL('kernel32', use_last_error=True)
            kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
            kernel.OpenProcess.restype = wintypes.HANDLE
            kernel.CloseHandle.argtypes = [wintypes.HANDLE]
            kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
            kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
            handle = kernel.OpenProcess(0x1000, False, pid)
            if not handle:
                return {'alive': False if ctypes.get_last_error() == 87 else None, 'created': None}
            try:
                code = wintypes.DWORD()
                if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
                    return {'alive': None, 'created': None}
                times = [wintypes.FILETIME() for _ in range(4)]
                created = None
                if kernel.GetProcessTimes(handle, *(ctypes.byref(t) for t in times)):
                    created = str((times[0].dwHighDateTime << 32) | times[0].dwLowDateTime)
                return {'alive': code.value == 259, 'created': created}
            finally:
                kernel.CloseHandle(handle)
        os.kill(pid, 0)
        stat = Path(f'/proc/{pid}/stat')
        if stat.exists():
            fields = stat.read_text().rsplit(')', 1)[1].split()
            return {'alive': fields[0] != 'Z', 'created': fields[19]}
        return {'alive': True, 'created': None}
    except ProcessLookupError:
        return {'alive': False, 'created': None}
    except (OSError, TypeError, ValueError):
        return {'alive': None, 'created': None}


def alive(pid, created=None):
    result = identity(pid)
    if created is not None and result['created'] is not None and str(created) != result['created']:
        return False
    return result['alive']
