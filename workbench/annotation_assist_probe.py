"""Verify Vision Workbench's Windows isolation before running inference."""

from contextlib import redirect_stderr, redirect_stdout
from ctypes import WinDLL, WinError, byref, c_int, create_string_buffer, get_last_error, sizeof
from ctypes import wintypes
import hashlib
import json
import os
from pathlib import Path
import runpy
import socket
import sys
import traceback


def token_capabilities():
    kernel = WinDLL("kernel32", use_last_error=True)
    security = WinDLL("advapi32", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    security.OpenProcessToken.argtypes = (wintypes.HANDLE, wintypes.DWORD, wintypes.LPHANDLE)
    security.GetTokenInformation.argtypes = (
        wintypes.HANDLE, wintypes.DWORD, wintypes.LPVOID, wintypes.DWORD, wintypes.LPDWORD,
    )
    token = wintypes.HANDLE()
    if not security.OpenProcessToken(kernel.GetCurrentProcess(), 0x0008, byref(token)):
        raise WinError(get_last_error())
    try:
        is_appcontainer, size = wintypes.DWORD(), wintypes.DWORD()
        if not security.GetTokenInformation(token, 29, byref(is_appcontainer),
                                            sizeof(is_appcontainer), byref(size)):
            raise WinError(get_last_error())
        if is_appcontainer.value != 1:
            raise RuntimeError("Worker token is outside an AppContainer.")
        security.GetTokenInformation(token, 30, None, 0, byref(size))
        if size.value < sizeof(wintypes.DWORD):
            raise WinError(get_last_error())
        buffer = create_string_buffer(size.value)
        if not security.GetTokenInformation(token, 30, buffer, size, byref(size)):
            raise WinError(get_last_error())
        return int(wintypes.DWORD.from_buffer(buffer).value)
    finally:
        kernel.CloseHandle(token)


def network_test(target):
    ip = target["ip"]
    family = socket.AF_INET6 if ":" in ip else socket.AF_INET
    try:
        with socket.socket(family, socket.SOCK_STREAM) as client:
            client.settimeout(3)
            client.connect((ip, target["port"]))
        return {"denied": False}
    except OSError as error:
        winerror = getattr(error, "winerror", None)
        return {"denied": isinstance(error, TimeoutError) or winerror == 10013, "winerror": winerror}


def read_denied(path):
    try:
        with Path(path).open("rb"):
            return False
    except PermissionError:
        return True


def native_gemma_cuda_available():
    from annotation_assist_gemma import RUNTIME_FILES

    runtime = Path(os.environ['VW_ANNOTATION_ASSIST_GEMMA_RUNTIME'])
    library = runtime / 'cudart64_13.dll'
    with library.open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != RUNTIME_FILES[library.name]:
            raise ValueError('The Gemma CUDA runtime failed its integrity check.')
    with os.add_dll_directory(str(runtime)):
        cuda = WinDLL(str(library))
        devices = c_int()
        return cuda.cudaGetDeviceCount(byref(devices)) == 0 and devices.value > 0


def verify(folder, prefix=''):
    config = json.loads(os.environ["VW_ANNOTATION_ASSIST_PROBE"])
    capability_count = token_capabilities()
    network = {target["name"]: network_test(target) for target in config["listeners"]}
    if "loopback_ipv4" not in network:
        raise ValueError("The control listener on IPv4 loopback was missing.")
    protected = {path: read_denied(path) for path in config["private_files"]}
    if not protected:
        raise ValueError("The host-only probe file was missing.")
    proof = {
        "schema": "vision_workbench.annotation_assist_isolation.v1",
        "profile": os.environ["VW_ANNOTATION_ASSIST_PROFILE"],
        "appcontainer_token": True,
        "os_network_capabilities": capability_count,
        "network_denied": all(row["denied"] for row in network.values()),
        "host_files_denied": all(protected.values()),
        "network": network,
        "protected_reference_files": protected,
        "cuda": None,
        "launcher_sha256": config["launcher_sha256"],
        "token_launcher_sha256": config["token_launcher_sha256"],
        "manifest_sha256": config.get("manifest_sha256"),
        "worker_sha256": {
            filename.name: hashlib.sha256(filename.read_bytes()).hexdigest()
            for filename in sorted(folder.glob("*.py"))
        },
    }
    destination = folder / f"{prefix}isolation.json"

    def publish_proof():
        destination.write_text(json.dumps(proof, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    publish_proof()
    if capability_count or not proof["network_denied"] or not proof["host_files_denied"]:
        raise PermissionError("AppContainer isolation could not be established.")
    if prefix == 'gemma-':
        proof["cuda"] = native_gemma_cuda_available()
    else:
        import torch

        proof["cuda"] = bool(torch.cuda.is_available())
        if proof["cuda"]:
            proof["cuda"] = torch.tensor([2, 3], device="cuda").sum().item() == 5
    publish_proof()
    if not proof["cuda"]:
        raise RuntimeError("The isolated runtime could not perform a CUDA operation.")
    print(f"Verified AppContainer, {len(network)} blocked network connections, "
          f"{len(protected)} inaccessible host paths.", flush=True)


def main():
    command = sys.argv[1:]
    folder = Path(__file__).resolve().parent
    prefix = 'gemma-' if os.environ.get('VW_ANNOTATION_ASSIST_OBSERVATION') == '1' else ''
    sys.path.insert(0, str(folder))
    with (folder / f"{prefix}worker.log").open("w", encoding="utf-8", buffering=1) as log:
        with redirect_stdout(log), redirect_stderr(log):
            try:
                verify(folder, prefix)
                sys.argv = command
                runpy.run_path(command[0], run_name="__main__")
            except SystemExit as exc:
                return exc.code
            except Exception:
                traceback.print_exc()
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
