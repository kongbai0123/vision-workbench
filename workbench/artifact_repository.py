"""Immutable dataset summaries; sidecars are caches, never authoritative data."""
from copy import deepcopy
from collections import OrderedDict
import threading

from .training_engine import atomic_json, read_json


class ArtifactRepository:
    def __init__(self):
        self._cache = OrderedDict()
        self._lock = threading.RLock()

    def dataset_summary(self, path, build, *, persist=False):
        stat = path.stat()
        signature = [stat.st_mtime_ns, stat.st_size]
        key = (str(path), *signature)
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                return deepcopy(cached)
        sidecar = path.with_name('summary.json')
        summary = None
        if sidecar.is_file():
            try:
                saved = read_json(sidecar)
                if saved.get('schema_version') == 1 and saved.get('manifest_signature') == signature:
                    summary = saved['summary']
            except (OSError, ValueError, KeyError):
                pass
        if summary is None:
            summary = build(read_json(path))
            if persist:
                atomic_json(sidecar, {'schema_version': 1, 'manifest_signature': signature, 'summary': summary})
        with self._lock:
            self._cache[key] = summary
            self._cache.move_to_end(key)
            while len(self._cache) > 128:
                self._cache.popitem(last=False)
        return deepcopy(summary)
