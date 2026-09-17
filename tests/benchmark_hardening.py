"""Read/write performance on isolated synthetic data, never user projects."""
import json
from pathlib import Path
import statistics
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image
from workbench.store import ProjectStore
from workbench.training import TrainingWorkspace
from workbench.training_engine import atomic_json


def main():
    with tempfile.TemporaryDirectory(prefix='hardening-benchmark-') as folder:
        root = Path(folder)
        store = ProjectStore(root / 'projects')
        pid = store.create_project('benchmark')['id']
        records = []
        for index in range(800):
            source = root / f'{index}.png'
            Image.new('RGB', (16, 16), (index % 256, index // 256, 60)).save(source)
            records.append({'path': source, 'shapes': [{'id': 'shape', 'type': 'rectangle',
                           'label': 'part', 'x': 1, 'y': 1, 'width': 8, 'height': 8}]})
        aid = store.add_assets(pid, records)['asset_ids'][0]
        full_times, delta_times = [], []
        for partial, timings in [(False, full_times), (True, delta_times)]:
            for _ in range(12):
                project = store.get_project(pid, include_assets=False)
                asset = store.get_asset(pid, aid)
                start = time.perf_counter()
                result = store.review(pid, [aid], 'approved', {aid: asset['revision']},
                                      delta_base=project['revision'] if partial else None)
                timings.append((time.perf_counter() - start) * 1000)
            if partial:
                delta_bytes = len(json.dumps(result).encode())
            else:
                full_bytes = len(json.dumps(result).encode())
        workspace = object.__new__(TrainingWorkspace)
        workspace.store = store
        workspace.runs = root / 'runs'
        run_dir = workspace.runs / pid / 'R001'
        run_dir.mkdir(parents=True)
        atomic_json(run_dir / 'run.json', {'run_id': 'R001', 'status': 'running', 'epoch': 100})
        durations = []
        for _ in range(100):
            start = time.perf_counter()
            workspace.status(pid)
            durations.append((time.perf_counter()-start)*1000)
        print(json.dumps({'images': 800, 'review_full_median_ms': round(statistics.median(full_times), 3),
                          'review_delta_median_ms': round(statistics.median(delta_times), 3),
                          'full_response_bytes': full_bytes, 'delta_response_bytes': delta_bytes,
                          'status_median_ms': round(statistics.median(durations), 3),
                          'status_max_ms': round(max(durations), 3)}, indent=2))


if __name__ == '__main__':
    main()
