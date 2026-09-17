"""CLI boundary for a Vision Workbench training run."""
from __future__ import annotations

import argparse
from pathlib import Path

from .training_engine import read_json


def main(argv=None):
    parser = argparse.ArgumentParser(description="Vision Workbench isolated training worker")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    from .engine_specs import engine_spec
    from .worker_lifecycle import worker_lifecycle
    run = read_json(args.run_dir / 'run.json')
    try:
        with worker_lifecycle(args.run_dir, run):
            engine_spec(run['engine']).train(args.dataset.resolve(), args.run_dir.resolve(), args.model_dir.resolve())
    except InterruptedError as exc:
        import time
        from .training_engine import atomic_json
        run.update(status='stopped', message=str(exc), completed_at=time.time(), updated_at=time.time())
        atomic_json(args.run_dir / 'run.json', run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
