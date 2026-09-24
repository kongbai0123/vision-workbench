"""Replay timing traces; never call training. Historical inputs are read-only."""
import argparse
import json
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from workbench.time_estimation import RunTiming


def replay(durations):
    now = [0.0]
    tracker = RunTiming({'training': len(durations)}, clock=lambda: now[0], wall=lambda: now[0])
    tracker.update('training', 0, len(durations))
    actual_remaining = sum(durations)
    errors, averages, truths = [], [], []
    start = time.perf_counter()
    first = None
    for index, seconds in enumerate(durations, 1):
        now[0] += seconds
        actual_remaining -= seconds
        result = tracker.update('training', index, len(durations))
        if first is None and result['remaining_seconds'] is not None:
            first = index
        # Rolling-origin evaluation: forecast sees only elapsed observations.
        # The first three observations are the explicitly labelled quick estimate.
        if 4 <= index < len(durations) and result['remaining_seconds'] is not None:
            prediction = result['remaining_seconds']
            errors.append(abs(prediction - actual_remaining))
            averages.append(abs(now[0] / index * (len(durations) - index) - actual_remaining))
            truths.append(actual_remaining)
    runtime = (time.perf_counter() - start) * 1000
    anomalies = tracker.phases['training'].anomalies
    tracker.close()
    return {'units': len(durations), 'forecast_count': len(errors), 'anomalous_gaps': anomalies, 'first_estimate_after_units': first,
            'mae_seconds': round(sum(errors) / len(errors), 4),
            'whole_run_mean_mae_seconds': round(sum(averages) / len(averages), 4),
            'weighted_absolute_percentage_error': round(sum(errors) / sum(truths) * 100, 3) if sum(truths) else 0,
            'estimator_ms_per_update': round(runtime / len(durations), 4)}


def historical_epoch_durations(path):
    points, seen = [], set()
    # Exclude truncated events and duplicate snapshots; no image/model reads.
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            try:
                event = json.loads(line);state = event['state'];metrics = state.get('metrics') or {}
                epoch = metrics.get('epoch')
                if state.get('status') != 'running' or not isinstance(epoch, int) or epoch in seen:
                    continue
                seen.add(epoch);points.append((epoch, float(event['time'])))
            except (ValueError, KeyError, TypeError):
                continue
    durations = [b[1]-a[1] for a, b in zip(points, points[1:]) if b[0] == a[0]+1 and b[1] > a[1]]
    return durations if len(durations) >= 5 else []


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history-root', type=Path)
    parser.add_argument('--output', type=Path, default=Path('qa-output/time-estimation/benchmark.json'))
    args = parser.parse_args()
    rng = random.Random(42)
    traces = {'steady': [2]*80, 'jitter': [rng.uniform(1, 3) for _ in range(80)],
              'cold_start': [20]+[2]*79, 'slowdown': [1]*40+[3]*40,
              'speedup': [3]*40+[1]*40}
    report = {'training_started': False, 'synthetic': {key: replay(values) for key, values in traces.items()},
              'historical': [], 'historical_scope': 'epoch completion intervals including validation, not Batch timing'}
    if args.history_root:
        for path in sorted(args.history_root.glob('*/runs/*/events.jsonl')):
            values = historical_epoch_durations(path)
            if values:
                # Keep private project names, IDs and raw timings out of the report.
                report['historical'].append(replay(values))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
