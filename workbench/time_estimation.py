"""Count-based ETA: tqdm rates, online error weighting, bounded phase history.

The range is a volatility band, not a calibrated probability interval. No model
is loaded or executed to produce a forecast. See docs/time-estimation.md.
"""
from collections import deque
from contextlib import contextmanager
from hashlib import sha256
import io
import json
import math
from pathlib import Path
import platform
from statistics import median
import time

from tqdm import tqdm

VERSION = 1
TERMINAL = {'completed', 'failed', 'stopped'}
LABELS = {'preparation': '準備資料與模型', 'training': '訓練', 'validation': '每輪驗證',
          'final_validation': '最終驗證', 'test': 'Test 評估', 'saving': '保存模型',
          'read_images': '讀取原圖', 'threshold': '門檻搜尋', 'checkpoint': '保存權重',
          'engine_validation': '引擎最終檢查', 'finalizing': '整理結果'}


class QuietMeter(tqdm):
    """Use actual tqdm EMA accounting without a terminal, monitor, or timer thread."""
    monitor_interval = 0

    def __init__(self, total, clock):
        super().__init__(total=total, smoothing=.3, mininterval=0, miniters=1,
                         file=io.StringIO(), leave=False, disable=False)
        # Pin tqdm in every runtime. This is the sole adapter to its clock fields;
        # an active phase clock excludes validation and queue time from Batch ETA.
        self._time = clock
        self.last_print_t = self.start_t = clock()

    def display(self, *args, **kwargs):
        pass


class PhaseRate:
    def __init__(self, total, completed=0):
        self.total, self.completed, self.seconds = total, completed, 0.0
        self.last_sample_seconds = 0.0
        self.samples = deque(maxlen=32)
        self.errors = [deque(maxlen=16) for _ in range(3)]
        self.estimates = [None] * 3
        self.meters = [QuietMeter(total, lambda: self.seconds) for _ in range(3)]
        self.anomalies = 0
        # Responsive EMA, stable EMA and cumulative rate. Score BEFORE updating.
        for meter, alpha in zip(self.meters, (.15, .5, 0)):
            meter.smoothing = alpha
            meter.reset(total)

    @staticmethod
    def seconds_per_unit(meter):
        data = meter.format_dict
        if data['rate']:
            return 1 / data['rate']
        return data['elapsed'] / data['n'] if data['n'] and data['elapsed'] > 0 else None

    def rate(self):
        values = self.estimates
        available = [(i, value) for i, value in enumerate(values) if value and math.isfinite(value)]
        if not available:
            return None
        weights = [(i, 1 / max(.000001, sum(self.errors[i]) / len(self.errors[i])) ** 2)
                   if self.errors[i] else (i, 1.0) for i, _ in available]
        return sum(values[i] * weight for i, weight in weights) / sum(weight for _, weight in weights)

    def observe(self, completed, total):
        if total != self.total or completed < self.completed:
            seconds, anomalies = self.seconds, self.anomalies
            self.close()
            self.__init__(total, completed)  # Changed work plan: never reuse stale velocity.
            self.seconds = self.last_sample_seconds = seconds
            self.anomalies = anomalies + 1
            for meter in self.meters:
                meter.reset(total)
            return
        delta = completed - self.completed
        if delta <= 0:
            return
        elapsed = self.seconds - self.last_sample_seconds
        if elapsed > 0:
            sample = elapsed / delta
            if len(self.samples) >= 4 and sample > max(30, median(self.samples) * 8):
                # An overnight suspension is not evidence that every next Batch
                # will take hours. Preserve elapsed/counts, reacquire fresh speed.
                self.anomalies += 1
                for meter in self.meters:
                    meter.reset(self.total)
                self.samples.clear()
                self.estimates = [None] * 3
                for errors in self.errors:
                    errors.clear()
                self.completed, self.last_sample_seconds = completed, self.seconds
                return
            for i, meter in enumerate(self.meters):
                rate = self.estimates[i]
                if rate:
                    self.errors[i].append(abs(rate - sample))
                meter.update(delta)
                self.estimates[i] = self.seconds_per_unit(meter)
            self.samples.append(sample)
        self.completed = completed
        self.last_sample_seconds = self.seconds

    def spread(self, rate):
        if not self.samples:
            return rate * .5
        residuals = [abs(value - rate) for value in self.samples]
        return max(rate * (.5 if len(self.samples) < 4 else .15), median(residuals) * 1.4826)

    def close(self):
        for meter in self.meters:
            meter.close()


def work_plan(summary, config, component=None):
    epochs = int(config.get('epochs', 1))
    splits = summary.get('splits', {})
    builtin = summary.get('batches_per_epoch') is None
    plan = {'preparation': 1}
    if builtin:
        plan.update(read_images=splits.get('train', 0), threshold=epochs)
    else:
        plan['training'] = epochs * summary['batches_per_epoch']
        if splits.get('val'):
            plan['validation'] = epochs
        elif component == 'ultralytics':
            # Pinned trainer validates its final epoch even with args.val=False.
            # This internal check is not independent held-out evaluation.
            plan['validation'] = 1
        if component == 'ultralytics':
            plan['engine_validation'] = 1
        plan['checkpoint'] = 1
    if splits.get('val'):
        plan['final_validation'] = 1
    if splits.get('test'):
        plan['test'] = 1
    plan['saving'] = 1
    return {key: value for key, value in plan.items() if value > 0}


def environment_key(python):
    """Local metadata only: no torch import, GPU query, process, or model load."""
    python = Path(python)
    paths = [python] + [python.parent.parent / 'Lib' / 'site-packages' / path for path in
                       ('torch/version.py', 'torchvision/version.py', 'ultralytics/__init__.py', 'tqdm/_version.py')]
    stamps = []
    for path in paths:
        try:
            stat = path.stat(); stamps.append([str(path), stat.st_size, stat.st_mtime_ns])
        except OSError:
            stamps.append([str(path), None])
    return sha256(json.dumps([platform.node(), platform.machine(), platform.processor(), stamps]).encode()).hexdigest()


def match_key(manifest_hash, engine, config, environment=None):
    # Exact dataset and execution settings; only the planned epoch count scales.
    fields = {key: value for key, value in config.items() if key != 'epochs'}
    return sha256(json.dumps([VERSION, manifest_hash, engine, fields, environment], sort_keys=True,
                            separators=(',', ':')).encode()).hexdigest()


def historical_estimate(runs, key, plan, now=None):
    now = time.time() if now is None else now
    candidates = [run for run in runs if isinstance(run, dict) and run.get('status') == 'completed'
                  and run.get('timing_match_key') == key
                  and isinstance(run.get('completed_at'), (int, float))
                  and 0 <= now - run.get('completed_at', 0) <= 30 * 86400
                  and isinstance(run.get('timing_profile'), dict)
                  and run.get('timing_profile', {}).get('schema_version') == VERSION][:12]
    priors = {}
    for phase in plan:
        values = []
        for run in candidates:
            phases = run['timing_profile'].get('phases', {})
            if not isinstance(phases, dict):
                continue
            value = phases.get(phase, {})
            if not isinstance(value, dict):
                continue
            units, seconds = value.get('completed', 0), value.get('seconds', 0)
            if (type(units) in (int, float) and type(seconds) in (int, float)
                    and math.isfinite(units) and math.isfinite(seconds)
                    and units > 0 and seconds > 0 and value.get('complete') and not value.get('anomalies')):
                values.append(seconds / units)
        if values:
            rate = median(values)
            priors[phase] = {'seconds_per_unit': rate, 'samples': len(values),
                             'spread': max(rate * .3, median(abs(v - rate) for v in values) * 1.4826)}
    missing = [phase for phase in plan if phase not in priors]
    seconds = sum(plan[p] * priors[p]['seconds_per_unit'] for p in plan if p in priors)
    spread = sum(plan[p] * priors[p]['spread'] for p in plan if p in priors)
    return {'schema_version': VERSION, 'source': 'history' if priors else 'awaiting_samples',
            'history_runs': len(candidates), 'remaining_seconds': seconds if not missing else None,
            'lower_seconds': max(0, seconds - spread) if not missing else None,
            'upper_seconds': seconds + spread if not missing else None,
            'missing_phases': missing, 'priors': priors, 'plan': plan,
            'scope': 'planned_run', 'training_started': False}


class RunTiming:
    def __init__(self, plan=None, priors=None, clock=time.monotonic, wall=time.time):
        self.clock, self.wall = clock, wall
        self.started = self.last = clock()
        self.phase, self.phases = None, {}
        self.plan, self.priors = dict(plan or {}), priors or {}
        self.done = set()
        self.last_advance = self.last

    def update(self, phase, completed=None, total=None, status='running'):
        now = self.clock()
        if self.phase:
            self.phases[self.phase].seconds += max(0, now - self.last)
        self.last = now
        if self.phase != phase:
            if self.phase == 'preparation':
                self.phases[self.phase].observe(1, 1)
                self.done.add(self.phase)
            self.phase = phase
            self.last_advance = now
        if phase not in self.phases:
            self.phases[phase] = PhaseRate(total or self.plan.get(phase, 1), completed or 0)
        meter = self.phases[phase]
        if total is not None:
            self.plan[phase] = total
            before = meter.completed
            meter.observe(completed, total)
            if completed > before:
                self.last_advance = now
            if completed >= total:
                self.done.add(phase)
            else:
                self.done.discard(phase)
        remaining, spread, missing = 0.0, 0.0, []
        phase_seconds = None
        sources = set()
        for name, count in self.plan.items():
            current = self.phases.get(name)
            units = 0 if name in self.done else max(0, count - (current.completed if current else 0))
            if not units:
                continue
            prior = self.priors.get(name, {})
            rate = current.rate() if current else None
            if rate:
                sources.add('live')
                width = current.spread(rate)
            else:
                rate = prior.get('seconds_per_unit')
                width = prior.get('spread', rate * .5 if rate else 0)
                if rate:
                    sources.add('history')
            if rate:
                seconds = units * rate
                # One-unit stages have no intermediate counters. Their historical
                # duration is reduced by actual active time, never fake progress.
                if current and not current.completed and name == phase:
                    seconds = max(0, seconds - current.seconds)
                remaining += seconds
                spread += units * width
                if name == phase:
                    phase_seconds = seconds
            else:
                missing.append(name)
        terminal = status in TERMINAL
        result = {'schema_version': VERSION, 'phase': phase, 'phase_label': LABELS.get(phase, phase),
                  'completed': meter.completed, 'total': meter.total, 'elapsed_seconds': now - self.started,
                  'phase_remaining_seconds': phase_seconds, 'remaining_seconds': remaining if not missing else None,
                  'lower_seconds': max(0, remaining - spread) if not missing else None,
                  'upper_seconds': remaining + spread if not missing else None,
                  'missing_phases': missing, 'source': '+'.join(sorted(sources)) or 'awaiting_samples',
                  'samples': len(meter.samples), 'anomalies': meter.anomalies, 'updated_at': self.wall(),
                  'stale_after_seconds': max(30, (meter.rate() or self.priors.get(phase, {}).get('seconds_per_unit', 0)) * 4),
                  'seconds_since_advance': now - self.last_advance,
                  'scope': 'planned_run', 'status': status}
        if terminal:
            result.update(remaining_seconds=0 if status == 'completed' else None,
                          phase_remaining_seconds=None, lower_seconds=None, upper_seconds=None)
        return result

    def profile(self):
        return {'schema_version': VERSION, 'phases': {name: {'seconds': p.seconds, 'completed': p.completed,
                 'total': p.total, 'anomalies': p.anomalies, 'complete': name in self.done} for name, p in self.phases.items()}}

    def close(self):
        for phase in self.phases.values():
            phase.close()


_trackers = {}


def record_timing(run_dir, run, changes):
    key = str(run_dir.resolve())
    tracker = _trackers.get(key)
    if tracker is None:
        tracker = _trackers[key] = RunTiming(run.get('timing_plan'), run.get('timing_prior'))
    phase = changes.get('phase', tracker.phase or 'preparation')
    work = changes.pop('timing_work', None)
    if phase == 'training' and 'batch' in changes and 'batches_per_epoch' in changes:
        batches, epoch = changes['batches_per_epoch'], changes.get('epoch', run.get('epoch', 1))
        work = {'completed': (epoch - 1) * batches + changes['batch'],
                'total': int(run['config']['epochs']) * batches}
    timing = tracker.update(phase, **(work or {}), status=changes.get('status', run.get('status', 'running')))
    run['timing'], run['timing_profile'] = timing, tracker.profile()
    if timing['status'] in TERMINAL:
        tracker.close()
        del _trackers[key]


@contextmanager
def timed_phase(run_dir, run, phase, *, completed=0, total=1):
    from .training_engine import _status
    _status(run_dir, run, status='running', phase=phase, message=LABELS.get(phase, phase),
            timing_work={'completed': completed, 'total': total})
    yield
    _status(run_dir, run, timing_work={'completed': completed + 1, 'total': total})
