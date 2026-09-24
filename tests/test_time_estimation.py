"""Deterministic clocks and callback replay only; no model or training execution."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from workbench.time_estimation import RunTiming, historical_estimate, match_key, work_plan, _trackers
from workbench.run_events import read_state
from workbench.ultralytics_engine import _RunMetricsRecorder, ULTRALYTICS_ENGINES
from workbench.training_engine import _status


class TimeEstimationTests(unittest.TestCase):
    def clocked(self, plan, priors=None):
        now = [0.0]
        tracker = RunTiming(plan, priors, clock=lambda: now[0], wall=lambda: 1000 + now[0])
        self.addCleanup(tracker.close)
        return now, tracker

    def test_first_completed_batch_has_a_real_tqdm_estimate(self):
        now, tracker = self.clocked({'training': 10})
        tracker.update('training', 0, 10)
        now[0] = 2
        result = tracker.update('training', 1, 10)
        self.assertAlmostEqual(result['remaining_seconds'], 18)
        self.assertEqual(result['samples'], 1)
        self.assertEqual(result['source'], 'live')
        self.assertEqual(tracker.phases['training'].meters[0].format_dict['rate'], .5)
        now[0] = 40
        result = tracker.update('training', 1, 10)
        self.assertEqual(result['samples'], 1)
        self.assertGreater(result['seconds_since_advance'], result['stale_after_seconds'])

    def test_validation_does_not_contaminate_batch_speed(self):
        now, tracker = self.clocked({'training': 4, 'validation': 1})
        tracker.update('training', 0, 4)
        now[0] = 2; tracker.update('training', 1, 4)
        tracker.update('validation', 0, 1)
        now[0] = 12; tracker.update('validation', 1, 1)
        tracker.update('training', 1, 4)
        now[0] = 14
        result = tracker.update('training', 2, 4)
        self.assertAlmostEqual(result['remaining_seconds'], 4)
        self.assertAlmostEqual(tracker.profile()['phases']['validation']['seconds'], 10)
        self.assertAlmostEqual(tracker.profile()['phases']['training']['seconds'], 4)

    def test_suspend_gap_reacquires_rate_without_learning_an_overnight_batch(self):
        now, tracker = self.clocked({'training': 10})
        tracker.update('training', 0, 10)
        for count in range(1, 5):
            now[0] += 2;tracker.update('training', count, 10)
        now[0] += 54000
        interrupted = tracker.update('training', 5, 10)
        self.assertIsNone(interrupted['remaining_seconds'])
        self.assertEqual(interrupted['anomalies'], 1)
        now[0] += 2
        resumed = tracker.update('training', 6, 10)
        self.assertAlmostEqual(resumed['remaining_seconds'], 8)
        self.assertEqual(tracker.profile()['phases']['training']['seconds'], 54010)

    def test_same_counter_does_not_change_any_tqdm_rate(self):
        now, tracker = self.clocked({'training': 10})
        tracker.update('training', 0, 10)
        now[0] = 2;first = tracker.update('training', 1, 10)
        now[0] = 20;again = tracker.update('training', 1, 10)
        self.assertEqual(first['remaining_seconds'], again['remaining_seconds'])
        self.assertEqual(first['stale_after_seconds'], again['stale_after_seconds'])

    def test_changed_work_plan_resets_the_estimator_and_terminal_cannot_keep_countdown(self):
        now, tracker = self.clocked({'training': 10})
        tracker.update('training', 0, 10)
        now[0] = 2; tracker.update('training', 1, 10)
        result = tracker.update('training', 1, 20)
        self.assertIsNone(result['remaining_seconds'])
        self.assertEqual(result['samples'], 0)
        self.assertIsNone(tracker.update('training', status='failed')['remaining_seconds'])
        self.assertEqual(tracker.update('training', status='completed')['remaining_seconds'], 0)

    def test_unknown_future_phases_are_not_silently_omitted(self):
        now, tracker = self.clocked({'training': 10, 'saving': 1})
        tracker.update('training', 0, 10)
        now[0] = 2
        result = tracker.update('training', 1, 10)
        self.assertIsNone(result['remaining_seconds'])
        self.assertEqual(result['phase_remaining_seconds'], 18)
        self.assertEqual(result['missing_phases'], ['saving'])

    def test_history_covers_cold_start_then_live_rate_replaces_it(self):
        priors = {p: {'seconds_per_unit': rate, 'spread': rate * .3}
                  for p, rate in [('preparation', 5), ('training', 3), ('saving', 4)]}
        now, tracker = self.clocked({'preparation': 1, 'training': 10, 'saving': 1}, priors)
        self.assertEqual(tracker.update('preparation')['remaining_seconds'], 39)
        now[0] = 2
        tracker.update('training', 0, 10)
        now[0] = 3
        result = tracker.update('training', 1, 10)
        self.assertEqual(result['remaining_seconds'], 13)
        self.assertEqual(result['source'], 'history+live')
        self.assertTrue(tracker.profile()['phases']['preparation']['complete'])

    def test_adaptive_tqdm_replay_beats_whole_run_mean_after_speed_change(self):
        durations = [8] + [1] * 19 + [3] * 20
        now, tracker = self.clocked({'training': len(durations)})
        tracker.update('training', 0, len(durations))
        adaptive, average = [], []
        for index, duration in enumerate(durations, 1):
            now[0] += duration
            result = tracker.update('training', index, len(durations))
            if 24 <= index < len(durations):
                truth = sum(durations[index:])
                adaptive.append(abs(result['remaining_seconds'] - truth))
                average.append(abs(now[0] / index * (len(durations) - index) - truth))
        self.assertLess(sum(adaptive), sum(average) * .5)

    def test_history_is_bounded_by_configuration_freshness_and_completed_profiles(self):
        key = match_key('dataset-sha', 'engine', {'epochs': 5, 'batch_size': 2})
        self.assertEqual(key, match_key('dataset-sha', 'engine', {'epochs': 10, 'batch_size': 2}))
        self.assertNotEqual(key, match_key('dataset-sha', 'engine', {'epochs': 5, 'batch_size': 4}))
        self.assertNotEqual(key, match_key('different-sha', 'engine', {'epochs': 5, 'batch_size': 2}))
        base = {'status': 'completed', 'completed_at': 1000, 'timing_match_key': key,
                'timing_profile': {'schema_version': 1, 'phases': {'training': {'completed': 20, 'seconds': 40, 'complete': True}}}}
        records = [base, {**base, 'status': 'stopped'}, {**base, 'completed_at': -9999999},
                   {**base, 'timing_match_key': 'other'}]
        estimate = historical_estimate(records, key, {'training': 40}, now=1001)
        self.assertEqual(estimate['history_runs'], 1)
        self.assertEqual(estimate['remaining_seconds'], 80)
        self.assertFalse(estimate['training_started'])
        self.assertIsNone(historical_estimate(records, key, {'training': 40, 'test': 1}, now=1001)['remaining_seconds'])

    def test_plan_uses_expanded_batches_and_baseline_does_not_invent_batches(self):
        summary = {'batches_per_epoch': 45, 'splits': {'train': 60, 'val': 20, 'test': 20}}
        plan = work_plan(summary, {'epochs': 10})
        self.assertEqual(plan['training'], 450)
        self.assertEqual(plan['validation'], 10)
        self.assertEqual(plan['checkpoint'], 1)
        baseline = work_plan({**summary, 'batches_per_epoch': None}, {'epochs': 10})
        self.assertEqual(baseline['read_images'], 60)
        self.assertEqual(baseline['threshold'], 10)
        self.assertNotIn('training', baseline)

    def test_callback_replay_preserves_timing_through_event_snapshot_merging(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            run = {'status': 'preparing', 'config': {'epochs': 2, 'batch_size': 2},
                   'timing_plan': {'training': 10, 'validation': 2}}
            now, tracker = self.clocked(run['timing_plan'])
            key = str(folder.resolve()); _trackers[key] = tracker
            self.addCleanup(_trackers.pop, key, None)
            recorder = _RunMetricsRecorder(folder, run, ULTRALYTICS_ENGINES['yolo26n_detect'])
            trainer = SimpleNamespace(epoch=0, train_loader=[None]*5, batch_i=0, metrics={}, tloss={})
            recorder.on_epoch_start(trainer)
            now[0] = 2; recorder.on_batch_end(trainer)
            trainer.batch_i = 1; now[0] = 4; recorder.on_batch_end(trainer)
            state = read_state(folder / 'run.json')
            self.assertEqual(state['timing']['completed'], 2)
            self.assertEqual(state['timing']['phase_remaining_seconds'], 16)
            recorder.on_validation_start(trainer)
            now[0] = 14; recorder.on_epoch_end(trainer)
            state = read_state(folder / 'run.json')
            self.assertEqual(state['timing']['phase'], 'validation')
            self.assertEqual(state['timing_profile']['phases']['validation']['seconds'], 10)
            recorder.on_validator_start(None)
            now[0] = 18;recorder.on_validator_end(None)
            self.assertEqual(run['timing']['phase'], 'engine_validation')
            self.assertEqual(run['timing']['completed'], 1)
            self.assertEqual(run['epoch'], 1)
            _status(folder, run, status='stopped')
            self.assertNotIn(key, _trackers)
            self.assertIsNone(read_state(folder / 'run.json')['timing']['remaining_seconds'])


class QuickEstimateTests(unittest.TestCase):
    from tests import test_training as fixtures
    setUp = fixtures.TrainingWorkflowTests.setUp
    tearDown = fixtures.TrainingWorkflowTests.tearDown

    def test_quick_estimate_does_not_load_images_or_launch_training(self):
        dataset = self.workspace.create_dataset_version(self.pid, {'preset': 'light', 'expansion_count': 2})
        with patch('subprocess.Popen', side_effect=AssertionError('training forbidden')), patch.object(Path, 'read_bytes', side_effect=AssertionError('quick estimate must not read image bytes')):
            report = self.workspace.estimate_training_time(self.pid, dataset['id'], {'engine': 'pixel_prototype_v1'})
        self.assertFalse(report['training_started'])
        self.assertIsNone(report['remaining_seconds'])
        self.assertEqual(self.workspace.list_runs(self.pid), [])
        self.assertEqual(list(self.workspace.runs_dir(self.pid).glob('R*')), [])
        folder = self.workspace.runs_dir(self.pid, create=True) / 'R001';folder.mkdir()
        phases = {phase: {'complete': True, 'completed': units, 'seconds': units * 2} for phase, units in report['plan'].items()}
        import time
        record = {'status': 'completed', 'completed_at': time.time(), 'timing_match_key': report['match_key'],
                  'timing_profile': {'schema_version': 1, 'phases': phases}}
        (folder / 'run.json').write_text(json.dumps(record), encoding='utf-8')
        with patch('subprocess.Popen', side_effect=AssertionError('training forbidden')):
            estimate = self.workspace.estimate_training_time(self.pid, dataset['id'], {'engine': 'pixel_prototype_v1'})
        self.assertEqual(estimate['remaining_seconds'], sum(report['plan'].values()) * 2)
