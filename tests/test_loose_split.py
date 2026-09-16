"""Explicit opt-in random allocation and downstream coverage policy."""
import unittest

from workbench.smart_splitting import smart_split
from workbench.training import dataset_readiness
from workbench.split_quality import loose_split_applies


class LooseSplitTests(unittest.TestCase):
    def assets(self, n=57):
        return [{'id': str(i), 'sha256': str(i), 'batch_id': 'same-source',
                 'shapes': [{'label': f'unique-{i}'}]} for i in range(n)]

    def test_single_source_unique_classes_and_reproducibility(self):
        assets = self.assets()
        options = {'strategy': 'random_loose', 'seed': 42}
        plan = smart_split(assets, options)
        self.assertTrue(plan['ready'])
        self.assertEqual(plan['image_counts'], {'train': 40, 'val': 11, 'test': 6})
        self.assertEqual(plan['assignments'], smart_split(list(reversed(assets)), options)['assignments'])
        self.assertNotEqual(plan['assignments'], smart_split(assets, {**options, 'seed': 9})['assignments'])
        self.assertFalse(plan['source_isolation'])
        self.assertTrue(plan['coverage']['missing_train_classes'])
        self.assertTrue(plan['warnings'])
        self.assertTrue(all('split' not in a for a in assets))
        for asset in assets:
            asset['shapes'] = [{'label': 'same'}]
        self.assertEqual(plan['assignments'], smart_split(assets, options)['assignments'])

    def test_manifest_policy_is_explicit_and_allocation_bound(self):
        assets = self.assets(6)
        plan = smart_split(assets, {'strategy': 'random_loose'})
        rows = [{**a, 'split': plan['assignments'][a['id']]} for a in assets]
        self.assertTrue(dataset_readiness({'assets': rows, 'split_plan': plan})['ready'])
        self.assertFalse(dataset_readiness({'assets': rows})['ready'])
        rows[0]['split'] = 'val' if rows[0]['split'] != 'val' else 'train'
        self.assertFalse(loose_split_applies(plan, rows))
        self.assertFalse(dataset_readiness({'assets': rows, 'split_plan': plan})['ready'])

    def test_duplicates_locks_and_zero_test(self):
        assets = self.assets(10)
        assets[1]['sha256'] = assets[0]['sha256']
        options = {'strategy': 'random_loose', 'ratios': [80, 20, 0]}
        plan = smart_split(assets, options)
        self.assertEqual(plan['assignments']['0'], plan['assignments']['1'])
        group = next(g for g in plan['groups'] if '0' in g['asset_ids'])
        locked = smart_split(assets, {**options, 'locks': {group['id']: 'val'}})
        self.assertEqual(locked['assignments']['0'], 'val')
        self.assertEqual(locked['image_counts']['test'], 0)
        with self.assertRaises(ValueError):
            smart_split(self.assets(2), {'strategy': 'random_loose'})
        with self.assertRaises(ValueError):
            smart_split([], {'strategy': 'random_loose'})

    def test_split_only_train_allowed_but_training_requires_validation(self):
        assets = self.assets(1)
        plan = smart_split(assets, {'strategy': 'random_loose', 'ratios': [100, 0, 0]})
        self.assertTrue(plan['ready'])
        report = dataset_readiness({'assets': [{**assets[0], 'split': 'train'}], 'split_plan': plan})
        self.assertFalse(report['ready'])
        self.assertIn('no_validation_split', [item['code'] for item in report['blockers']])

    def test_all_train_policy_is_explicit_and_does_not_claim_evaluation(self):
        assets = self.assets(4)
        for asset in assets:
            asset['shapes'] = [{'label': 'same'}]
        plan = smart_split(assets, {'purpose': 'all_train'})
        rows = [{**asset, 'split': 'train'} for asset in assets]
        report = dataset_readiness({'assets': rows, 'split_plan': plan})
        self.assertTrue(report['ready'])
        self.assertEqual(report['purpose'], 'all_train')
        self.assertEqual(plan['image_counts'], {'train': 4, 'val': 0, 'test': 0})


if __name__ == '__main__':
    unittest.main()
