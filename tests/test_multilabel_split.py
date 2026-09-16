from collections import Counter
from copy import deepcopy
import unittest

from workbench.smart_splitting import smart_split, source_groups


def data():
    return [{'id': str(i), 'sha256': str(i), 'batch_id': 'capture', 'source': {}, 'split': '',
             'shapes': [{'label': 'common'}] * (1 + i % 3) +
                       ([{'label': 'rare'}] if i < 3 else []) +
                       ([{'label': 'pair'}] if i % 2 == 0 else [])}
            for i in range(24)]


class MultilabelSplitTests(unittest.TestCase):
    def options(self, **kwargs):
        return {'strategy': 'multilabel', 'source_isolation': False, **kwargs}

    def test_multiple_objects_are_not_multiple_images_and_pairs_stay_together(self):
        assets = data(); before = deepcopy(assets)
        plan = smart_split(assets, self.options())
        self.assertTrue(plan['ready'])
        self.assertEqual(assets, before)
        self.assertEqual(plan, smart_split(list(reversed(assets)), self.options()))
        self.assertEqual(plan['class_image_totals']['common'], 24)
        self.assertEqual(plan['class_totals']['common'], 48)
        self.assertEqual(plan['class_source_counts']['common'], 1)
        self.assertEqual(plan['class_group_counts']['common'], 24)
        for split in ('train', 'val', 'test'):
            members = [a for a in assets if plan['assignments'][a['id']] == split]
            actual = Counter(s['label'] for a in members for s in a['shapes'])
            presence = Counter(k for a in members for k in {s['label'] for s in a['shapes']})
            self.assertEqual(dict(actual), {k:v for k,v in plan['class_counts'][split].items() if v})
            self.assertEqual(dict(presence), {k:v for k,v in plan['class_image_counts'][split].items() if v})
            self.assertEqual(presence['rare'], 1)
        self.assertTrue(any(x['label'] == 'rare' for x in plan['scarcity']))
        self.assertEqual(sum(plan['image_counts'].values()), len(assets))
        for pair in plan['pair_distribution']:
            for split in ('train','val','test'):
                expected = sum(set(pair['labels']).issubset({s['label'] for s in a['shapes']})
                               for a in assets if plan['assignments'][a['id']] == split)
                self.assertEqual(pair['counts'][split], expected)

    def test_all_modes_keep_duplicates_locks_and_input_immutable(self):
        assets = data(); assets[1]['sha256'] = assets[0]['sha256']
        groups = source_groups(assets, strategy='class_balanced')
        locked = next(g for g in groups if g['asset_ids'] == ['10'])
        for mode in ('hybrid','presence','instances','cooccurrence'):
            with self.subTest(mode=mode):
                plan = smart_split(assets, self.options(balance_mode=mode, locks={locked['id']: 'test'}))
                self.assertEqual(plan['assignments']['0'], plan['assignments']['1'])
                self.assertEqual(plan['assignments']['10'], 'test')

    def test_source_isolation_and_impossible_rare_coverage(self):
        with self.assertRaisesRegex(ValueError, '群組'):
            smart_split(data(), self.options(source_isolation=True))
        assets = data()
        for a in assets:
            a['shapes'] = [{'label': 'common'}]
        assets[0]['shapes'].append({'label': 'unique'})
        plan = smart_split(assets, self.options())
        self.assertFalse(plan['ready'])
        self.assertTrue(any(b['label'] == 'unique' for b in plan['blockers']))

    def test_grouped_multilabel_and_zero_test(self):
        assets = data()
        for i,a in enumerate(assets): a['batch_id'] = str(i % 3)
        plan = smart_split(assets, self.options(source_isolation=True, ratios=[80,20,0]))
        self.assertEqual(plan['image_counts']['test'], 0)
        for batch in ('0','1','2'):
            self.assertEqual(len({plan['assignments'][a['id']] for a in assets if a['batch_id'] == batch}),1)

    def test_invalid_new_options(self):
        for options in (self.options(source_isolation='false'),self.options(balance_mode='random')):
            with self.assertRaises(ValueError): smart_split(data(), options)
