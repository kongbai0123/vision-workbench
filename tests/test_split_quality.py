import unittest

from workbench.split_quality import split_class_coverage


class SplitCoverageTests(unittest.TestCase):
    def test_project_and_immutable_assets_have_same_coverage(self):
        project = [{'id': 'a', 'split': 'train', 'shapes': [{'label': 'part'}, {'label': 'part'}]},
                   {'id': 'b', 'split': 'val', 'shapes': [{'label': 'rare'}]},
                   {'id': 'c', 'split': 'test', 'shapes': []}]
        manifest = [{('asset_id' if key == 'id' else key): value for key, value in asset.items()}
                    for asset in project]
        coverage = split_class_coverage(project)
        self.assertEqual(coverage, split_class_coverage(manifest))
        self.assertFalse(coverage['ready'])
        self.assertEqual(coverage['missing_train_classes'], ['rare'])
        self.assertEqual(coverage['class_counts']['train'], {'part': 2, 'rare': 0})
        self.assertEqual(coverage['class_totals'], {'part': 2, 'rare': 1})
        self.assertEqual(coverage['image_counts'], {'train': 1, 'val': 1, 'test': 1})
        self.assertEqual(coverage['blockers'][0]['code'], 'train_class_missing')

    def test_preview_assignments_override_current_split_without_mutation(self):
        assets = [{'asset_id': 'a', 'split': 'test', 'shapes': [{'label': 'part'}]},
                  {'asset_id': 'b', 'split': 'train', 'shapes': []}]
        coverage = split_class_coverage(assets, {'a': 'train', 'b': 'val'}, active_splits=('train', 'val'))
        self.assertFalse(coverage['ready'])
        self.assertEqual(coverage['missing_evaluation_classes'], {'val': ['part']})
        self.assertEqual(coverage['blockers'][0]['code'], 'validation_class_missing')
        self.assertEqual(coverage['warnings'], [])
        self.assertEqual(assets[0]['split'], 'test')

    def test_classes_on_unassigned_images_still_require_training_coverage(self):
        coverage = split_class_coverage([{'id': 'a', 'shapes': [{'label': 'part'}]}])
        self.assertEqual(coverage['missing_train_classes'], ['part'])
        self.assertEqual(coverage['image_counts'], {'train': 0, 'val': 0, 'test': 0})

    def test_absent_labels_and_empty_shapes_do_not_create_spurious_classes(self):
        coverage = split_class_coverage([{'id': 'a', 'split': 'train', 'shapes': [{}, {'label': ''}]},
                                         {'id': 'b', 'split': 'val', 'shapes': [{'label': 'part'}]}])
        self.assertEqual(coverage['class_totals'], {'part': 1})
        self.assertEqual(coverage['missing_train_classes'], ['part'])
