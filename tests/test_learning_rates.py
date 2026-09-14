from types import SimpleNamespace
import unittest
from workbench.learning_rates import LearningRateSchedule, measured_rates
from workbench.training_parameters import validate_config
from workbench.model_registry import MODELS


class LearningRateTests(unittest.TestCase):
    def schedule(self, **extra):
        opt=SimpleNamespace(param_groups=[{'lr':.01}])
        return LearningRateSchedule(opt,{'learning_rate':.01,'min_learning_rate':.001,'epochs':6,**extra})
    def test_cosine_warmup_starts_low_and_finishes_at_minimum(self):
        scheduler=self.schedule(scheduler='cosine',warmup_epochs=2)
        rates=[scheduler.start_epoch(e)['train/learning_rate'] for e in range(1,7)]
        self.assertEqual(rates[:2],[.005,.01]);self.assertAlmostEqual(rates[-1],.001)
        self.assertTrue(all(a>=b for a,b in zip(rates[2:],rates[3:])))
    def test_plateau_uses_completed_validation_for_next_epoch(self):
        scheduler=self.schedule(scheduler='plateau',lr_patience=1,lr_factor=.5)
        rates=[]
        for e,score in enumerate([.7,.7,.7,.8,.8,.8],1):
            rates.append(scheduler.start_epoch(e)['train/learning_rate']);scheduler.finish_epoch(score)
        self.assertEqual(rates,[.01,.01,.01,.005,.005,.005])
        self.assertEqual(scheduler.state_dict()['next_rate'],.0025)
    def test_fixed_and_linear(self):
        fixed=self.schedule(scheduler='fixed',warmup_epochs=2)
        self.assertEqual([fixed.start_epoch(e)['train/learning_rate'] for e in range(1,7)],[.01]*6)
        linear=self.schedule(scheduler='linear');self.assertAlmostEqual(linear.start_epoch(6)['train/learning_rate'],.001)
    def test_measurements_keep_distinct_optimizer_groups(self):
        rates=measured_rates(SimpleNamespace(param_groups=[{'lr':.001},{'lr':.002}]))
        self.assertEqual(rates,{'train/learning_rate':.001,'lr/group_1':.002})
        self.assertEqual(measured_rates(None),{})
    def test_validation_rejects_invalid_schedule_and_ranges(self):
        definition=next(m for m in MODELS if m['key']=='maskrcnn_resnet50_fpn')
        for options in ({'scheduler':'unknown'},{'scheduler':'cosine','epochs':2,'warmup_epochs':2},
                        {'min_learning_rate':.1,'learning_rate':.01},{'lr_factor':1}):
            with self.subTest(options=options),self.assertRaises(ValueError):validate_config(definition,options)
        ultra=next(m for m in MODELS if m['key']=='yolo26n_seg')
        with self.assertRaises(ValueError):validate_config(ultra,{'scheduler':'plateau'})
