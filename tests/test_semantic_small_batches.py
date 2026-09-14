import importlib.util
import unittest

from workbench.torchvision_engines import _semantic_training_mode


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("torchvision"),
                     "Requires the optional TorchVision training environment")
class SemanticSmallBatchTests(unittest.TestCase):
    def test_singleton_pooled_batch_trains_without_losing_affine_gradients(self):
        import torch
        from torchvision.models.segmentation.deeplabv3 import ASPPPooling

        model = ASPPPooling(16, 8)
        normalization = next(layer for layer in model.modules() if isinstance(layer, torch.nn.BatchNorm2d))
        initial_mean = normalization.running_mean.clone()
        _semantic_training_mode(model, torch, 1)
        inputs = torch.randn(1, 16, 4, 4)
        output = model(inputs)
        output.square().mean().backward()
        self.assertEqual(output.shape, (1, 8, 4, 4))
        self.assertFalse(normalization.training)
        self.assertIsNotNone(normalization.weight.grad)
        self.assertTrue(torch.equal(initial_mean, normalization.running_mean))

        _semantic_training_mode(model, torch, 2)
        self.assertTrue(normalization.training)
        self.assertEqual(model(torch.randn(2, 16, 4, 4)).shape, (2, 8, 4, 4))
        self.assertEqual(int(normalization.num_batches_tracked), 1)


if __name__ == "__main__":
    unittest.main()
