import csv
import unittest
import tempfile
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from dr_model.data import FOLDERS, load_records, load_rgb, split_records
from dr_model.metrics import binary_auc, metrics
from dr_model.model import InferenceModel, OrderedHead, RetinalModel, class_probabilities, ordinal_loss
from dr_model.train import fit_temperature


class PipelineTests(unittest.TestCase):
    def test_duplicate_groups_cannot_leak(self):
        rows = [dict(id=f'{g}-{i}-{j}', grade=g, group=f'{g}-{i}')
                for g in range(5) for i in range(20) for j in range(2)]
        splits = split_records(rows)
        sets = [{r['group'] for r in values} for values in splits.values()]
        for i in range(3):
            for j in range(i): self.assertFalse(sets[i] & sets[j])
        self.assertEqual(sum(map(len, splits.values())), len(rows))
        self.assertEqual(splits, split_records(rows))
        self.assertNotEqual(splits, split_records(rows, seed=43))

    def test_conflicting_duplicate_labels_are_all_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (root/'train.csv').open('w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['id_code', 'diagnosis'])
                for name, grade, color in [('a',0,'red'), ('b',1,'red'), ('c',2,'blue')]:
                    folder = root/'colored_images'/FOLDERS[grade]
                    folder.mkdir(parents=True)
                    Image.new('RGB',(8,8),color).save(folder/(name+'.png'))
                    writer.writerow([name,grade])
            rows, audit = load_records(root, return_audit=True)
            self.assertEqual([r['id'] for r in rows], ['c'])
            self.assertEqual(audit['source_images'], 3)
            self.assertEqual(len(audit['conflicting_groups_excluded']), 1)
            (root/'colored_images'/FOLDERS[2]/'c.png').unlink()
            with self.assertRaises(FileNotFoundError): load_records(root)

    def test_metrics_and_tied_auc(self):
        labels = np.arange(5)
        probs = np.array([[.9 if g > i else .1 for i in range(4)] for g in labels])
        report = metrics(labels, probs)
        self.assertEqual(report['qwk'], 1)
        self.assertEqual(report['sensitivity'], 1)
        self.assertEqual(report['specificity'], 1)
        self.assertEqual(binary_auc([0,1], [.5,.5]), .5)
        self.assertEqual(binary_auc([0,1], [1.,0.]), 0)
        self.assertIsNone(binary_auc([0,0], [.2,.3]))

    def test_metrics_can_compare_explicit_prediction_rules(self):
        labels = np.array([0, 1, 2, 3, 4])
        cumulative = np.array([[.8, .6, .4, .2]] * 5)
        default = metrics(labels, cumulative)
        explicit = metrics(labels, cumulative, predictions=labels)
        self.assertEqual(explicit['accuracy'], 1.0)
        self.assertEqual(explicit['macro_f1'], 1.0)
        self.assertEqual(explicit['qwk'], 1.0)
        self.assertEqual(explicit['mean_absolute_grade_error'], 0.0)
        self.assertEqual(default['accuracy'], .2)
        self.assertEqual(default['threshold_auc'], explicit['threshold_auc'])
        with self.assertRaises(ValueError):
            metrics(labels, cumulative, predictions=np.array([5]*5))

    def test_ordered_head_after_optimizer_updates(self):
        head = OrderedHead(8)
        optimizer = torch.optim.Adam(head.parameters(), lr=.1)
        x, labels = torch.randn(20,8), torch.arange(20) % 5
        for _ in range(5):
            optimizer.zero_grad()
            logits = head(x)
            self.assertTrue(torch.all(logits[:,:-1] > logits[:,1:]))
            loss = ordinal_loss(logits, labels)
            loss.backward()
            optimizer.step()
        probabilities = class_probabilities(head(x).sigmoid())
        self.assertTrue(torch.all(probabilities >= 0))
        torch.testing.assert_close(probabilities.sum(1), torch.ones(20))

    def test_rgb_resize_does_not_swap_red_blue(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'red.png'
            Image.new('RGB', (9,7), (255,0,0)).save(path)
            rgb = np.array(load_rgb(path, 224))
            self.assertEqual(rgb.shape, (224,224,3))
            np.testing.assert_array_equal(rgb[0,0], [255,0,0])

    def test_calibration_is_positive_and_does_not_change_grades(self):
        logits = torch.randn(30,4)
        labels = torch.arange(30) % 5
        temperature, report = fit_temperature(logits, labels)
        self.assertGreater(temperature, 0)
        self.assertLessEqual(report['after_bce'], report['before_bce'] + 1e-6)
        torch.testing.assert_close((logits.sigmoid()>.5).sum(1), ((logits/temperature).sigmoid()>.5).sum(1))

    def test_mobile_forward_and_trace_roundtrip(self):
        torch.set_num_threads(2)
        model = InferenceModel(RetinalModel(pretrained=False).eval()).eval()
        x = torch.rand(1,224,224,3) * 255
        with torch.no_grad():
            expected = model(x)
            traced = torch.jit.trace(model, x)
            with tempfile.TemporaryDirectory() as directory:
                path = str(Path(directory)/'model.pt')
                traced.save(path)
                actual = torch.jit.load(path)(x)
        self.assertEqual(tuple(actual.shape), (1,4))
        torch.testing.assert_close(actual, expected)
        self.assertTrue(torch.all(actual[:,:-1] >= actual[:,1:]))

    def check_full_model_training_step(self, device):
        torch.set_num_threads(2)
        model = RetinalModel(pretrained=False).to(device).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        image = torch.rand(2, 224, 224, 3, device=device) * 255
        labels = torch.tensor([0, 4], device=device)
        layouts = []
        handle = model.features.register_forward_pre_hook(
            lambda module, args: layouts.append(args[0].is_contiguous())
        )
        try:
            loss = ordinal_loss(model(image), labels)
            loss.backward()
            self.assertEqual(layouts, [True])
            self.assertTrue(torch.isfinite(loss).item())
            for parameter in model.parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all().item())
            before = model.head.score.weight.detach().clone()
            optimizer.step()
            self.assertFalse(torch.equal(before, model.head.score.weight))
        finally:
            handle.remove()

    def test_full_model_training_step_cpu(self):
        self.check_full_model_training_step('cpu')

    @unittest.skipUnless(torch.backends.mps.is_available(), 'MPS unavailable on this host')
    def test_full_model_training_step_mps(self):
        self.check_full_model_training_step('mps')

if __name__ == '__main__':
    unittest.main()
