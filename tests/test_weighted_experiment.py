"""Loss math and control checks only: no optimization, images or saved experiments."""

import ast
import copy
import json
from pathlib import Path
import unittest

import numpy as np
import torch

from src import experiment_data as datasets
from src import experiment_results as results
from src import experiment_training as training
from src import weighted_experiment as weighted


class WeightedExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]
        cls.data = datasets.load_experiment_data(cls.root)
        cls.baseline = results.load_run(cls.root, weighted.MODEL_NAME, weighted.BASELINE_RUN, cls.data)

    def controlled_fixture(self):
        # Configuration-only fixture; these booleans are not an experimental verification record.
        config = copy.deepcopy(self.baseline['config'])
        config.update(loss=weighted.WEIGHTED_LOSS, run_name=weighted.WEIGHTED_RUN,
                      loss_weighting=weighted.class_weight_plan(self.data),
                      baseline_reference=weighted.baseline_reference(self.baseline),
                      initialization_verification={key: True for key in [
                          'builder_matches_baseline', 'all_initial_tensors_equal',
                          'frozen_features_match_baseline_checkpoint', 'shuffle_indices_equal']})
        return config

    def test_weights_training_only_and_class_order(self):
        plan = weighted.class_weight_plan(self.data)
        self.assertEqual(plan['counts'], [282, 348, 287, 416, 336, 96])
        np.testing.assert_allclose(plan['weights'], 1765 / (6 * np.array(plan['counts'])), rtol=0, atol=1e-15)
        self.assertEqual(plan['classes'], self.data['classes'])
        changed_validation = {**self.data, 'validation': self.data['validation'].assign(label='trash')}
        self.assertEqual(plan, weighted.class_weight_plan(changed_validation))

    def test_weighted_mean_matches_hand_formula(self):
        # Mathematical logits, not simulated model predictions or measured results.
        config = self.controlled_fixture()
        logits = torch.tensor([[1., -1., 0., 0.5, 0.2, 0.1], [0., 1., 0., 0., 0., -1.],
                               [0.5, 0.3, 0., -0.1, 0.2, 2.]])
        labels = torch.tensor([0, 5, 5])
        criterion = training.training_criterion(config, torch.device('cpu'))
        probabilities = np.exp(logits.numpy()) / np.exp(logits.numpy()).sum(axis=1, keepdims=True)
        losses = -np.log(probabilities[np.arange(3), labels.numpy()])
        weights = np.array(config['loss_weighting']['weights'])[labels.numpy()]
        expected = (weights * losses).sum() / weights.sum()
        self.assertAlmostEqual(criterion(logits, labels).item(), expected, places=6)
        # Uneven batches must be combined using weight sums, not their sample counts.
        numerator, denominator = 0., 0.
        for indices in [slice(0, 1), slice(1, 3)]:
            batch_weight = criterion.weight[labels[indices]].sum().item()
            numerator += criterion(logits[indices], labels[indices]).item() * batch_weight
            denominator += batch_weight
        self.assertAlmostEqual(numerator / denominator, expected, places=6)

    def test_baseline_loss_stays_unweighted(self):
        criterion = training.training_criterion(self.baseline['config'], torch.device('cpu'))
        self.assertIsNone(criterion.weight)
        source = ast.parse((self.root / 'src/experiment_training.py').read_text())
        evaluation = next(node for node in source.body if isinstance(node, ast.FunctionDef)
                          and node.name == 'evaluate_validation')
        loss_calls = [node for node in ast.walk(evaluation) if isinstance(node, ast.Call)
                      and isinstance(node.func, ast.Attribute) and node.func.attr == 'CrossEntropyLoss']
        self.assertEqual(len(loss_calls), 1)
        self.assertEqual(loss_calls[0].args, [])
        self.assertEqual(loss_calls[0].keywords, [])

    def test_all_controlled_settings_are_fixed(self):
        config = self.controlled_fixture()
        self.assertTrue(weighted.validate_controlled_config(self.data, self.baseline, config))
        changes = {'seed': 43, 'learning_rate': 0.01, 'epochs': 6, 'batch_size': 16,
                   'feature_mode': 'train', 'device': 'cpu', 'augmentation': 'flip',
                   'validation_rows_sha256': 'changed', 'manifest_sha256': 'changed',
                   'checkpoint_selection': 'latest', 'weights': 'other'}
        for key, value in changes.items():
            with self.subTest(setting=key), self.assertRaises(ValueError):
                weighted.validate_controlled_config(self.data, self.baseline, {**config, key: value})

    def test_wrong_weights_and_baseline_reference_are_rejected(self):
        config = self.controlled_fixture()
        changed = copy.deepcopy(config)
        changed['loss_weighting']['weights'][0] += 0.01
        with self.assertRaises(ValueError):
            weighted.validate_controlled_config(self.data, self.baseline, changed)
        with self.assertRaises(ValueError):
            weighted.validate_controlled_config(self.data, self.baseline, {**config, 'baseline_reference': {}})

    def test_weighted_run_cannot_enter_three_baseline_ranking(self):
        with self.assertRaises(ValueError):
            results.comparison_table([{'config': self.controlled_fixture()}])
        with self.assertRaises(ValueError):
            training.train_baseline(None, None, None, None, self.controlled_fixture(), None)

    def test_saved_baseline_comparison_still_loads(self):
        runs = [results.load_run(self.root, name, weighted.BASELINE_RUN, self.data) for name in results.MODEL_NAMES]
        table = results.comparison_table(runs)
        self.assertEqual(len(table), 3)
        self.assertEqual(table.iloc[0]['Model'], 'EfficientNet-B0')

    def test_pair_comparison_checks_validation_membership(self):
        # Reuse real baseline predictions as a membership fixture, not a revised result.
        fixture = {**self.baseline, 'config': self.controlled_fixture(),
                   'predictions': self.baseline['predictions'].sample(frac=1, random_state=42).copy()}
        table = weighted.comparison_table(self.data, self.baseline, fixture)
        self.assertTrue(table['Validation images'].eq(378).all())
        fixture['predictions'].iloc[0, fixture['predictions'].columns.get_loc('path')] = 'not_an_approved_image.jpg'
        with self.assertRaises(ValueError):
            weighted.comparison_table(self.data, self.baseline, fixture)

    def test_new_notebook_training_is_opt_in_and_builder_unchanged(self):
        path = self.root / '05_efficientnetb0_weighted_loss.ipynb'
        notebook = json.loads(path.read_text())
        self.assertEqual(notebook['metadata']['kernelspec']['name'], 'scientific')
        mode, calls = None, []
        for cell in notebook['cells']:
            if cell['cell_type'] != 'code':
                continue
            tree = ast.parse(''.join(cell['source']))
            for node in tree.body:
                if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'MODE' for t in node.targets):
                    mode = ast.literal_eval(node.value)
                if isinstance(node, ast.If) and ast.unparse(node.test) == "MODE == 'train'":
                    calls.extend(n for n in ast.walk(ast.Module(body=node.body, type_ignores=[]))
                                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                                 and n.func.attr == 'train_weighted')
        self.assertIn(mode, {'review', 'train', 'reload'})
        self.assertEqual(len(calls), 1)
        self.assertEqual(weighted.builder_ast(path), weighted.builder_ast(self.root / '03_efficientnetb0.ipynb'))


if __name__ == '__main__':
    unittest.main()
