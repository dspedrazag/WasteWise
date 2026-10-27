"""Small correctness checks; no model construction, training or notebook execution."""

import ast
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src import experiment_data as data
from src import experiment_results as results


class MetricTests(unittest.TestCase):
    def test_hand_calculated_metrics_include_absent_class(self):
        # A mathematical fixture, not an experimental result.
        result = results.classification_metrics(np.array([0, 0, 1, 1]),
                                                 np.array([0, 1, 1, 1]), ['a', 'b', 'c'])
        self.assertEqual(result['confusion_matrix'], [[1, 1, 0], [0, 2, 0], [0, 0, 0]])
        self.assertEqual(result['accuracy'], 0.75)
        self.assertAlmostEqual(result['macro_f1'], (2 / 3 + 0.8) / 3)
        self.assertAlmostEqual(result['weighted_f1'], (2 / 3 + 0.8) / 2)
        self.assertEqual(result['per_class'][2]['f1'], 0)
        self.assertEqual(result['per_class'][2]['one_vs_rest_accuracy'], 1)

    def test_invalid_labels_rejected(self):
        for truth, predicted in [([], []), ([0], [2]), ([0], [-1]), ([0.5], [0]), ([0, 1], [0])]:
            with self.subTest(truth=truth, predicted=predicted), self.assertRaises(ValueError):
                results.classification_metrics(np.asarray(truth), np.asarray(predicted), ['a', 'b'])


class DataBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.population = data.load_experiment_data()

    def test_approved_membership_and_stable_order(self):
        population = self.population
        self.assertEqual(len(population['train']), 1765)
        self.assertEqual(len(population['validation']), 378)
        self.assertTrue(set(population['train']['path']).isdisjoint(population['validation']['path']))
        self.assertEqual(data.rows_sha256(population['validation']),
                         data.rows_sha256(population['validation'].sample(frac=1, random_state=42)))

    def test_test_image_cannot_be_opened_through_validation(self):
        split = pd.read_csv(self.population['root'] / 'manifests/split_manifest.csv')
        test_path = split.loc[split['subset'] == 'test', 'path'].iloc[0]
        with patch('src.experiment_data.Image.open') as image_open:
            with self.assertRaises(ValueError):
                data.open_image(self.population, test_path, 'validation')
            with self.assertRaises(ValueError):
                data.open_image(self.population, test_path, 'test')
            image_open.assert_not_called()

    def test_path_traversal_rejected_before_image_access(self):
        with patch('src.experiment_data.Image.open') as image_open:
            with self.assertRaises(ValueError):
                data.open_image(self.population, '../dataset-resized.zip', 'train')
            image_open.assert_not_called()


class RunBoundaryTests(unittest.TestCase):
    def test_existing_run_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as root:
            directory = results.require_new_run(root, 'resnet18', 'baseline')
            directory.mkdir(parents=True)
            with self.assertRaises(FileExistsError):
                results.require_new_run(root, 'resnet18', 'baseline')
            with self.assertRaises(ValueError):
                results.run_path(root, 'resnet18', '../escape')

    def test_empty_comparison_cannot_select_candidate(self):
        self.assertTrue(results.comparison_table([]).empty)
        self.assertIn('pending', results.selection_evidence([]))

    def test_mismatched_protocol_cannot_be_ranked(self):
        config = dict(results.BASELINE, model_name='resnet18', epochs=10)
        with self.assertRaises(ValueError):
            results.comparison_table([{'config': config}])

    def test_notebooks_are_unexecuted_and_training_is_opt_in(self):
        root = Path(__file__).resolve().parents[1]
        names = ['01_mobilenetv3.ipynb', '02_resnet18.ipynb',
                 '03_efficientnetb0.ipynb', '04_model_comparison.ipynb']
        for name in names:
            notebook = json.loads((root / name).read_text())
            self.assertEqual(notebook['metadata']['kernelspec']['name'], 'scientific')
            mode = None
            train_calls = 0
            for cell in notebook['cells']:
                if cell['cell_type'] != 'code':
                    continue
                self.assertIsNone(cell['execution_count'])
                self.assertEqual(cell['outputs'], [])
                tree = ast.parse(''.join(cell['source']))
                for node in tree.body:
                    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'MODE' for t in node.targets):
                        mode = ast.literal_eval(node.value)
                    if isinstance(node, ast.If) and ast.unparse(node.test) == "MODE == 'train'":
                        train_calls += sum(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                                           and n.func.attr == 'train_baseline' for n in ast.walk(ast.Module(body=node.body, type_ignores=[])))
            if name != '04_model_comparison.ipynb':
                self.assertEqual(mode, 'review')
                self.assertEqual(train_calls, 1)


if __name__ == '__main__':
    unittest.main()
