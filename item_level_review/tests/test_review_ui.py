"""Focused tests. All saves and pictures use temporary synthetic records."""

from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import patch
import ast
import contextlib
import copy
import csv
import fcntl
import hashlib
import io
import json
import os
import sys
import tempfile
import unittest

from PIL import Image, ImageDraw
import nbformat

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from item_level_review import review_ui as module


def encode_rows(rows, fields):
    output = io.StringIO(newline='')
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode('utf-8')


def make_backend(folder):
    """Use the notebook's actual declarations with no real setup/image access."""
    folder = Path(folder)
    folder.mkdir(exist_ok=True)
    nb = nbformat.read(ROOT / 'item_level_review/item_review.ipynb', as_version=4)
    namespace = dict(Path=Path, csv=csv, io=io, hashlib=hashlib, datetime=datetime,
                     timezone=timezone, commit_manifest=module.commit_manifest)
    constants = {'SOURCE_LABELS', 'MATERIAL_LABELS', 'SCOPE_STATUSES', 'ITEM_CRITERIA',
                 'ANNOTATION_FIELDS', 'STATUSES', 'FIELDS'}
    functions = []
    for index in (2, 7):
        for node in ast.parse(nb.cells[index].source).body:
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                if name in constants:
                    namespace[name] = ast.literal_eval(node.value)
            elif isinstance(node, ast.FunctionDef):
                functions.append(node)
    exec(compile(ast.Module(body=functions, type_ignores=[]), '<review declarations>', 'exec'), namespace)
    base = {}
    rows = []
    assignments = [('plastic', 'train'), ('plastic', 'test'), ('metal', 'validation'),
                   ('metal', 'train'), ('cardboard', 'train'), ('cardboard', '')]
    for i, (label, subset) in enumerate(assignments, 1):
        path = f'{label}/SYNTHETIC_{i}.png'
        base[path] = dict(path=path, original_label=label, subset=subset,
                          group_id=f'fixture_{i}' if subset else '', excluded=str(not subset).lower())
        row = dict.fromkeys(namespace['FIELDS'], '')
        row.update(base[path], review_status='unreviewed')
        rows.append(row)
        destination = folder / 'images' / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        image = Image.new('RGB', (512, 384), '#edf1f5')
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle((160, 65, 350, 300), radius=18, fill='#637f9b')
        draw.text((30, 20), f'SYNTHETIC TEST IMAGE {i} — {label}', fill='#111111')
        draw.text((30, 350), 'No TrashNet image or real annotation is used.', fill='#111111')
        image.save(destination)
    source = folder / 'source_metadata.json'
    source.write_text(json.dumps(base))
    manifest = folder / 'synthetic_review.csv'
    manifest.write_bytes(encode_rows(rows, namespace['FIELDS']))
    namespace.update(BASE_ROWS=base, REVIEW_FILE=manifest, HISTORY_FILE=folder / 'corrections.jsonl',
                     RECORD_DIGESTS={source: hashlib.sha256(source.read_bytes()).hexdigest()})
    namespace['original_image'] = lambda path: folder / 'images' / path
    return namespace


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='wastewise-ui-test-')
        self.folder = Path(self.temp.name)
        self.ns = make_backend(self.folder)
        self.manifest = self.ns['REVIEW_FILE']
        self.history = self.ns['HISTORY_FILE']
        self.paths = list(self.ns['BASE_ROWS'])
        self.ui = module.ReviewUI(self.ns['load_review'], self.ns['save_decisions'],
                                  self.ns['original_image'], self.ns['ITEM_CRITERIA'], self.ns['MATERIAL_LABELS'])
        self.ui.reviewer.value = 'Synthetic test reviewer'

    def tearDown(self):
        self.ui.close()
        self.temp.cleanup()

    def rows(self):
        return self.ns['load_review']()

    def decision(self, material='plastic', item='plastic_beverage_bottle', scope='supported', reason=''):
        return dict(reviewed_material_label=material, item_type=item, scope_status=scope, reason=reason)

    def save(self, path, decision=None, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return self.ns['save_decisions'](
                {path: decision or self.decision()}, reviewer='Synthetic test reviewer',
                status='confirmed', confirmed_by_user=True, **kwargs)

    def fill(self, material='plastic', item='plastic_beverage_bottle', notes=''):
        self.ui.material.value = material
        self.ui.item.value = item
        self.ui.notes.value = notes

    def events(self):
        return [json.loads(line) for line in self.history.read_text().splitlines()]

    def test_initial_category_and_navigation_do_not_write(self):
        before = self.manifest.read_bytes()
        self.assertEqual(self.ui.current_path, self.paths[0])
        self.ui.skip_button.click()
        self.assertEqual(self.ui.current_path, self.paths[1])
        self.ui.previous_button.click()
        self.assertEqual(self.ui.current_path, self.paths[0])
        self.ui.category.value = 'metal'
        self.assertEqual(self.ui.current_path, self.paths[2])
        self.ui.category.value = 'cardboard'
        self.assertEqual(self.ui.current_path, self.paths[4])
        self.ui.enlarge.value = True
        self.assertEqual(self.ui.photo.layout.width, '1024px')
        self.assertEqual(self.manifest.read_bytes(), before)
        self.assertFalse(self.history.exists())

    def test_save_updates_overall_and_category_progress(self):
        self.fill()
        self.ui.save_button.click()
        self.assertEqual(self.rows()[0]['review_status'], 'confirmed')
        self.assertEqual(self.rows()[0]['reason'], '')
        self.assertEqual(self.ui.overall_bar.value, 1)
        self.assertEqual(self.ui.category_bar.value, 1)
        self.assertIn('1 / 6 confirmed; 5 remaining', self.ui.progress.value)
        self.assertEqual(self.ui.current_path, self.paths[1])
        self.assertFalse(self.history.exists())
        for row in self.rows():
            for key, value in self.ns['BASE_ROWS'][row['path']].items():
                self.assertEqual(row[key], value)

    def test_ambiguous_notes_required_in_ui_and_backend(self):
        before = self.manifest.read_bytes()
        self.fill('', 'unknown_object', '   ')
        self.ui.save_button.click()
        self.assertIn('short explanation', self.ui.message.value)
        self.assertEqual(self.manifest.read_bytes(), before)
        with self.assertRaisesRegex(ValueError, 'short explanation'):
            self.save(self.paths[0], self.decision('', 'unknown_object', 'ambiguous'))
        self.ui.notes.value = 'The synthetic object cannot be identified.'
        self.ui.save_button.click()
        self.assertEqual(self.rows()[0]['scope_status'], 'ambiguous')

    def test_unsaved_navigation_and_filter_changes_require_discard(self):
        before = self.manifest.read_bytes()
        self.fill()
        self.ui.next_button.click()
        self.assertEqual(self.ui.current_path, self.paths[0])
        self.assertEqual(self.ui.discard_box.layout.display, '')
        self.ui.keep_button.click()
        self.assertEqual(self.ui.item.value, 'plastic_beverage_bottle')
        self.ui.category.value = 'metal'
        self.assertEqual(self.ui.category.value, 'plastic')
        self.ui.discard_button.click()
        self.assertEqual(self.ui.category.value, 'metal')
        self.assertEqual(self.ui.current_path, self.paths[2])
        self.assertEqual(self.manifest.read_bytes(), before)

    def test_resume_and_category_complete_view(self):
        self.save(self.paths[0])
        self.save(self.paths[1])
        self.ui._refresh()
        self.assertIsNone(self.ui.current_path)
        self.assertTrue(self.ui.save_button.disabled)
        self.ui.view.value = 'confirmed'
        self.assertEqual(len(self.ui._paths), 2)
        self.assertTrue(self.ui.material.disabled)
        self.assertTrue(self.ui.save_button.disabled)
        self.ui.category.value = 'metal'
        self.ui.view.value = 'remaining'
        self.assertEqual(self.ui.current_path, self.paths[2])
        resumed = module.ReviewUI(self.ns['load_review'], self.ns['save_decisions'],
                                  self.ns['original_image'], self.ns['ITEM_CRITERIA'], self.ns['MATERIAL_LABELS'])
        try:
            self.assertEqual(resumed.overall_bar.value, 2)
            self.assertIsNone(resumed.current_path)
            resumed.category.value = 'metal'
            self.assertEqual(resumed.current_path, self.paths[2])
        finally:
            resumed.close()

    def test_explicit_correction_preview_and_history(self):
        self.save(self.paths[0])
        before = self.rows()[0].copy()
        self.ui.view.value = 'confirmed'
        self.ui.edit_button.click()
        self.ui.notes.value = 'Updated synthetic evidence.'
        self.assertTrue(self.ui.save_button.disabled)
        self.ui.preview_button.click()
        self.assertIn('Saved value', self.ui.comparison.value)
        self.assertTrue(self.ui.correction_button.disabled)
        self.ui.replace_confirm.value = True
        self.assertFalse(self.ui.correction_button.disabled)
        self.ui.notes.value = 'Changed again after preview.'
        self.assertTrue(self.ui.correction_button.disabled)
        self.assertFalse(self.ui.replace_confirm.value)
        self.ui.preview_button.click()
        self.ui.replace_confirm.value = True
        self.ui.correction_button.click()
        self.assertEqual(self.rows()[0]['reason'], 'Changed again after preview.')
        events = self.events()
        self.assertEqual([event['event'] for event in events], ['prepared', 'committed'])
        self.assertEqual(events[0]['before'], before)
        self.assertEqual(events[0]['after'], self.rows()[0])
        self.assertTrue(self.ui.material.disabled)

    def test_ordinary_save_cannot_replace_confirmed(self):
        self.save(self.paths[0])
        before = self.manifest.read_bytes()
        with self.assertRaisesRegex(ValueError, 'protected'):
            self.save(self.paths[0], self.decision(reason='Unapproved change'))
        with self.assertRaisesRegex(ValueError, 'displayed snapshot'):
            self.save(self.paths[0], correct_confirmed=True)
        self.assertEqual(self.manifest.read_bytes(), before)

    def test_stale_form_preserves_newer_annotation(self):
        self.fill()
        self.save(self.paths[0], self.decision(reason='Saved by another synthetic session'))
        before = self.manifest.read_bytes()
        self.ui.save_button.click()
        self.assertIn('changed since it was displayed', self.ui.message.value)
        self.assertEqual(self.manifest.read_bytes(), before)
        self.ui.reload_button.click()
        self.ui.discard_button.click()
        self.assertEqual(self.ui.current_path, self.paths[1])

    def test_test_images_manual_only_and_no_metadata_changes(self):
        before = self.manifest.read_bytes()
        with self.assertRaisesRegex(ValueError, 'manual annotation'):
            self.ns['save_decisions']({self.paths[1]: self.decision()}, reviewer='AI')
        self.assertEqual(self.manifest.read_bytes(), before)
        self.save(self.paths[1])
        row = self.rows()[1]
        self.assertEqual(row['subset'], 'test')
        self.assertEqual(row['group_id'], 'fixture_2')

    def test_exclusions_and_revised_material_preserved(self):
        self.save(self.paths[2])  # Synthetic metal-folder record reviewed as plastic.
        self.assertEqual(self.rows()[2]['original_label'], 'metal')
        self.assertEqual(self.rows()[2]['reviewed_material_label'], 'plastic')
        self.save(self.paths[5], self.decision('cardboard', 'pizza_box', 'supported'))
        row = self.rows()[5]
        self.assertEqual((row['excluded'], row['subset'], row['group_id']), ('true', '', ''))

    def test_invalid_batch_and_source_change_cannot_write(self):
        before = self.manifest.read_bytes()
        with self.assertRaisesRegex(ValueError, 'requires unsupported'):
            self.ns['save_decisions']({self.paths[0]: self.decision(),
                                       self.paths[2]: self.decision('metal', 'other_metal_object', 'supported')},
                                      reviewer='Synthetic', status='confirmed', confirmed_by_user=True)
        self.assertEqual(self.manifest.read_bytes(), before)
        source = next(iter(self.ns['RECORD_DIGESTS']))
        source.write_text('changed synthetic source')
        with self.assertRaisesRegex(ValueError, 'source record changed'):
            self.save(self.paths[0])
        self.assertEqual(self.manifest.read_bytes(), before)

    def test_stale_raw_and_row_order_rejected(self):
        before = self.manifest.read_bytes()
        self.save(self.paths[0])
        current = self.manifest.read_bytes()
        with self.assertRaisesRegex(RuntimeError, 'CSV changed'):
            self.ns['write_review'](self.rows(), before)
        with self.assertRaisesRegex(ValueError, 'row order'):
            self.ns['write_review'](list(reversed(self.rows())), current)
        self.assertEqual(self.manifest.read_bytes(), current)

    def test_failed_correction_write_is_recovered_without_replaying(self):
        self.save(self.paths[0])
        before = self.manifest.read_bytes()
        snapshot = self.rows()[0].copy()
        with patch.object(module.os, 'replace', side_effect=OSError('synthetic disk failure')):
            with self.assertRaisesRegex(OSError, 'synthetic disk failure'):
                self.save(self.paths[0], self.decision(reason='Attempted correction'),
                          expected_rows={self.paths[0]: snapshot}, correct_confirmed=True)
        self.assertEqual(self.manifest.read_bytes(), before)
        self.assertEqual([e['event'] for e in self.events()], ['prepared'])
        self.save(self.paths[2])
        self.assertEqual([e['event'] for e in self.events()], ['prepared', 'not_applied'])
        self.assertEqual(self.rows()[0], snapshot)
        self.assertFalse(list(self.folder.glob('.item-review-*.tmp')))

    def test_completion_failure_keeps_saved_correction_and_recovers(self):
        self.save(self.paths[0])
        snapshot = self.rows()[0].copy()
        append = module._append_history

        def fail_completion(path, event):
            if event['event'] == 'committed':
                raise OSError('synthetic log failure')
            return append(path, event)

        with patch.object(module, '_append_history', side_effect=fail_completion):
            notice = self.save(self.paths[0], self.decision(reason='Saved correction'),
                               expected_rows={self.paths[0]: snapshot}, correct_confirmed=True)
        self.assertIn('Saved.', notice)
        self.assertEqual(self.rows()[0]['reason'], 'Saved correction')
        self.save(self.paths[2])
        self.assertEqual([e['event'] for e in self.events()], ['prepared', 'committed'])
        self.assertTrue(self.events()[-1]['recovered'])

    def test_history_failure_or_concurrent_lock_blocks_save(self):
        before = self.manifest.read_bytes()
        self.history.write_text('invalid history\n')
        with self.assertRaisesRegex(ValueError, 'history needs inspection'):
            self.save(self.paths[0])
        self.assertEqual(self.manifest.read_bytes(), before)
        with (self.folder / '.review.lock').open('a+b') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(RuntimeError, 'Another review is saving'):
                self.save(self.paths[0])
        self.assertEqual(self.manifest.read_bytes(), before)

    def test_missing_image_disables_saving(self):
        self.ui.original_image = lambda _: self.folder / 'missing.png'
        self.ui._refresh()
        self.assertFalse(self.ui.photo.value)
        self.assertTrue(self.ui.save_button.disabled)
        self.assertIn('Image unavailable', self.ui.message.value)

    def test_notebook_format_and_syntax(self):
        nb = nbformat.read(ROOT / 'item_level_review/item_review.ipynb', as_version=4)
        nbformat.validate(nb)
        for index, cell in enumerate(nb.cells):
            if cell.cell_type == 'code':
                compile(cell.source, f'cell-{index}', 'exec')


if __name__ == '__main__':
    unittest.main()
