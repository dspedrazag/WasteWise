"""Notebook controls and safe persistence for the manual item-review workflow."""

from contextlib import redirect_stdout
from datetime import datetime, timezone
from html import escape
from pathlib import Path
import fcntl
import hashlib
import io
import json
import os
import re
import tempfile
import uuid

import ipywidgets as widgets
from IPython.display import display


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _append_history(path, event):
    if path.is_symlink():
        raise ValueError('The correction history must not be a symlink.')
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(event, ensure_ascii=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def _recover_history(path, current_raw):
    """Resolve a prepared correction by checksum; never replay an annotation."""
    if path.is_symlink():
        raise ValueError('The correction history must not be a symlink.')
    if not path.exists():
        return
    pending = {}
    try:
        for line in path.read_text(encoding='utf-8').splitlines():
            event = json.loads(line)
            transaction = event['transaction_id']
            if event['event'] == 'prepared':
                if pending:
                    raise ValueError('Multiple unfinished corrections in history.')
                pending[transaction] = event
            elif event['event'] in ('committed', 'not_applied'):
                if transaction not in pending:
                    raise ValueError('Correction history has an unmatched completion.')
                pending.pop(transaction)
            else:
                raise ValueError('Unknown correction-history event.')
    except (ValueError, KeyError, TypeError) as error:
        raise ValueError('Correction history needs inspection; no annotations were changed.') from error
    for transaction, event in pending.items():
        digest = _digest(current_raw)
        if digest == event['after_sha256']:
            outcome = 'committed'
        elif digest == event['before_sha256']:
            outcome = 'not_applied'
        else:
            raise RuntimeError('An interrupted correction does not match the CSV; stop and inspect the history.')
        _append_history(path, dict(event=outcome, transaction_id=transaction,
                                  recovered=True, recorded_at=datetime.now(timezone.utc).isoformat()))


def commit_manifest(review_file, payload, expected_raw, history_file, correction=None):
    """Replace the CSV under a shared lock and retain before/after corrections.

    A prepared event is durable before the CSV changes. A completion event marks
    the outcome. If interrupted, the next save resolves it from exact checksums.
    The CSV is always the sole current annotation record.
    """
    review_file, history_file = Path(review_file), Path(history_file)
    lock_path = history_file.parent / '.review.lock'
    if lock_path.is_symlink():
        raise ValueError('The review lock must not be a symlink.')
    with lock_path.open('a+b') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Another review is saving. Try again after it finishes.') from error
        if review_file.is_symlink() or review_file.read_bytes() != expected_raw:
            raise RuntimeError('The CSV changed while saving. Reload before trying again.')
        _recover_history(history_file, expected_raw)
        temporary = None
        transaction = None
        try:
            with tempfile.NamedTemporaryFile(dir=review_file.parent, prefix='.item-review-',
                                             suffix='.tmp', delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            if correction is not None:
                transaction = str(uuid.uuid4())
                _append_history(history_file, dict(
                    event='prepared', transaction_id=transaction,
                    recorded_at=datetime.now(timezone.utc).isoformat(),
                    before_sha256=_digest(expected_raw), after_sha256=_digest(payload),
                    before=correction['before'], after=correction['after']))
            # Also detect a writer that does not use the lock (e.g. an old kernel).
            if review_file.is_symlink() or review_file.read_bytes() != expected_raw:
                raise RuntimeError('The CSV changed while saving. Reload before trying again.')
            os.replace(temporary, review_file)
            temporary = None
            if transaction:
                try:
                    _append_history(history_file, dict(event='committed', transaction_id=transaction,
                                                      recorded_at=datetime.now(timezone.utc).isoformat()))
                except OSError:
                    return 'Saved. History completion is pending and will be checked before the next save.'
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
        return ''


def _natural_key(path):
    return [int(part) if part.isdigit() else part for part in re.split(r'(\d+)', path)]


class ReviewUI:
    """One-image manual review; all annotation writes go through notebook checks."""

    def __init__(self, load_review, save_decisions, original_image, item_criteria, material_labels):
        self.load_review = load_review
        self.save_decisions = save_decisions
        self.original_image = original_image
        self.item_criteria = item_criteria
        self._loading = False
        self._current = None
        self._paths = []
        self._editing = False
        self._image_loaded = False
        self._pending_action = None
        self._active_category = 'plastic'
        self._active_view = 'remaining'
        self._baseline = None

        wide = {'description_width': '140px'}
        self.category = widgets.Dropdown(options=[('Plastic', 'plastic'), ('Metal', 'metal'),
                                                   ('Cardboard', 'cardboard')], description='Original category:',
                                         style=wide, layout=widgets.Layout(width='340px', max_width='100%'))
        self.view = widgets.Dropdown(options=[('Remaining', 'remaining'), ('All', 'all'),
                                               ('Confirmed', 'confirmed')], description='Show:')
        self.picker = widgets.Dropdown(options=[], description='Image:',
                                       layout=widgets.Layout(width='650px', max_width='100%'))
        self.reviewer = widgets.Text(description='Reviewer:', placeholder='Your name', style=wide)
        self.progress = widgets.HTML()
        self.overall_bar = widgets.IntProgress(description='Overall:', min=0, max=1)
        self.category_bar = widgets.IntProgress(description='Category:', min=0, max=1)
        self.header = widgets.HTML()
        self.photo = widgets.Image(format='jpeg', layout=widgets.Layout(width='640px', max_width='100%'))
        self.enlarge = widgets.ToggleButton(description='Enlarge image', icon='search-plus')
        self.material = widgets.Dropdown(
            options=[('Choose material…', None), ('Cannot verify material', '')]
                    + [(label.title(), label) for label in material_labels],
            description='Reviewed material:', style=wide, layout=widgets.Layout(width='480px', max_width='100%'))
        self.item = widgets.Dropdown(
            options=[('Choose item type…', None)] + [(key.replace('_', ' ').capitalize(), key) for key in item_criteria],
            description='Item type:', style=wide, layout=widgets.Layout(width='600px', max_width='100%'))
        self.scope = widgets.Dropdown(options=[('Choose scope…', None), ('Supported', 'supported'),
                                               ('Unsupported', 'unsupported'), ('Ambiguous', 'ambiguous')],
                                      description='Scope:', style=wide,
                                      layout=widgets.Layout(width='480px', max_width='100%'))
        self.hint = widgets.HTML()
        self.notes = widgets.Textarea(description='Review notes:', style=wide,
                                      placeholder='Required for ambiguous scope; optional otherwise.',
                                      layout=widgets.Layout(width='650px', max_width='100%', height='90px'))
        self.save_button = widgets.Button(description='Save', button_style='success', icon='check',
                                         tooltip='Confirm this annotation, save it, and advance.')
        self.skip_button = widgets.Button(description='Skip', tooltip='Leave the image pending and advance.')
        self.previous_button = widgets.Button(description='Previous', icon='arrow-left')
        self.next_button = widgets.Button(description='Next', icon='arrow-right')
        self.reload_button = widgets.Button(description='Reload saved record', icon='refresh')
        self.edit_button = widgets.Button(description='Edit confirmed', icon='pencil')
        self.cancel_edit_button = widgets.Button(description='Cancel edit')
        self.preview_button = widgets.Button(description='Preview correction')
        self.comparison = widgets.HTML()
        self.replace_confirm = widgets.Checkbox(description='I authorize this correction.',
                                                indent=False, layout=widgets.Layout(width='auto'))
        self.correction_button = widgets.Button(description='Save correction', button_style='warning')
        self.correction_box = widgets.VBox([widgets.HBox([self.preview_button, self.cancel_edit_button]),
                                            self.comparison, self.replace_confirm, self.correction_button])
        self.discard_button = widgets.Button(description='Discard edits and continue', button_style='warning')
        self.keep_button = widgets.Button(description='Keep editing')
        self.discard_box = widgets.VBox([widgets.HTML('<b>You have unsaved edits.</b>'),
                                        widgets.HBox([self.discard_button, self.keep_button])])
        self.message = widgets.HTML()
        self.widget = widgets.VBox([
            widgets.HTML('<h3>Manual item-level review</h3><p>Read the notebook criteria first. '
                         'Test images are for annotation only. No model predictions are displayed.</p>'),
            widgets.HBox([self.category, self.view]), self.progress,
            widgets.HBox([self.overall_bar, self.category_bar]), self.picker, self.header,
            self.photo, self.enlarge, self.reviewer, self.material, self.item, self.scope,
            self.hint, self.notes,
            widgets.HBox([self.previous_button, self.next_button, self.skip_button, self.save_button]),
            widgets.HBox([self.edit_button, self.reload_button]), self.correction_box,
            self.discard_box, self.message])

        # Keep controls usable in a narrow Jupyter pane as well as a full window.
        for box in (self.widget, self.correction_box, self.discard_box):
            box.layout.width = '100%'
            for child in box.children:
                if isinstance(child, widgets.HBox):
                    child.layout.flex_flow = 'row wrap'
                    child.layout.width = '100%'

        self.category.observe(self._filter_changed, names='value')
        self.view.observe(self._filter_changed, names='value')
        self.picker.observe(self._picker_changed, names='value')
        self.item.observe(self._item_changed, names='value')
        for control in (self.material, self.scope, self.notes, self.reviewer):
            control.observe(self._form_changed, names='value')
        self.enlarge.observe(self._resize, names='value')
        self.replace_confirm.observe(self._authorize_correction, names='value')
        self.save_button.on_click(lambda _: self._save(False))
        self.correction_button.on_click(lambda _: self._save(True))
        self.previous_button.on_click(lambda _: self._navigate(-1))
        self.next_button.on_click(lambda _: self._navigate(1))
        self.skip_button.on_click(lambda _: self._navigate(1))
        self.edit_button.on_click(self._edit_confirmed)
        self.cancel_edit_button.on_click(lambda _: self._request(lambda: self._refresh(self.current_path)))
        self.preview_button.on_click(self._preview_correction)
        self.reload_button.on_click(lambda _: self._request(lambda: self._refresh(self.current_path)))
        self.discard_button.on_click(self._discard)
        self.keep_button.on_click(lambda _: self._cancel_navigation())
        self._refresh()

    @property
    def current_path(self):
        return self._current['path'] if self._current else None

    def show(self):
        display(self.widget)
        return self

    def close(self):
        """Disable an old panel when its launch cell is rerun."""
        for control in (self.save_button, self.correction_button, self.edit_button,
                        self.next_button, self.previous_button, self.skip_button):
            control.disabled = True
        self.widget.close()

    def _notify(self, text, error=False):
        color = '#a12622' if error else '#165a36'
        self.message.value = f'<p style="color:{color}">{escape(text)}</p>'

    def _form(self):
        return dict(reviewed_material_label=self.material.value, item_type=self.item.value,
                    scope_status=self.scope.value, reason=self.notes.value)

    def _dirty(self):
        return self._current is not None and self._form() != self._baseline

    def _cancel_navigation(self):
        self._pending_action = None
        self.discard_box.layout.display = 'none'

    def _request(self, action):
        if self._dirty():
            self._pending_action = action
            self.discard_box.layout.display = ''
            return
        self._cancel_navigation()
        try:
            action()
        except Exception as error:
            self._notify(str(error), error=True)

    def _discard(self, _):
        action = self._pending_action
        self._cancel_navigation()
        if action:
            try:
                action()
            except Exception as error:
                self._notify(str(error), error=True)

    def _filter_changed(self, _):
        if self._loading:
            return
        category, view = self.category.value, self.view.value
        self._loading = True
        self.category.value, self.view.value = self._active_category, self._active_view
        self._loading = False

        def apply():
            self._active_category, self._active_view = category, view
            self._refresh()
        self._request(apply)

    def _picker_changed(self, change):
        if self._loading or change['new'] is None:
            return
        target = change['new']
        self._loading = True
        self.picker.value = self.current_path
        self._loading = False
        self._request(lambda: self._refresh(target))

    def _navigate(self, step):
        if not self._paths or self.current_path not in self._paths:
            return
        index = self._paths.index(self.current_path)
        target = self._paths[(index + step) % len(self._paths)]
        self._request(lambda: self._refresh(target))

    def _update_progress(self, rows):
        confirmed = sum(row['review_status'] == 'confirmed' for row in rows)
        self.overall_bar.max = max(1, len(rows))
        self.overall_bar.value = confirmed
        lines = [f'<b>Overall:</b> {confirmed} / {len(rows)} confirmed; {len(rows)-confirmed} remaining.',
                 '<table><tr><th style="text-align:left;padding-right:20px">Original category</th>'
                 '<th style="padding-right:20px">Confirmed / total</th><th>Remaining</th></tr>']
        for category in ('plastic', 'metal', 'cardboard'):
            subset = [row for row in rows if row['original_label'] == category]
            count = sum(row['review_status'] == 'confirmed' for row in subset)
            lines.append(f'<tr><td>{category.title()}</td><td>{count} / {len(subset)}</td>'
                         f'<td>{len(subset)-count}</td></tr>')
            if category == self._active_category:
                self.category_bar.max = max(1, len(subset))
                self.category_bar.value = count
        lines.append('</table><small>Remaining includes unreviewed images and unconfirmed proposals. '
                     'Excluded images stay in the annotation count.</small>')
        self.progress.value = ''.join(lines)

    def _refresh(self, path=None):
        rows = self.load_review()
        self._update_progress(rows)
        filtered = [row for row in rows if row['original_label'] == self._active_category
                    and (self._active_view == 'all'
                         or (row['review_status'] == 'confirmed') == (self._active_view == 'confirmed'))]
        filtered.sort(key=lambda row: _natural_key(row['path']))
        self._paths = [row['path'] for row in filtered]
        by_path = {row['path']: row for row in filtered}
        path = path if path in by_path else (self._paths[0] if self._paths else None)
        self._current = dict(by_path[path]) if path else None
        self._editing = False
        self._image_loaded = False
        self._cancel_navigation()
        self._loading = True
        try:
            self.category.value, self.view.value = self._active_category, self._active_view
            self.picker.options = [(Path(p).name, p) for p in self._paths]
            self.picker.value = path
            self.photo.value = b''
            self.message.value = ''
            if self._current:
                row = self._current
                self.header.value = (f'<b>{escape(row["path"])}</b><br>Original material: '
                                     f'{escape(row["original_label"])} · Subset: {escape(row["subset"] or "not assigned")} '
                                     f'· Excluded: {escape(row["excluded"])} · Status: {escape(row["review_status"])}')
                reviewed = row['review_status'] != 'unreviewed'
                self.material.value = row['reviewed_material_label'] if reviewed else None
                self.item.value = row['item_type'] if reviewed else None
                self.scope.value = row['scope_status'] if reviewed else None
                self.notes.value = row['reason']
                try:
                    source = Path(self.original_image(path))
                    self.photo.format = 'png' if source.suffix.lower() == '.png' else 'jpeg'
                    self.photo.value = source.read_bytes()
                    self._image_loaded = True
                except (OSError, ValueError) as error:
                    self._notify(f'Image unavailable; saving is disabled. {error}', error=True)
            else:
                self.header.value = '<b>No images in this category/view.</b> Choose All or Confirmed to revisit saved reviews.'
                self.material.value = self.item.value = self.scope.value = None
                self.notes.value = ''
            self._baseline = self._form()
            self._invalidate_preview()
        finally:
            self._loading = False
        self._set_controls()
        self._show_hint()

    def _set_controls(self):
        confirmed = bool(self._current and self._current['review_status'] == 'confirmed')
        editable = self._image_loaded and (not confirmed or self._editing)
        for control in (self.material, self.item, self.scope, self.notes):
            control.disabled = not editable
        self.save_button.disabled = not editable or confirmed
        self.edit_button.disabled = not (confirmed and self._image_loaded and not self._editing)
        self.correction_box.layout.display = '' if self._editing else 'none'
        self.preview_button.disabled = not self._editing
        self.skip_button.disabled = not self._current or confirmed
        self.previous_button.disabled = self.next_button.disabled = not self._paths
        self.picker.disabled = not self._paths
        self.enlarge.disabled = not self._image_loaded

    def _show_hint(self):
        item = self.item.value
        if item in self.item_criteria:
            material, scope = self.item_criteria[item]
            text = f'Established criterion: {scope} scope.'
            if material:
                text += f' Requires independently verified {material}.'
            self.hint.value = escape(text)
        else:
            self.hint.value = 'Select a type using the notebook criteria. Do not infer it from the source folder.'

    def _item_changed(self, _):
        if self._loading:
            return
        # Scope follows the fixed definition, never a model prediction.
        if self.item.value in self.item_criteria:
            self.scope.value = self.item_criteria[self.item.value][1]
        self._form_changed(None)
        self._show_hint()

    def _form_changed(self, _):
        if not self._loading:
            self._invalidate_preview()

    def _invalidate_preview(self):
        self.comparison.value = ''
        self.replace_confirm.value = False
        self.replace_confirm.disabled = True
        self.correction_button.disabled = True

    def _resize(self, change):
        self.photo.layout.width = '1024px' if change['new'] else '640px'

    def _validated_decision(self):
        if not self.reviewer.value.strip():
            raise ValueError('Enter your reviewer name before saving.')
        decision = self._form()
        if any(decision[key] is None for key in ('reviewed_material_label', 'item_type', 'scope_status')):
            raise ValueError('Choose reviewed material, item type, and scope. Use Cannot verify material when needed.')
        material, scope = self.item_criteria[decision['item_type']]
        if material and decision['reviewed_material_label'] != material:
            raise ValueError(f'This item requires verified {material}. Choose a less specific item if uncertain.')
        if decision['scope_status'] != scope:
            raise ValueError(f'This item uses {scope} scope under the established criteria.')
        if decision['scope_status'] == 'ambiguous' and not decision['reason'].strip():
            raise ValueError('Add a short explanation for ambiguous scope.')
        return decision

    def _edit_confirmed(self, _):
        if self.edit_button.disabled:
            return
        self._editing = True
        self._set_controls()
        self._notify('Edit the annotation, then preview and explicitly confirm the correction.')

    def _preview_correction(self, _):
        try:
            decision = self._validated_decision()
            if not self._editing or not self._dirty():
                raise ValueError('Change an annotation field before previewing a correction.')
            lines = ['<table><tr><th>Field</th><th>Saved value</th><th>Proposed value</th></tr>']
            for key, value in decision.items():
                lines.append(f'<tr><td style="padding:6px">{escape(key)}</td>'
                             f'<td style="padding:6px;white-space:pre-wrap">{escape(self._current[key]) or "(blank)"}</td>'
                             f'<td style="padding:6px;white-space:pre-wrap">{escape(value) or "(blank)"}</td></tr>')
            lines.append(f'</table><p>Correction reviewer: {escape(self.reviewer.value.strip())}</p>')
            self.comparison.value = ''.join(lines)
            self.replace_confirm.value = False
            self.replace_confirm.disabled = False
            self.correction_button.disabled = True
        except Exception as error:
            self._notify(str(error), error=True)

    def _authorize_correction(self, change):
        self.correction_button.disabled = not (change['new'] and self._editing
                                               and bool(self.comparison.value) and self._dirty())

    def _save(self, correction):
        button = self.correction_button if correction else self.save_button
        if button.disabled or not self._current:
            return
        try:
            decision = self._validated_decision()
            if correction and not (self._editing and self.replace_confirm.value and self.comparison.value):
                raise ValueError('Preview and authorize the correction first.')
            path = self.current_path
            index = self._paths.index(path)
            next_path = self._paths[(index + 1) % len(self._paths)] if len(self._paths) > 1 else None
            button.disabled = True
            with redirect_stdout(io.StringIO()):
                notice = self.save_decisions(
                    {path: decision}, reviewer=self.reviewer.value.strip(), status='confirmed',
                    confirmed_by_user=True, expected_rows={path: dict(self._current)},
                    correct_confirmed=correction)
            self._refresh(path if correction else next_path)
            self._notify(notice or ('Correction saved.' if correction else f'Saved {Path(path).name}.'))
        except Exception as error:
            self._set_controls()
            if correction:
                self._invalidate_preview()
            self._notify(str(error), error=True)
