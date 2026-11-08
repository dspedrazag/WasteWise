# Manual item-level review

Use `item_review.ipynb` in this folder to review the 1,295 plastic, metal, and
cardboard images. This notebook includes manual annotation and read-only results
and validation. Model development remains outside this workflow.

## Open the notebook

1. Open Terminal and start JupyterLab with the project's existing environment:

   ```bash
   cd /Users/sofiapedraza/Projects/WasteWise
   /Users/sofiapedraza/python-envs/scientific/bin/python -m jupyterlab
   ```

2. In JupyterLab, open `item_level_review/item_review.ipynb`.
3. Select **Python (scientific)**. Read the annotation criteria at the top,
   including the restriction on test images.
4. Choose **Kernel → Restart Kernel and Run All Cells**. Only this review
   notebook needs to run. Its optional gallery and dictionary examples are
   inactive. The Interactive review workspace section opens the panel;
   Item-Level Review Results and Validation appears later in the notebook.

The notebook also works if Jupyter starts from this folder. It finds the project
root and resolves the original dataset and manifest from there. Existing saved
gallery outputs were preserved and may appear above the panel; they are
historical displays, not new review results. A static notebook preview cannot
operate the controls: use JupyterLab with a running kernel.

The approved widget installation added `ipywidgets==8.1.9`,
`jupyterlab_widgets==3.0.17`, and `widgetsnbextension==4.0.16` to `scientific`.
All pre-existing package versions were preserved. If widgets appear as text,
restart JupyterLab and refresh its browser tab. No notebook cell installs packages.

## Review and save

1. Enter your name in **Reviewer**.
2. Select **Plastic**, **Metal**, or **Cardboard** under **Original category**.
   **Remaining** shows unreviewed images and unconfirmed proposals. The panel
   shows confirmed/total and remaining counts overall and for all three categories.
3. Inspect the photograph and filename. **Enlarge image** increases its display
   size. Original material, subset, exclusion, and review status are visible.
4. Choose **Reviewed material** independently of the original folder, or choose
   **Cannot verify material**. Select an **Item type** using the established
   criteria. The interface suggests its corresponding **Scope** and checks
   consistency before saving. No item or material is inferred by a model.
5. Add notes when useful. A short explanation is **required for ambiguous scope**;
   notes are optional for supported/unsupported annotations. For an “other” item,
   naming the identifiable object in the notes is useful evidence.
6. Click **Save** only after personally reviewing the annotation. This records
   `confirmed`, your name, and the UTC time in the existing manifest, then advances.
   A confirmed review can still have ambiguous scope or an unverified material.

**Skip** leaves an image pending. **Previous**, **Next**, and the filename selector
browse the selected category/view. Browsing wraps around at the ends. Navigation
never saves automatically. If you have unsaved edits, choose **Discard edits and
continue** or **Keep editing**. Reviewer names remain in the current session, but
the controls do not silently reuse a previous reviewer's name after reopening.

Excluded images remain in this review population and its progress counts. Their
experimental exclusion is preserved even when their annotation is confirmed.

## Correct a confirmed annotation

1. Select its original category and choose **Confirmed** or **All** under **Show**.
2. Select the filename. Saved annotations initially open with locked fields.
3. Click **Edit confirmed** and make the correction.
4. Click **Preview correction** and compare the saved and proposed values.
5. Check **I authorize this correction** after reviewing the preview.
6. Click **Save correction**. The corrected image stays visible with locked fields.

Changing any field or the reviewer after previewing requires a fresh preview and
authorization. The ordinary **Save** action cannot replace a confirmed record.
**Cancel edit** returns to the saved record, with a discard check if needed.

`corrections.jsonl` is created on the first correction. It records the before/after
row, reviewer/timestamp, transaction identifier, CSV checksums, and outcome. The
CSV remains the sole authoritative current annotation manifest. A `prepared`
event retains the prior values before the CSV changes; `committed` confirms the
write, and `not_applied` identifies a prepared change that did not reach the CSV.
An interrupted event is resolved by exact checksums before the next save, without
replaying annotations. If the history cannot be reconciled, saving stops for
inspection. Do not edit or delete the history to bypass that check.

## Resume and handle a save problem

- After reopening, run the review notebook again, enter your name, and choose your
  category. **Remaining** loads the first remaining image from the saved manifest.
  Skipped images remain pending. Saved annotations are never reset. Unsaved drafts
  and the browsing position are not saved across kernel shutdowns.
- When a category has no remaining images, choose **All** or **Confirmed** to revisit
  its records, or switch categories.
- If the image cannot be loaded, saving is disabled. Restore access to its existing
  dataset path; do not relocate the original image.
- If a record changed in another session, the stale form cannot overwrite it.
  Choose **Reload saved record**, discard the stale draft, and inspect the current
  values before trying again. Prefer one active review kernel.
- If a save reports a failure, retain the form and inspect the message. The CSV is
  validated and replaced atomically. A message beginning **Saved. History completion
  is pending** means the annotation was saved; history completion is checked before
  the next save. The internal `.review.lock` coordinates concurrent saves.

## Files and preserved data

| Location relative to project root | Purpose |
| --- | --- |
| `item_level_review/item_review.ipynb` | Review notebook, fixed annotation criteria, original helper functions, panel launcher, and read-only results and validation. |
| `item_level_review/review_ui.py` | Widget controls and correction persistence. |
| `item_level_review/tests/test_review_ui.py` | Tests using temporary synthetic records and generated test pictures. |
| `item_level_review/corrections.jsonl` | Correction audit events, created only when needed. |
| `manifests/item_review_v1.csv` | The existing authoritative annotation manifest; its path and columns are unchanged. |
| `data/dataset-resized/` | Original images, read directly without moving, copying, or editing them. |

Results and validation are included in `item_review.ipynb`; no separate report
files or exports are generated. The old project-root notebook was
moved, so open the new path rather than an old bookmark. There is one
authoritative review notebook and one authoritative annotation manifest.
Jupyter checkpoint copies are local snapshots.

Original image paths, material labels, split assignments, group identifiers, and
exclusions remain read-only. Item scope does not decide recyclability or training
eligibility. Test images may be reviewed manually for annotation/documentation
only, without predictions. Their annotations must not inform model selection,
threshold tuning, retraining, or changes to the evaluation procedure. The fixed
criteria must not be revised in response to test examples.

## Focused checks

From the project directory:

```bash
/Users/sofiapedraza/python-envs/scientific/bin/python -B -m unittest discover -s item_level_review/tests -v
```

The tests do not change the real manifest or use original images. They cover
category navigation, progress, ambiguity notes, saving, stale forms, explicit
corrections/history, interrupted saves, and metadata preservation. No model or
experiment notebook is executed.
