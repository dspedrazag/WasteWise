# WasteWise

WasteWise is a course prototype for image-based waste classification and
Gainesville-specific disposal guidance. The completed experiments compare three
pretrained classifiers and a class-weighted-loss follow-up using TrashNet.

## Dataset and inclusion

[TrashNet](https://github.com/garythung/trashnet), created by Gary Thung and Mindy
Yang, contains 2,527 images in six original categories: cardboard, glass, metal,
paper, plastic, and trash. The saved audit found readable RGB JPEG images with
dimensions of 512 × 384 pixels. Class sizes differ substantially, particularly
for the smaller `trash` category.

**The inclusion decision is final: 2,521 images are eligible for experiments.**
Six images in three identical decoded-image pairs have conflicting category
labels and are excluded through the manifest only. All 2,527 original images
and `data/dataset-resized.zip` are preserved unchanged. No additional exclusions
are part of this preparation.

Different views, lighting conditions, and backgrounds remain useful variation.
Supported photographs of the same object or closely related captures should
stay within one subset. Perceptual similarity and shared packaging alone do not
establish object identity. The existing review is incomplete evidence of
relatedness, so residual leakage remains a limitation.

## Notebook organization

| Notebook | Role and current status |
| --- | --- |
| [playground.ipynb](playground.ipynb) | **Primary dataset exploration and preparation notebook.** Original exploration, exclusions, approved split, leakage limitations, and integrity checks. Image examples use training images only. |
| [01_mobilenetv3.ipynb](01_mobilenetv3.ipynb) | Completed MobileNetV3-Small baseline. |
| [02_resnet18.ipynb](02_resnet18.ipynb) | Completed ResNet18 baseline. |
| [03_efficientnetb0.ipynb](03_efficientnetb0.ipynb) | Completed EfficientNet-B0 baseline. |
| [04_model_comparison.ipynb](04_model_comparison.ipynb) | Three-model comparison, candidate selection, and weighted-loss comparison. |
| [05_efficientnetb0_weighted_loss.ipynb](05_efficientnetb0_weighted_loss.ipynb) | Completed class-weighted cross-entropy experiment. |

Each model notebook has its own architecture definition and run controls, with
shared loading, training, evaluation and visualization code in
`src/experiment_data.py`, `src/experiment_training.py` and
`src/experiment_results.py`, with the controlled follow-up in
`src/weighted_experiment.py`. Completed experiment notebooks default to
`MODE="reload"` to display saved results without training. The unweighted
EfficientNet-B0 remains the selected model. [The experiment guide](results/experiments/README.md)
documents environment setup, saved artifacts, timing, and execution modes.

The verified
seed-42 split contains **1,765 training, 378 validation, and 378 test images**,
targeting 70/15/15 proportions. All six class allocations are within one image
of their ideal proportions. The held-out test set remains unused for model
selection, tuning, and evaluation throughout this homework.

## Authoritative preparation records

A manifest is a table describing files and their experimental use. The active
records are:

- `manifests/image_inventory.csv`: all original image paths, labels, properties,
  and fingerprints.
- `manifests/duplicate_review.csv`: the six final exclusions and their evidence.
- `manifests/dataset_metadata.json`: final counts, class order, record checksums,
  methodology, grouping limitations, and the verified split reference.
- `manifests/integrity_before.json`: the original ZIP and source-integrity baseline.
- `manifests/source_checksums_before.csv`: original per-file SHA-256 fingerprints.
- `manifests/split_manifest.csv`: one row per eligible image, with its original
  label, subset, and stable group identifier.
- `manifests/split_metadata.json`: seed, algorithm, achieved counts, grouping
  decisions, input/code checksums, reproducibility evidence, and limitations.

SHA-256 fingerprints allow byte-for-byte preservation checks. The source-file
baseline covers 2,527 images and one non-image metadata file. Image eligibility
is derived from the inventory minus the six exclusions, without copying or
relabeling images.

[The retained review evidence](manifests/archive/data_review_v1/README.md) includes
the three checksum-bound inputs required to reconstruct and verify the split,
plus four inputs used by the legacy review-generation command. Superseded reports
and galleries are retained in the external backup described below. The split
adopts 32 supported proposals, two supported subgroups, and nine conservative
damage/handling links: **43 related groups covering 99 images**, plus 2,422
singleton IDs. A singleton ID does not certify independence. All accepted
groups stay within one subset. Generic resemblance and shared commercial design
were not sufficient to accept broader groups.

Residual identity uncertainty remains: 38 unaccepted or partially accepted
broad proposals and 44 unaccepted candidate pairs have members across subsets.
Their evidence and treatment are recorded explicitly; the review cannot
guarantee zero leakage. Inclusion remains unchanged at 2,521 images.

`src/data_splitting.py` reconstructs the assignments in memory and verifies the
saved manifest without reading test-image pixels. Verify the split with:

```bash
python -m src.data_splitting verify
```

The separate `create` action refuses to overwrite either existing split file.
All experiments load the saved manifest and use the same validation rows.

## Python dependencies

The experiments used **Python 3.13.16** on macOS with Apple MPS acceleration.
The verified package versions are documented here instead of separate
requirements files:

| Package | Version |
| --- | --- |
| torch | 2.14.1 |
| torchvision | 0.29.1 |
| numpy | 2.5.3 |
| pandas | 3.0.6 |
| Pillow | 12.3.0 |
| matplotlib | 3.11.2 |
| jupyterlab | 4.6.4 |
| ipykernel | 7.4.0 |
| IPython | 9.17.1 |

Programmatic notebook verification used `nbformat==5.11.1` and
`nbclient==0.11.0`; these are not additional model libraries. Classification
metrics use NumPy, so scikit-learn is not required.

Each saved experiment's `config.json` records its model-library versions,
platform, device, seed, and data fingerprints. This version list is not a full
environment lock: transitive dependencies are not pinned, and matching versions
does not guarantee identical results across devices. Installation commands are
in the [experiment guide](results/experiments/README.md#environment).

## Viewing and rerunning the notebooks

All six notebooks include saved tables and figures for viewing on GitHub without
running Python or obtaining the dataset. The image examples are embedded in the
notebooks; they are not a replacement for the original image files.

For local execution, use an environment with the documented packages and start
JupyterLab from the cloned project directory:

```bash
cd WasteWise
python -m jupyterlab
```

Here, `python` denotes that environment's interpreter. Select its Jupyter kernel;
the original experiments used **Python (scientific)**. No notebook installs or
upgrades packages.

The published files alone do not support a complete top-to-bottom execution of
the six notebooks. Playground requires `data/dataset-resized.zip` and the extracted
images in `data/dataset-resized/`. Notebooks 01–05 require the extracted images to
recreate their example galleries. Their saved metrics, training curves, confusion
matrices, and numerical comparisons can be loaded from the published manifests
and JSON/CSV results without images or model checkpoints.

The model notebooks retain `MODE="reload"` and `RESTORE_CHECKPOINT=False`;
notebook 04 directly loads saved results. Reload does not run inference or
remeasure timing. A missing image stops its visualization cell; saved figures
remain viewable in the published notebook. Restoring a model for inference
requires its excluded `best_model.pt`. The [experiment guide](results/experiments/README.md)
describes the additional requirements for retraining.

Playground locates the project root from the current directory or its parents,
then resolves dataset paths through `data_preparation.get_project_root()`.
Model notebooks also support `ROOT_OVERRIDE` or `WASTEWISE_ROOT` when the checkout
is elsewhere. Each notebook initializes its own variables independently.

Playground uses read-only local dataset checks in place of its earlier
download/extraction cells. Its image-property summaries use the saved inventory,
and its examples use training images only. Running it does not recreate the split
or write source images. The original class-distribution figure and historical
exploration remain available; the exact pre-update notebook is preserved as
`playground_before_preparation.ipynb` in an external local backup.

For a read-only source-integrity check outside JupyterLab:

```bash
python src/data_preparation.py verify
```

The older `audit` and `review` commands generate detailed evidence, refuse to
overwrite existing artifacts, and are not required for the current workflow.
Importing the module only defines functions; it does not run an audit.

## Repository contents

The published project contains source code, completed notebooks with saved
outputs, and the CSV/JSON records needed for data loading and verification. The
original image dataset is excluded by `.gitignore` and must be provided locally
at the recorded paths; verify its fingerprints before reproducing the analysis.

Historical notebooks, eight superseded review records, and the review galleries
are preserved in an external local backup, which is not part of the published
project. Galleries retain their layout under `data_review/`. The three
source/notebook snapshots matching historical experiment hashes are in
`experiment_provenance/`; current loaders and split
verification do not require them. Redundant verification outputs, autogenerated
notebook checkpoints, and two superseded archive records were removed.

Historical notebooks and gallery links require their original project-relative
layout to be restored before reuse. The retained evidence tables still record
those original gallery paths. The removed `source_checksums_after.csv` was
identical to the retained `manifests/source_checksums_before.csv`. Exact historical
source versions require the external provenance snapshots and are not included
in a fresh checkout. Model checkpoints remain local and excluded from Git;
completed configurations, metrics, training histories, and prediction tables
remain eligible for publication.
