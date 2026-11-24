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
| [playground.ipynb](playground.ipynb) | **Primary dataset exploration and preparation notebook.** Original exploration, exclusions, approved split, leakage limitations, and integrity checks, followed by validation-set confidence analysis and external-image testing. |
| [01_mobilenetv3.ipynb](01_mobilenetv3.ipynb) | Completed MobileNetV3-Small baseline. |
| [02_resnet18.ipynb](02_resnet18.ipynb) | Completed ResNet18 baseline. |
| [03_efficientnetb0.ipynb](03_efficientnetb0.ipynb) | Completed EfficientNet-B0 baseline. |
| [04_model_comparison.ipynb](04_model_comparison.ipynb) | Three-model comparison, candidate selection, and weighted-loss comparison. |
| [05_efficientnetb0_weighted_loss.ipynb](05_efficientnetb0_weighted_loss.ipynb) | Completed class-weighted cross-entropy experiment. |
| [06_wastewise_app_prototype.ipynb](06_wastewise_app_prototype.ipynb) | Gradio prototype with item-level guidance, the provisional 85% confidence threshold, and saved workflow validation. |

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
The verified experiment package versions are documented here. The standalone
Docker application has separate pinned dependencies in
[`application/requirements.txt`](application/requirements.txt).

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

The notebooks include saved results for viewing on GitHub without running Python
or obtaining the dataset. Image examples in the analysis notebooks are embedded;
they are not a replacement for the original image files.

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
all notebooks. Playground requires `data/dataset-resized.zip` and the extracted
images in `data/dataset-resized/`. Notebooks 01–05 require the extracted images to
recreate their example galleries. Their saved metrics, training curves, confusion
matrices, and numerical comparisons can be loaded from the published manifests
and JSON/CSV results without images or model checkpoints.

The model notebooks retain `MODE="reload"` and `RESTORE_CHECKPOINT=False`;
notebook 04 directly loads saved results. Reload does not run inference or
remeasure timing. A missing image stops its visualization cell; saved figures
remain viewable in the published notebook. Restoring a model for inference
requires its matching `best_model.pt`; the selected EfficientNet-B0 baseline
checkpoint is included in the repository. The [experiment guide](results/experiments/README.md)
describes the additional requirements for retraining.

Playground locates the project root from the current directory or its parents,
then resolves dataset paths through `data_preparation.get_project_root()`.
Model notebooks also support `ROOT_OVERRIDE` or `WASTEWISE_ROOT` when the checkout
is elsewhere. Each notebook initializes its own variables independently.

Playground uses read-only local dataset checks in place of its earlier
download/extraction cells. Its image-property summaries use the saved inventory,
and its original preparation examples use training images only. The appended
confidence analysis uses validation images and a separate 20-image external set;
it does not evaluate the held-out TrashNet test set. Its inference cells require
the selected EfficientNet-B0 checkpoint. The external test's
[support files](data/confidence_test/README.md) document sources, licenses,
reviewed labels, and duplicate checks; downloaded images remain excluded from Git
and must be obtained locally before rerunning those cells. Running the notebook
does not recreate the split or write source images. The original
class-distribution figure and historical exploration remain available; the
exact pre-update notebook is preserved as
`playground_before_preparation.ipynb` in an external local backup.

For a read-only source-integrity check outside JupyterLab:

```bash
python src/data_preparation.py verify
```

The older `audit` and `review` commands generate detailed evidence, refuse to
overwrite existing artifacts, and are not required for the current workflow.
Importing the module only defines functions; it does not run an audit.

## Local Docker deployment and monitoring

[`application/app.py`](application/app.py) runs the standalone Gradio workflow
with the selected unweighted EfficientNet-B0 model, CPU inference, the provisional
85% confidence threshold, and the existing UF recycling rules. Docker Compose runs
WasteWise, Prometheus, and Grafana locally, including on an Apple Silicon Mac.

Use Docker Desktop with Docker Compose and BuildKit support for named build
contexts. Keep `application/` inside the complete project checkout so Compose can
read `src/` and mount these existing files without copying them:

- `results/experiments/efficientnet_b0/baseline_seed42_5epochs/` contains
  `best_model.pt`, `config.json`, `metrics.json`, and `complete.json`, all included
  in the repository. The app checks the saved artifact hashes. The selected
  checkpoint is from the completed unweighted EfficientNet-B0 baseline run.
- `data/WasteWise_UF_Recycling_Rules.xlsx` is the authoritative rules workbook
  included in the repository. The model directory and workbook are mounted
  read-only; the original image dataset is not required to run the app.

From the repository root, start all three services:

```bash
cd application
docker compose up -d --build
```

Open [WasteWise](http://localhost:7860), [Prometheus](http://localhost:9090), and
[Grafana](http://localhost:3000). These ports bind only to `127.0.0.1`. Stop the
services from `application/` with:

```bash
docker compose down
```

The Docker image uses Python 3.13 (`python:3.13-slim-bookworm`) and the direct
dependency versions pinned in `application/requirements.txt`, including CPU
PyTorch/torchvision. Compose selects Prometheus `v3.16.0` and Grafana `13.3.1`.
This deployment environment is separate from the macOS/MPS experiment environment
above. The Python base-image patch version and transitive Python dependencies
are resolved during the build, so the configuration is not a complete immutable
environment lock.

### Manual Grafana configuration

Complete Grafana's first-login setup, add a **Prometheus** data source with the
server URL `http://prometheus:9090`, and select **Save & test**. This is the Compose
service address used by Grafana. Create dashboard panels manually using that
data source; no dashboard or data-source provisioning is included.

| Panel | PromQL query |
| --- | --- |
| Mean inference latency (seconds, last 5 minutes) | `sum(rate(wastewise_inference_duration_seconds_sum[5m])) / sum(rate(wastewise_inference_duration_seconds_count[5m]))` |
| Prediction count | `sum(wastewise_predictions_total)` |
| Predicted materials | `sum by (material) (wastewise_predictions_total)` |
| Withholding event count | `sum(wastewise_withholding_total)` |
| Withholding rate (percent, 0–100) | `100 * sum(wastewise_decisions_total{outcome="withheld"}) / sum(wastewise_decisions_total)` |

Count panels and the withholding percentage above cover the current app process;
counters reset when it restarts. Latency measures the model call and softmax,
excluding image decoding, preprocessing, and queue time. Predictions include
below-threshold results and expose all six material labels. The percentage uses
the first completed outcome per scan attempt (`withheld` or `guidance`), excluding
unfinished scans and processing errors; repeated rendering and navigation do not
add completed decisions. The event counter is not the percentage numerator.
Before any completed decisions, the percentage is undefined. An idle latency
window likewise has no samples.

Prometheus scrapes `wastewise:8000` every five seconds inside Compose; this metrics
port is not published to the host. Metrics contain aggregate numbers and fixed
material/outcome labels, without images, filenames, or personal/session labels.
Uploads use temporary container storage in `/tmp` (`tmpfs`) with the app's
existing session-disconnect and process-exit cleanup. Named volumes preserve
Prometheus history and manually created Grafana settings across ordinary
`docker compose down` / `up` cycles; those volumes are not repository files.

## Repository contents

The published project contains source code, completed notebooks with saved
outputs, and the CSV/JSON records needed for data loading and verification. The
original image dataset is excluded by `.gitignore` and must be provided locally
at the recorded paths; verify its fingerprints before reproducing the analysis.

Historical notebooks, eight superseded review records, and the review galleries
are preserved in an external local backup that is not part of the published
project. Galleries retain their layout under `data_review/`. The three
source/notebook snapshots matching historical experiment hashes are in
`experiment_provenance/`; current loaders and split
verification do not require them. Redundant verification outputs, autogenerated
notebook checkpoints, and two superseded archive records were removed.

[Baseline training-code evidence](provenance/baseline_training_evidence.json)
preserves the exact fingerprint-matching ResNet-18 notebook as UTF-8 text and
the recorded input cells from all three baseline training sessions. It identifies
the matching shared modules: the tracked `src/experiment_data.py` and the backup's
`experiment_provenance/baseline_experiment_training.py` and
`baseline_experiment_results.py`. The exact fingerprinted MobileNetV3-Small and
EfficientNet-B0 notebook files remain unrecovered; session history supports their
code provenance but does not establish those full-file matches or identical
retraining results. The evidence archive is not an executable notebook.

Historical notebooks and gallery links require their original project-relative
layout to be restored before reuse. The retained evidence tables still record
those original gallery paths. The removed `source_checksums_after.csv` was
identical to the retained `manifests/source_checksums_before.csv`. Exact historical
source versions require the external provenance snapshots and are not included
in a fresh checkout. The selected EfficientNet-B0 baseline checkpoint is included
in the repository; other model checkpoints remain local and excluded from Git.
Completed configurations, metrics, training histories, and prediction tables
remain eligible for publication.
