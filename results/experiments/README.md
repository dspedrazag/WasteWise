# WasteWise experiments

The three independent notebooks use the approved split without modifying it:

- `01_mobilenetv3.ipynb`: MobileNetV3-Small.
- `02_resnet18.ipynb`: ResNet18.
- `03_efficientnetb0.ipynb`: EfficientNet-B0.
- `04_model_comparison.ipynb`: completed results, error analysis and selection.
- `05_efficientnetb0_weighted_loss.ipynb`: completed weighted-loss follow-up.

All three baselines and the weighted-loss experiment are complete. Their real
metrics, histories, and predictions are saved in separate run directories.
The experiment notebooks default to `MODE="reload"`; checkpoints remain local
and are excluded from Git.

## Environment

Use a Python environment and Jupyter kernel with the packages documented in
the [project README](../../README.md#python-dependencies) and the model-library
versions recorded in each run's `config.json`. The original kernel was
**Python (scientific)**. Scikit-learn is unnecessary:
metrics are computed from a confusion matrix with NumPy.

For reproducing the documented direct dependencies in a compatible Python
3.13 environment, the equivalent installation command is:

```bash
python -m pip install \
  torch==2.14.1 torchvision==0.29.1 numpy==2.5.3 pandas==3.0.6 \
  Pillow==12.3.0 matplotlib==3.11.2 jupyterlab==4.6.4 \
  ipykernel==7.4.0 IPython==9.17.1
```

Here, `python` denotes the chosen environment's interpreter. This setup command
is not part of notebook execution. Optional programmatic
notebook verification also used `nbformat==5.11.1` and `nbclient==0.11.0`.
Transitive dependencies are not locked by this list. Each completed run records
Python, platform, device, and shared-code fingerprints in `config.json`.

The notebooks search the current directory and its parents, then the existing
`~/Projects/WasteWise` checkout. If the checkout moves, set `ROOT_OVERRIDE` or
`WASTEWISE_ROOT`. They do not change the kernel working directory or depend on
variables from another notebook.

## Execute one baseline

1. Open the first model notebook in JupyterLab and select **Python (scientific)**.
2. Inspect the notebook in `review` mode. Missing PyTorch is reported without
   preventing the dataset summary from being inspected.
3. For an intentional new experiment, select a new run name and set
   `MODE="train"`. The first model
   construction downloads the requested ImageNet weights if absent from the
   standard local Torch cache; it never uploads or downloads TrashNet images.
4. Run cells sequentially. Epoch progress prints below the training cell. Tables,
   loss curves, macro-F1 curves, confusion matrix, real validation predictions
   and timing results appear inline after training.
5. Save the notebook with **Cmd/Ctrl+S** to preserve the executed outputs on GitHub.
   Review that baseline before proceeding to the next notebook.

All baselines use seed 42, CrossEntropyLoss, Adam (learning rate 0.001, zero weight
decay), batch size 32 and five epochs. The complete original classification head
is replaced with one six-output linear layer. Pretrained convolutional features
and pooling are retained. Parameters outside that head are frozen; the backbone
also stays in evaluation mode so batch-normalization statistics and stochastic
layers do not change. The best validation macro-F1 epoch is saved, with the earliest
epoch winning exact ties. All five epochs are run; there is no early stopping.

The explicit `IMAGENET1K_V1` weight transforms are used for both subsets without
augmentation. MobileNetV3-Small and ResNet18 use bilinear resizing; EfficientNet-B0
uses bicubic resizing. All three use a 256-pixel shorter edge, a 224-pixel center
crop, and the weight-specific ImageNet normalization. The transform actually used
is saved in the configuration. See the linked Torchvision documentation in each
notebook. Center cropping is a baseline choice and can remove peripheral material.

MPS is selected when available; otherwise CPU is selected. If an operation is
unsupported on MPS, the run stops rather than silently mixing devices. Choose
`DEVICE_PREFERENCE="cpu"` explicitly and a new run name to retry. Seeded order and
deterministic-operation warnings improve reproducibility but do not guarantee
bit-identical results across devices or versions.

## Saved files and reload

```text
results/experiments/
  README.md
  mobilenetv3_small/<RUN_NAME>/
  resnet18/<RUN_NAME>/
  efficientnet_b0/<RUN_NAME>/
    config.json                  # Actual settings, weights, preprocessing, versions, fingerprints
    history.csv                  # All epoch losses, accuracy, validation macro-F1 and duration
    validation_predictions.csv   # One row per approved validation image, probabilities and labels
    metrics.json                 # Aggregate/per-class scores, confusion counts, timing samples
    best_model.pt                # Full selected model state, including the frozen features
    complete.json                # Written last; artifact fingerprints identify a complete run
```

The default run name is `baseline_seed42_5epochs`. Existing run directories are
never reused for training, even if incomplete. Choose a distinct name for an
intentional new run; no old checkpoint is overwritten. An interrupted run without
`complete.json` cannot enter the comparison. The full checkpoint is saved once,
after all five epochs; interrupted runs do not support training resumption.

The saved outputs are viewable on GitHub without executing the notebooks.
`MODE="reload"` with the original `RUN_NAME` reads recorded results without
training. With `RESTORE_CHECKPOINT=False`, numerical tables, training curves, and
confusion matrices need neither PyTorch nor `best_model.pt`. Original images in
`data/dataset-resized/` are still required by the example and prediction galleries;
without them, those cells stop with a missing-file error. Notebook 04 has the
same image requirement for its error gallery. Reload does not run inference or
replace the original timings.

`RESTORE_CHECKPOINT=True` additionally requires PyTorch and the local checkpoint;
it reconstructs the model with `weights_only=True` without downloading pretrained
weights. The ZIP is needed for Playground's original-dataset checks, but not for
the model notebooks' reload mode.

Retraining requires the original images, the unchanged approved manifests,
PyTorch/Torchvision, and the specified ImageNet weights. Baseline training may
download weights if they are not cached. The weighted experiment additionally
requires the original EfficientNet-B0 checkpoint and cached ImageNet weights for
its initialization checks; its controls also reject changes to the recorded
baseline device and software settings. Completed run directories are protected against
overwriting; new baseline runs need distinct names, while the weighted notebook
retains its fixed controlled-run name and is not a general rerun interface.
Matching seeds and package versions does not guarantee identical results across
devices. Published configuration hashes retain the original run provenance;
editorial notebook updates do not rewrite those records.

Small JSON/CSV files and notebooks with saved outputs can be committed.
Checkpoints (`*.pt`) and original data are already excluded by `.gitignore`.
Do not clear notebook outputs when preparing the homework submission.

## Comparison and the weighted-loss experiment

Choose one completed run per architecture in the comparison notebook. It checks
the shared settings, source-code fingerprints and exact training/validation
membership before ranking. Partial results can be listed but are not ranked.
Metrics are recalculated from saved predictions to check their consistency.

Macro-F1 is primary; also inspect class F1, errors, training duration and local
latency. Latency uses the same seeded validation image sample, batch size one,
ten warm-up iterations and fifty timed repetitions, with MPS synchronization.
Forward-only and end-to-end measurements are reported separately. Filesystem
cache, power state and other applications affect measurements; compare the
recorded device/software context as well.

The completed follow-up changed only the training loss to class-weighted
cross-entropy, using training-set class counts. Validation loss remained
unweighted. Its results are saved under
`efficientnet_b0/weighted_ce_seed42_5epochs/`; the unweighted baseline remains the
selected model. Notebook 04 compares both configurations while preserving the
original three-model comparison. No HiPerGator job has been launched. The test
set is unused for image inspection, evaluation, latency, selection, or tuning.

## Reproducibility checks

Notebook schemas and Python syntax can be checked without executing notebooks.
The lightweight tests cover hand-calculated metric behavior, protected test-image
access, split membership, refusal to overwrite a run, and pending comparison
behavior. Their mathematical fixtures are not experimental results:

```bash
python -m unittest discover -s tests -v
```

The completed notebooks retain their measured outputs. Reload mode reads saved
predictions and measurements without retraining or replacing experimental results.
