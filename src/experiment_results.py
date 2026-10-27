"""Metrics, saved-run loading and notebook figures; PyTorch is not required."""

import importlib.metadata
import json
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .experiment_data import open_image
from .data_preparation import file_sha256


MODEL_NAMES = ['mobilenetv3_small', 'resnet18', 'efficientnet_b0']
BASELINE = {
    'seed': 42, 'loss': 'CrossEntropyLoss', 'optimizer': 'Adam',
    'learning_rate': 0.001, 'batch_size': 32, 'epochs': 5,
    'weight_decay': 0.0, 'num_workers': 0, 'augmentation': 'none',
    'head': 'single_linear_layer', 'feature_mode': 'frozen_eval',
    'checkpoint_selection': 'highest_validation_macro_f1_earliest_tie',
    'precision': 'float32',
}


def package_table():
    rows = []
    for package in ['torch', 'torchvision', 'numpy', 'pandas', 'Pillow', 'matplotlib', 'ipykernel']:
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            version = 'MISSING — installation needs approval'
        rows.append({'Package': package, 'Version': version})
    return pd.DataFrame(rows).set_index('Package')


def classification_metrics(y_true, y_pred, classes):
    """All six classes contribute to macro-F1; undefined ratios are zero."""
    truth, predicted = np.asarray(y_true), np.asarray(y_pred)
    count = len(classes)
    if truth.ndim != 1 or truth.shape != predicted.shape or truth.size == 0:
        raise ValueError('Expected equally sized, nonempty label vectors.')
    if not (np.issubdtype(truth.dtype, np.integer) and np.issubdtype(predicted.dtype, np.integer)):
        raise ValueError('Labels must be integer class indices.')
    if np.any((truth < 0) | (truth >= count) | (predicted < 0) | (predicted >= count)):
        raise ValueError('Label outside the recorded class order.')
    matrix = np.zeros((count, count), dtype=np.int64)
    np.add.at(matrix, (truth, predicted), 1)
    tp = matrix.diagonal().astype(float)
    support = matrix.sum(axis=1)
    predicted_count = matrix.sum(axis=0)
    precision = np.divide(tp, predicted_count, out=np.zeros(count), where=predicted_count != 0)
    recall = np.divide(tp, support, out=np.zeros(count), where=support != 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(count), where=(precision + recall) != 0)
    tn = truth.size - support - predicted_count + tp
    per_class = [
        {'class': label, 'precision': float(precision[i]), 'recall': float(recall[i]),
         'f1': float(f1[i]), 'support': int(support[i]),
         'one_vs_rest_accuracy': float((tp[i] + tn[i]) / truth.size)}
        for i, label in enumerate(classes)
    ]
    return {'accuracy': float(tp.sum() / truth.size), 'macro_f1': float(f1.mean()),
            'weighted_f1': float(np.dot(f1, support) / truth.size),
            'n_validation': int(truth.size), 'per_class': per_class,
            'confusion_matrix': matrix.tolist()}


def run_path(root, model_name, run_name):
    if model_name not in MODEL_NAMES or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}', run_name):
        raise ValueError('Use a recognized model and a short run name without directory separators.')
    base = Path(root).resolve() / 'results/experiments'
    path = base / model_name / run_name
    if not path.resolve().is_relative_to(base.resolve()):
        raise ValueError('Run path escapes the experiment results directory.')
    return path


def require_new_run(root, model_name, run_name):
    path = run_path(root, model_name, run_name)
    if path.exists():
        raise FileExistsError(f'{path} already exists. Reload it or choose a new RUN_NAME; it will not be overwritten.')
    return path


def write_json_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def load_run(root, model_name, run_name, data, require_checkpoint=False):
    """Accept only a completed run using the exact approved validation examples."""
    directory = run_path(root, model_name, run_name)
    complete = json.loads((directory / 'complete.json').read_text())
    required = {'config.json', 'metrics.json', 'history.csv', 'validation_predictions.csv', 'best_model.pt'}
    if complete.get('status') != 'complete' or set(complete.get('sha256', {})) != required:
        raise ValueError('Missing or incomplete experiment artifacts.')
    for filename, expected in complete['sha256'].items():
        path = directory / filename
        if filename == 'best_model.pt' and not path.exists() and not require_checkpoint:
            continue  # Checkpoints are deliberately omitted from Git.
        if file_sha256(path) != expected:
            raise ValueError(f'Saved experiment artifact changed: {filename}')
    config = json.loads((directory / 'config.json').read_text())
    metrics = json.loads((directory / 'metrics.json').read_text())
    history = pd.read_csv(directory / 'history.csv')
    predictions = pd.read_csv(directory / 'validation_predictions.csv')
    if config['model_name'] != model_name or config['run_name'] != run_name or config['classes'] != data['classes']:
        raise ValueError('Saved model identity or class order differs.')
    for key in ['manifest_sha256', 'train_rows_sha256', 'validation_rows_sha256']:
        if config[key] != data[key]:
            raise ValueError(f'Saved experiment uses different examples: {key}')
    expected = data['validation'].set_index('path')['label']
    if not predictions['path'].is_unique or set(predictions['path']) != set(expected.index):
        raise ValueError('Predictions must contain every validation example exactly once.')
    if not predictions['path'].map(expected).eq(predictions['true_label']).all():
        raise ValueError('Prediction labels do not match the approved validation labels.')
    class_ids = {name: i for i, name in enumerate(data['classes'])}
    truth = predictions['true_label'].map(class_ids)
    predicted = predictions['predicted_label'].map(class_ids)
    if predicted.isna().any() or not truth.eq(predictions['true_index']).all() or not predicted.eq(predictions['predicted_index']).all():
        raise ValueError('Prediction class indices are inconsistent.')
    probabilities = predictions[[f'prob_{label}' for label in data['classes']]].to_numpy()
    if (not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any()
            or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5)
            or not np.array_equal(probabilities.argmax(axis=1), predicted.to_numpy())
            or not np.allclose(probabilities.max(axis=1), predictions['confidence'], atol=1e-6)):
        raise ValueError('Saved probabilities and predicted labels disagree.')
    if not predictions['correct'].eq(truth.eq(predicted)).all():
        raise ValueError('Saved correct/incorrect flags disagree with labels.')
    recalculated = classification_metrics(truth.to_numpy(), predicted.to_numpy(), data['classes'])
    for key in ['accuracy', 'macro_f1', 'weighted_f1', 'n_validation']:
        if not np.isclose(metrics[key], recalculated[key], atol=1e-10, rtol=0):
            raise ValueError(f'Saved metric differs from predictions: {key}')
    if metrics['confusion_matrix'] != recalculated['confusion_matrix']:
        raise ValueError('Saved confusion matrix differs from predictions.')
    for saved, actual in zip(metrics['per_class'], recalculated['per_class'], strict=True):
        if saved['class'] != actual['class'] or any(not np.isclose(saved[k], actual[k]) for k in actual if k != 'class'):
            raise ValueError('Saved per-class metrics differ from predictions.')
    if history['epoch'].tolist() != list(range(1, config['epochs'] + 1)):
        raise ValueError('The full configured training budget was not completed.')
    best_row = history.loc[history['validation_macro_f1'].idxmax()]
    if int(best_row['epoch']) != metrics['best_epoch'] or not np.isclose(best_row['validation_macro_f1'], metrics['macro_f1']):
        raise ValueError('The saved checkpoint epoch does not follow the recorded selection rule.')
    return {'directory': directory, 'config': config, 'metrics': metrics,
            'history': history, 'predictions': predictions}


def plot_examples(data, subset='train', seed=42):
    rows = data[subset] if subset in ('train', 'validation') else None
    if rows is None:
        raise ValueError('Test images are not available to this visualization.')
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    for ax, label in zip(axes.flat, data['classes']):
        choices = rows.loc[rows['label'] == label, 'path'].tolist()
        path = choices[int(rng.integers(len(choices)))]
        ax.imshow(open_image(data, path, subset))
        ax.set_title(label)
        ax.axis('off')
    fig.suptitle(f'Real {subset} examples (original RGB images)')
    fig.tight_layout()
    return fig


def plot_history(run):
    history = run['history']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history['epoch'], history['train_loss'], marker='o', label='Training')
    axes[0].plot(history['epoch'], history['validation_loss'], marker='o', label='Validation')
    axes[0].set(title='Cross-entropy loss', xlabel='Epoch', ylabel='Loss')
    axes[0].legend()
    axes[1].plot(history['epoch'], history['validation_macro_f1'], marker='o')
    axes[1].axvline(run['metrics']['best_epoch'], color='gray', linestyle='--', label='Selected epoch')
    axes[1].set(title='Validation macro-F1', xlabel='Epoch', ylabel='Macro-F1', ylim=(0, 1))
    axes[1].legend()
    for ax in axes:
        ax.set_xticks(history['epoch'])
        ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def plot_confusion(run, ax=None):
    matrix = np.asarray(run['metrics']['confusion_matrix'])
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(matrix, cmap='Blues', vmin=0)
    classes = run['config']['classes']
    ax.set(xticks=range(len(classes)), yticks=range(len(classes)),
           xticklabels=classes, yticklabels=classes,
           xlabel='Predicted category', ylabel='True category',
           title=f"{run['config']['display_name']} — validation counts")
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, str(matrix[i, j]), ha='center', va='center',
                    color='white' if matrix[i, j] > matrix.max() / 2 else 'black')
    ax.figure.tight_layout()
    return ax.figure


def prediction_examples(run, correct=False, limit=8):
    rows = run['predictions']
    if correct:
        return rows.loc[rows['correct']].sort_values(['confidence', 'path'], ascending=[True, True]).head(limit)
    errors = rows.loc[~rows['correct']].sort_values(['confidence', 'path'], ascending=[False, True])
    # If there are no errors, weak correct predictions remain useful to inspect.
    return errors.head(limit) if len(errors) else prediction_examples(run, correct=True, limit=limit)


def plot_predictions(data, rows, title):
    if len(rows) == 0:
        print('No predictions in this category.')
        return None
    fig, axes = plt.subplots(int(np.ceil(len(rows) / 4)), 4,
                             figsize=(13, 3.4 * int(np.ceil(len(rows) / 4))), squeeze=False)
    for ax in axes.flat:
        ax.axis('off')
    for ax, (_, row) in zip(axes.flat, rows.iterrows()):
        ax.imshow(open_image(data, row['path'], 'validation'))
        ax.set_title(f"True: {row['true_label']}\nPredicted: {row['predicted_label']} ({row['confidence']:.1%})", fontsize=10)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def confusion_pairs(run, limit=5):
    errors = run['predictions'].loc[~run['predictions']['correct']]
    return errors.groupby(['true_label', 'predicted_label']).size().sort_values(ascending=False).head(limit).rename('Images').reset_index()


def comparison_table(runs):
    """Validate comparable protocols; retain model order until all three exist."""
    configs = [run['config'] for run in runs]
    if len({c['model_name'] for c in configs}) != len(configs):
        raise ValueError('Choose one run per architecture.')
    for config in configs:
        if any(config.get(key) != expected for key, expected in BASELINE.items()) or config.get('loss_weighting'):
            raise ValueError('Baseline comparison requires the shared five-epoch head-only configuration.')
    keys = ['classes', 'manifest_sha256', 'train_rows_sha256', 'validation_rows_sha256', 'module_sha256']
    if configs and any(config[key] != configs[0][key] for config in configs[1:] for key in keys):
        raise ValueError('Runs use different examples, class orders or shared code; do not rank them together.')
    rows = []
    for run in runs:
        c, m = run['config'], run['metrics']
        rows.append({'Model': c['display_name'], 'Macro-F1': m['macro_f1'], 'Accuracy': m['accuracy'],
                     'Weighted F1': m['weighted_f1'], 'Validation images': m['n_validation'],
                     'Best epoch': m['best_epoch'], 'Trainable parameters': c['trainable_parameters'],
                     'Training seconds': m['training_seconds'], 'Device': c['device'],
                     'Forward median ms': m['latency']['forward']['median_ms'],
                     'End-to-end median ms': m['latency']['end_to_end']['median_ms']})
    frame = pd.DataFrame(rows)
    if len(runs) == len(MODEL_NAMES):
        frame = frame.sort_values('Macro-F1', ascending=False, kind='stable')
    return frame


def selection_evidence(runs):
    comparison_table(runs)  # Validate the protocol before naming a candidate.
    if {run['config']['model_name'] for run in runs} != set(MODEL_NAMES):
        return 'Selection pending: all three completed baseline results are required.'
    best = max(run['metrics']['macro_f1'] for run in runs)
    tied = [run for run in runs if np.isclose(run['metrics']['macro_f1'], best, atol=1e-12, rtol=0)]
    if len(tied) > 1:
        names = ', '.join(run['config']['display_name'] for run in tied)
        return f'Macro-F1 tie between {names} ({best:.4f}). Review per-class performance and measured local latency before choosing one model.'
    winner = tied[0]
    weakest = min(winner['metrics']['per_class'], key=lambda row: row['f1'])
    others = [run['metrics']['macro_f1'] for run in runs if run is not winner]
    pairs = confusion_pairs(winner, limit=1)
    finding = (f"Its most frequent validation confusion is {pairs.iloc[0]['true_label']} → "
               f"{pairs.iloc[0]['predicted_label']} ({int(pairs.iloc[0]['Images'])} images).") if len(pairs) else 'No incorrect validation predictions were recorded.'
    modification = (
        'A targeted follow-up hypothesis is to unfreeze only the final feature-extraction block, '
        'allowing more task-specific features for these errors. The validation evidence does not prove '
        'that frozen features caused them; inspect the examples before accepting this change. '
        'Retain the same validation rows and metrics, record the revised configuration separately, '
        'and measure its latency again on this MacBook. No improvement run is started here.'
        if len(pairs) else
        'There is no validation-error evidence to justify fine-tuning yet. Review runtime constraints '
        'before specifying the required targeted modification; do not inspect the test set.'
    )
    return (
        f"**Leading baseline by macro-F1: {winner['config']['display_name']}** ({best:.4f}; "
        f"gap to second place {best - max(others):.4f}). Its lowest class F1 is "
        f"{weakest['class']} ({weakest['f1']:.4f}). {finding}\n\n"
        'This is a single-seed validation comparison, not evidence of a statistically significant advantage. '
        'Check class-specific trade-offs and the measured time/latency before recording the final selection.\n\n'
        + modification
    )
