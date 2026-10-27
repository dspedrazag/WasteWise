"""Frozen-head training for the baselines and the controlled weighted-loss run."""

import importlib.metadata
import json
import platform
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except ImportError as error:
    raise ImportError('PyTorch is required for training/checkpoint loading. Install torch and torchvision '
                      'in the scientific environment only after approval, then restart the kernel.') from error

from .experiment_data import open_image, subset_rows
from .experiment_results import (BASELINE, classification_metrics, load_run,
                                 require_new_run, write_json_new)
from .data_preparation import file_sha256


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # MPS/kernel/version differences can still prevent bit-for-bit reproducibility.
    torch.use_deterministic_algorithms(True, warn_only=True)


def select_device(preference='auto'):
    if preference not in ('auto', 'cpu'):
        raise ValueError('Choose auto (MPS when available) or cpu.')
    return torch.device('mps' if preference == 'auto' and torch.backends.mps.is_available() else 'cpu')


def synchronize(device):
    if device.type == 'mps':
        torch.mps.synchronize()


class ManifestDataset(Dataset):
    def __init__(self, data, subset, transform):
        self.data = data
        self.subset = subset
        self.rows = subset_rows(data, subset)
        self.transform = transform
        self.class_to_index = {label: i for i, label in enumerate(data['classes'])}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        image = open_image(self.data, row['path'], self.subset)
        return self.transform(image), self.class_to_index[row['label']], row['path']


def make_loaders(data, transform, config):
    generator = torch.Generator().manual_seed(config['seed'])
    training = ManifestDataset(data, 'train', transform)
    validation = ManifestDataset(data, 'validation', transform)
    common = {'batch_size': config['batch_size'], 'num_workers': 0,
              'pin_memory': False, 'drop_last': False}
    # Zero workers avoids multiprocessing/kernel issues on macOS Jupyter.
    return (DataLoader(training, shuffle=True, generator=generator, **common),
            DataLoader(validation, shuffle=False, **common))


def model_information(model, head, weights):
    parameters = list(model.parameters())
    trainable = {id(p) for p in parameters if p.requires_grad}
    if trainable != {id(p) for p in head.parameters()} or not trainable:
        raise ValueError('Only the replacement classification head may be trainable.')
    transform = weights.transforms()
    return {
        'weights': str(weights), 'weights_url': weights.url,
        'total_parameters': sum(p.numel() for p in parameters),
        'trainable_parameters': sum(p.numel() for p in parameters if p.requires_grad),
        'trainable_parameter_names': [name for name, p in model.named_parameters() if p.requires_grad],
        'preprocessing': {
            'resize_size': transform.resize_size, 'crop_size': transform.crop_size,
            'interpolation': transform.interpolation.value, 'antialias': transform.antialias,
            'mean': transform.mean, 'std': transform.std,
            'training_and_validation': 'identical deterministic weights.transforms(); no augmentation',
        },
    }


def make_config(data, model_name, display_name, run_name, model, head, weights, device, notebook_name, settings):
    info = model_information(model, head, weights)
    versions = {name: importlib.metadata.version(name)
                for name in ['torch', 'torchvision', 'numpy', 'pandas', 'Pillow', 'matplotlib']}
    module_names = ['src/experiment_data.py', 'src/experiment_training.py', 'src/experiment_results.py']
    return {
        **settings, **info, 'architecture': str(model),
        'schema_version': 1, 'model_name': model_name, 'display_name': display_name,
        'run_name': run_name, 'classes': data['classes'],
        'manifest_sha256': data['manifest_sha256'],
        'train_rows_sha256': data['train_rows_sha256'],
        'validation_rows_sha256': data['validation_rows_sha256'],
        'n_train': len(data['train']), 'n_validation': len(data['validation']),
        'device': str(device), 'python': platform.python_version(),
        'platform': platform.platform(), 'machine': platform.machine(),
        'cpu_threads': torch.get_num_threads(), 'package_versions': versions,
        'deterministic_algorithms': 'enabled with warnings; cross-device equality is not guaranteed',
        'optimizer_kwargs': {'betas': [0.9, 0.999], 'eps': 1e-8, 'amsgrad': False},
        'module_sha256': {name: file_sha256(data['root'] / name) for name in module_names},
        'notebook': notebook_name, 'notebook_sha256': file_sha256(data['root'] / notebook_name),
        'latency_protocol': {'batch_size': 1, 'sample_count': 20, 'warmup': 10, 'repeats': 50,
                             'seed': 42, 'subset': 'validation'},
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
    }


def evaluate_validation(model, loader, device, classes):
    if loader.dataset.subset != 'validation':
        raise ValueError('This evaluation routine accepts validation images only.')
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    rows = []
    with torch.inference_mode():
        for images, labels, paths in loader:
            logits = model(images.to(device))
            loss = loss_fn(logits, labels.to(device))
            total_loss += loss.item() * len(labels)
            probabilities = logits.softmax(dim=1).cpu().numpy()
            for path, truth, probs in zip(paths, labels.tolist(), probabilities):
                prediction = int(probs.argmax())
                rows.append({
                    'path': path, 'true_label': classes[truth], 'predicted_label': classes[prediction],
                    'true_index': truth, 'predicted_index': prediction,
                    'confidence': float(probs[prediction]), 'correct': truth == prediction,
                    **{f'prob_{name}': float(probs[i]) for i, name in enumerate(classes)},
                })
    predictions = pd.DataFrame(rows)
    metrics = classification_metrics(predictions['true_index'].to_numpy(), predictions['predicted_index'].to_numpy(), classes)
    metrics['validation_loss'] = total_loss / len(predictions)
    return metrics, predictions


def benchmark_latency(model, dataset, device, protocol):
    """Batch-one local latency; synchronized MPS timings, no test examples."""
    if dataset.subset != 'validation':
        raise ValueError('Latency measurements use validation images only.')
    rng = np.random.default_rng(protocol['seed'])
    indices = rng.choice(len(dataset), min(protocol['sample_count'], len(dataset)), replace=False).tolist()
    paths = [dataset.rows.iloc[i]['path'] for i in indices]
    # Preprocessing and device transfer are excluded from forward-only timing.
    tensors = [dataset[i][0].unsqueeze(0).to(device) for i in indices]
    model.eval()
    forward_times, full_times = [], []
    with torch.inference_mode():
        for i in range(protocol['warmup']):
            model(tensors[i % len(tensors)])
        synchronize(device)
        for i in range(protocol['repeats']):
            tensor = tensors[i % len(tensors)]
            synchronize(device)
            start = time.perf_counter()
            model(tensor)
            synchronize(device)
            forward_times.append((time.perf_counter() - start) * 1000)
        # Warm the full decode/transform/transfer/forward/argmax path as well.
        for i in range(protocol['warmup']):
            tensor = dataset[indices[i % len(indices)]][0].unsqueeze(0).to(device)
            model(tensor).argmax(dim=1).cpu().item()
        synchronize(device)
        for i in range(protocol['repeats']):
            synchronize(device)
            start = time.perf_counter()
            tensor = dataset[indices[i % len(indices)]][0].unsqueeze(0).to(device)
            model(tensor).argmax(dim=1).cpu().item()
            synchronize(device)
            full_times.append((time.perf_counter() - start) * 1000)

    def summarize(values):
        return {'mean_ms': float(np.mean(values)), 'median_ms': float(np.median(values)),
                'p95_ms': float(np.percentile(values, 95)), 'samples_ms': values}

    return {**protocol, 'device': str(device), 'sample_paths': paths,
            'forward': summarize(forward_times), 'end_to_end': summarize(full_times),
            'forward_definition': 'resident batch-one tensor to logits, with device synchronization',
            'end_to_end_definition': 'JPEG read, RGB conversion, preprocessing, transfer, forward and predicted class on CPU',
            'limitations': 'Warm local filesystem/model measurements; excludes model loading, UI and camera capture. Run on comparable device/load.'}


def train_baseline(data, model, head, transform, config, device):
    """Keep the original, unweighted baseline behavior."""
    if any(config.get(key) != value for key, value in BASELINE.items()) or config.get('loss_weighting'):
        raise ValueError('This function is limited to the approved baseline configuration.')
    return _train_head(data, model, head, transform, config, device)


def train_weighted(data, model, head, transform, config, device):
    """Run the approved loss-only experiment after its pre-training checks."""
    from .weighted_experiment import validate_training_start
    validate_training_start(data, model, head, transform, config, device)
    return _train_head(data, model, head, transform, config, device)


def training_criterion(config, device):
    """Construct the training loss; validation always uses unweighted CE."""
    if config['loss'] == 'CrossEntropyLoss':
        return nn.CrossEntropyLoss()
    if config['loss'] != 'ClassWeightedCrossEntropyLoss':
        raise ValueError('Unsupported training loss.')
    weights = torch.tensor(config['loss_weighting']['weights'], dtype=torch.float32, device=device)
    if weights.ndim != 1 or len(weights) != len(config['classes']) or not torch.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError('Class weights must be finite, positive and in the saved class order.')
    return nn.CrossEntropyLoss(weight=weights, reduction='mean')


def _train_head(data, model, head, transform, config, device):
    """Shared execution path; all files are new and no test images are available."""
    if {id(p) for p in model.parameters() if p.requires_grad} != {id(p) for p in head.parameters()}:
        raise ValueError('Trainable parameters extend beyond the new classification head.')
    directory = require_new_run(data['root'], config['model_name'], config['run_name'])
    directory.mkdir(parents=True, exist_ok=False)
    write_json_new(directory / 'config.json', config)
    model.to(device)
    train_loader, validation_loader = make_loaders(data, transform, config)
    optimizer = torch.optim.Adam(head.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    loss_fn = training_criterion(config, device)
    history = []
    best_score, best_epoch, best_state = -1.0, None, None
    synchronize(device)
    training_start = time.perf_counter()
    for epoch in range(1, config['epochs'] + 1):
        epoch_start = time.perf_counter()
        # Freezing gradients alone would not freeze BatchNorm running statistics.
        model.eval()
        head.train()
        loss_sum, correct, seen = 0.0, 0, 0
        loss_denominator, unweighted_sum = 0.0, 0.0
        for images, labels, _ in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = loss_fn(logits, labels)
            if not torch.isfinite(loss).item():
                raise FloatingPointError('Non-finite training loss; this run remains incomplete.')
            loss.backward()
            optimizer.step()
            # Weighted CE divides by the target-class weight sum, not batch size.
            denominator = len(labels) if loss_fn.weight is None else loss_fn.weight[labels].sum().item()
            loss_sum += loss.item() * denominator
            loss_denominator += denominator
            if loss_fn.weight is not None:
                unweighted_sum += nn.functional.cross_entropy(logits.detach(), labels, reduction='sum').item()
            correct += (logits.argmax(dim=1) == labels).sum().item()
            seen += len(labels)
        metrics, _ = evaluate_validation(model, validation_loader, device, data['classes'])
        if not np.isfinite(metrics['validation_loss']):
            raise FloatingPointError('Non-finite validation loss; this run remains incomplete.')
        if metrics['macro_f1'] > best_score:
            best_score, best_epoch = metrics['macro_f1'], epoch
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
        synchronize(device)
        row = {'epoch': epoch, 'train_loss': loss_sum / loss_denominator, 'train_accuracy': correct / seen,
               'validation_loss': metrics['validation_loss'], 'validation_accuracy': metrics['accuracy'],
               'validation_macro_f1': metrics['macro_f1'], 'epoch_seconds': time.perf_counter() - epoch_start}
        if loss_fn.weight is not None:
            row['train_unweighted_loss'] = unweighted_sum / seen
        history.append(row)
        print(f"Epoch {epoch}/{config['epochs']} | train loss {row['train_loss']:.4f} | "
              f"val loss {row['validation_loss']:.4f} | val accuracy {row['validation_accuracy']:.4f} | "
              f"val macro-F1 {row['validation_macro_f1']:.4f} | {row['epoch_seconds']:.1f}s", flush=True)
    synchronize(device)
    training_seconds = time.perf_counter() - training_start
    model.load_state_dict(best_state)
    metrics, predictions = evaluate_validation(model, validation_loader, device, data['classes'])
    metrics.update({'best_epoch': best_epoch, 'training_seconds': training_seconds,
                    'training_time_definition': 'all five training epochs, epoch validation and in-memory best-state copies; excludes weight download, final evaluation, latency and saving',
                    'latency': benchmark_latency(model, validation_loader.dataset, device, config['latency_protocol'])})
    if not np.isclose(metrics['macro_f1'], best_score, atol=1e-12, rtol=0):
        raise RuntimeError('Reloaded best checkpoint did not reproduce its validation macro-F1.')
    # Each artifact is created once. A failed run has no complete.json and cannot be ranked.
    with (directory / 'best_model.pt').open('xb') as stream:
        torch.save({'model_state_dict': best_state, 'model_name': config['model_name'],
                    'classes': data['classes'], 'best_epoch': best_epoch,
                    'manifest_sha256': data['manifest_sha256']}, stream)
    pd.DataFrame(history).to_csv(directory / 'history.csv', index=False, mode='x')
    predictions.to_csv(directory / 'validation_predictions.csv', index=False, mode='x')
    write_json_new(directory / 'metrics.json', metrics)
    names = ['config.json', 'metrics.json', 'history.csv', 'validation_predictions.csv', 'best_model.pt']
    write_json_new(directory / 'complete.json', {
        'status': 'complete', 'completed_at_utc': datetime.now(timezone.utc).isoformat(),
        'sha256': {name: file_sha256(directory / name) for name in names},
    })
    print(f"Saved completed run: {directory}", flush=True)
    return load_run(data['root'], config['model_name'], config['run_name'], data, require_checkpoint=True)


def restore_checkpoint(model, run, data, device):
    """Load a local completed state; model construction must use weights=None."""
    path = run['directory'] / 'best_model.pt'
    complete = json.loads((run['directory'] / 'complete.json').read_text())
    if file_sha256(path) != complete['sha256']['best_model.pt']:
        raise ValueError('Checkpoint checksum differs from the completed run.')
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if (checkpoint['model_name'] != run['config']['model_name'] or checkpoint['classes'] != data['classes']
            or checkpoint['manifest_sha256'] != data['manifest_sha256']
            or checkpoint['best_epoch'] != run['metrics']['best_epoch']):
        raise ValueError('Checkpoint identity differs from the saved experiment.')
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.to(device).eval()
    return model
