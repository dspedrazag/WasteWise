"""Checks and comparison for one EfficientNet-B0 loss-only experiment.

Importing this module does not construct models, train, or write result files.
"""

import ast
import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from . import experiment_results as results
from .data_preparation import file_sha256


MODEL_NAME = 'efficientnet_b0'
BASELINE_RUN = 'baseline_seed42_5epochs'
WEIGHTED_RUN = 'weighted_ce_seed42_5epochs'
WEIGHTED_LOSS = 'ClassWeightedCrossEntropyLoss'


def class_weight_plan(data):
    """Use training labels only; preserve the approved class-to-index order."""
    counts = data['train']['label'].value_counts().reindex(data['classes'], fill_value=0)
    if counts.le(0).any() or int(counts.sum()) != len(data['train']):
        raise ValueError('Every training label must belong to a nonempty approved class.')
    return {
        'method': 'balanced_inverse_training_frequency',
        'formula': 'N_train / (N_classes * count_c)',
        'classes': list(data['classes']), 'counts': counts.astype(int).tolist(),
        'n_train': len(data['train']),
        'weights': (len(data['train']) / (len(data['classes']) * counts)).tolist(),
        'reduction': 'mean', 'validation_loss': 'unweighted_cross_entropy',
    }


def weight_table(data):
    plan = class_weight_plan(data)
    return pd.DataFrame({'Training images': plan['counts'], 'Loss weight': plan['weights']},
                        index=pd.Index(plan['classes'], name='Category'))


def baseline_reference(baseline):
    return {'model_name': MODEL_NAME, 'run_name': BASELINE_RUN,
            'config_sha256': file_sha256(baseline['directory'] / 'config.json'),
            'complete_sha256': file_sha256(baseline['directory'] / 'complete.json')}


def validate_controlled_config(data, baseline, config):
    """Reject every baseline-setting change except the training loss itself."""
    original = baseline['config']
    results.comparison_table([baseline])  # Verify the reference is an unweighted baseline.
    if original['model_name'] != MODEL_NAME or original['run_name'] != BASELINE_RUN:
        raise ValueError('Expected the original EfficientNet-B0 baseline.')
    for key in ['classes', 'manifest_sha256', 'train_rows_sha256', 'validation_rows_sha256']:
        if original[key] != data[key]:
            raise ValueError(f'Baseline differs from approved data: {key}')
    if config.get('loss') != WEIGHTED_LOSS or config.get('run_name') != WEIGHTED_RUN:
        raise ValueError('Use the approved weighted-loss name and a separate run directory.')
    if config.get('loss_weighting') != class_weight_plan(data):
        raise ValueError('Weights must be computed only from the current approved training labels.')
    if config.get('baseline_reference') != baseline_reference(baseline):
        raise ValueError('The recorded baseline reference changed.')
    # These fields describe the new run or its changed implementation, not hyperparameters.
    provenance = {'run_name', 'module_sha256', 'notebook', 'notebook_sha256', 'created_at_utc'}
    changed = [key for key in original if key not in provenance | {'loss'}
               and config.get(key) != original[key]]
    if changed:
        raise ValueError('Loss-only controls differ from the baseline: ' + ', '.join(changed))
    allowed_additions = {'loss_weighting', 'baseline_reference', 'initialization_verification'}
    if set(config) - set(original) - allowed_additions:
        raise ValueError('Unexpected experimental configuration fields.')
    initialization = config.get('initialization_verification', {})
    checks = ['builder_matches_baseline', 'all_initial_tensors_equal',
              'frozen_features_match_baseline_checkpoint', 'shuffle_indices_equal']
    if not all(initialization.get(key) is True for key in checks):
        raise ValueError('The initialization and data-order checks must pass before training.')
    return True


def builder_ast(path):
    """Read only the architecture definition; never execute another notebook."""
    notebook = json.loads(Path(path).read_text())
    definitions = [node for cell in notebook['cells'] if cell['cell_type'] == 'code'
                   for node in ast.parse(''.join(cell['source'])).body
                   if isinstance(node, ast.FunctionDef) and node.name == 'build_model']
    if len(definitions) != 1:
        raise ValueError('Expected exactly one build_model definition in the notebook.')
    return ast.dump(definitions[0], include_attributes=False)


def state_sha256(state):
    """Hash tensor values, names, shapes and types without serialization metadata."""
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(json.dumps([name, str(array.dtype), list(array.shape)]).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def prepare_experiment(data, baseline, build_model, device, notebook_name):
    """Reconstruct initialization and check controls without inference or optimization."""
    import torch
    from torch import nn
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
    from . import experiment_training as training

    original = baseline['config']
    baseline_source = builder_ast(data['root'] / original['notebook'])
    if builder_ast(data['root'] / notebook_name) != baseline_source:
        raise ValueError('The weighted notebook must retain the baseline model definition.')
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    cached_weights = Path(torch.hub.get_dir()) / 'checkpoints' / Path(urlparse(weights.url).path).name
    if not cached_weights.is_file():
        raise FileNotFoundError(f'Expected existing ImageNet weights at {cached_weights}; no download was attempted.')
    weights_digest = file_sha256(cached_weights)
    if not weights_digest.startswith(cached_weights.stem.rsplit('-', 1)[1]):
        raise ValueError('Cached ImageNet weights failed their published checksum prefix.')

    # Reconstruct the original sequence, including the RNG consumed by the original head.
    training.seed_everything(original['seed'])
    reference = efficientnet_b0(weights=weights)
    reference.requires_grad_(False)
    feature_count = reference.classifier[-1].in_features
    reference.classifier = nn.Linear(feature_count, len(data['classes']))
    reference_state = reference.state_dict()
    reference_digest = state_sha256(reference_state)

    # Leave the global RNG exactly where the original initialization procedure leaves it.
    training.seed_everything(original['seed'])
    model, head, transform, actual_weights = build_model(pretrained=True)
    initial_state = model.state_dict()
    same_initial = (reference_state.keys() == initial_state.keys()
                    and all(torch.equal(reference_state[k], initial_state[k]) for k in reference_state))
    if not same_initial:
        raise ValueError('The revised initial tensors differ from the reconstructed baseline.')
    complete = json.loads((baseline['directory'] / 'complete.json').read_text())
    checkpoint_path = baseline['directory'] / 'best_model.pt'
    if file_sha256(checkpoint_path) != complete['sha256']['best_model.pt']:
        raise ValueError('Baseline checkpoint integrity check failed.')
    saved = torch.load(checkpoint_path, map_location='cpu', weights_only=True)['model_state_dict']
    frozen_names = [name for name in initial_state if not name.startswith('classifier.')]
    frozen_equal = all(name in saved and torch.equal(initial_state[name], saved[name]) for name in frozen_names)
    if not frozen_equal:
        raise ValueError('Frozen features differ from the saved baseline checkpoint.')

    settings = {key: original[key] for key in results.BASELINE}
    settings['loss'] = WEIGHTED_LOSS
    config = training.make_config(data, MODEL_NAME, original['display_name'], WEIGHTED_RUN,
                                  model, head, actual_weights, device, notebook_name, settings)
    # Compare full seeded batch-index sequences without opening any training images.
    reference_loader, _ = training.make_loaders(data, transform, original)
    revised_loader, _ = training.make_loaders(data, transform, config)
    # DataLoader consumes one base-seed draw before its RandomSampler is iterated.
    for loader in [reference_loader, revised_loader]:
        torch.empty((), dtype=torch.int64).random_(generator=loader.generator)
    original_order = list(reference_loader.batch_sampler)
    revised_order = list(revised_loader.batch_sampler)
    order_equal = original_order == revised_order
    verification = {
        'builder_matches_baseline': True, 'builder_ast_sha256': hashlib.sha256(baseline_source.encode()).hexdigest(),
        'all_initial_tensors_equal': same_initial,
        'reconstructed_baseline_state_sha256': reference_digest,
        'revised_initial_state_sha256': state_sha256(initial_state),
        'head_initial_state_sha256': state_sha256(head.state_dict()),
        'torch_rng_sha256': hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
        'frozen_features_match_baseline_checkpoint': frozen_equal,
        'frozen_state_tensors_checked': len(frozen_names),
        'shuffle_indices_equal': order_equal,
        'first_epoch_batch_indices_sha256': hashlib.sha256(json.dumps(revised_order).encode()).hexdigest(),
        'imagenet_weights_sha256': weights_digest,
        'limitation': 'The historical initial head was not saved. Equality is to a seed-42 reconstruction '
                      'of the original initialization, not to an archived initial checkpoint.',
    }
    config.update(loss_weighting=class_weight_plan(data), baseline_reference=baseline_reference(baseline),
                  initialization_verification=verification)
    config['module_sha256']['src/weighted_experiment.py'] = file_sha256(Path(__file__))
    validate_controlled_config(data, baseline, config)
    return model, head, transform, config


def validate_training_start(data, model, head, transform, config, device):
    """Recheck settings and initialization immediately before reserving a run."""
    import torch
    from . import experiment_training as training
    from .experiment_data import load_experiment_data, rows_sha256
    from torchvision.models import EfficientNet_B0_Weights

    current_data = load_experiment_data(data['root'])
    baseline = results.load_run(data['root'], MODEL_NAME, BASELINE_RUN, current_data, require_checkpoint=True)
    validate_controlled_config(current_data, baseline, config)
    for key in ['manifest_sha256', 'train_rows_sha256', 'validation_rows_sha256']:
        if data[key] != current_data[key]:
            raise ValueError('Data changed after notebook setup; restart the kernel.')
    for subset in ['train', 'validation']:
        if rows_sha256(data[subset]) != current_data[f'{subset}_rows_sha256']:
            raise ValueError('In-memory dataset rows changed after setup.')
    if str(device) != config['device']:
        raise ValueError('Training device changed after the control checks.')
    for name, digest in config['module_sha256'].items():
        if file_sha256(data['root'] / name) != digest:
            raise ValueError('Shared code changed after setup; restart the kernel.')
    initialization = config['initialization_verification']
    if state_sha256(model.state_dict()) != initialization['revised_initial_state_sha256']:
        raise ValueError('Model is no longer at the verified initialization; rerun setup.')
    if hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest() != initialization['torch_rng_sha256']:
        raise ValueError('Torch RNG changed after initialization; restart and run sequentially.')
    if {id(p) for p in model.parameters() if p.requires_grad} != {id(p) for p in head.parameters()}:
        raise ValueError('Only the original replacement head may be trainable.')
    info = training.model_information(model, head, EfficientNet_B0_Weights.IMAGENET1K_V1)
    if any(config[key] != value for key, value in info.items()):
        raise ValueError('The model definition or trainable parameters changed after setup.')
    actual_transform = {key: getattr(transform, key) for key in
                        ['resize_size', 'crop_size', 'antialias', 'mean', 'std']}
    actual_transform['interpolation'] = transform.interpolation.value
    if any(config['preprocessing'][key] != value for key, value in actual_transform.items()):
        raise ValueError('The preprocessing transform changed after setup.')
    training.training_criterion(config, device)  # Validate dtype, device and class order before writing files.


def load_weighted_run(data, baseline, require_checkpoint=False):
    run = results.load_run(data['root'], MODEL_NAME, WEIGHTED_RUN, data, require_checkpoint=require_checkpoint)
    validate_controlled_config(data, baseline, run['config'])
    if 'train_unweighted_loss' not in run['history'] or not np.isfinite(run['history']['train_unweighted_loss']).all():
        raise ValueError('The weighted run must retain finite unweighted training-loss diagnostics.')
    return run


def comparison_table(data, baseline, revised):
    """Compare this controlled pair; do not add weighted runs to baseline ranking."""
    validate_controlled_config(data, baseline, revised['config'])
    expected = data['validation'].set_index('path')['label'].sort_index()
    for run in [baseline, revised]:
        observed = run['predictions'].set_index('path')['true_label'].sort_index()
        if not observed.equals(expected.rename('true_label')):
            raise ValueError('The two runs must predict the exact approved validation images.')
    rows = []
    for label, run in [('Unweighted baseline', baseline), ('Weighted cross-entropy', revised)]:
        m = run['metrics']
        trash = next(row for row in m['per_class'] if row['class'] == 'trash')
        rows.append({'Configuration': label, 'Macro-F1': m['macro_f1'], 'Accuracy': m['accuracy'],
                     'Weighted F1': m['weighted_f1'], 'Trash precision': trash['precision'],
                     'Trash recall': trash['recall'], 'Trash F1': trash['f1'],
                     'Validation loss (unweighted)': m['validation_loss'], 'Validation images': m['n_validation'],
                     'Best epoch': m['best_epoch'], 'Training seconds': m['training_seconds'],
                     'Forward median ms': m['latency']['forward']['median_ms'],
                     'End-to-end median ms': m['latency']['end_to_end']['median_ms']})
    return pd.DataFrame(rows).set_index('Configuration')


def interpretation(baseline, revised):
    """Describe measured changes without inventing an improvement or significance."""
    if revised is None:
        return '**Outcome pending.** The weighted-loss model has not been trained or reloaded. '
    old, new = baseline['metrics'], revised['metrics']
    delta = new['macro_f1'] - old['macro_f1']
    direction = 'increased' if delta > 1e-12 else 'decreased' if delta < -1e-12 else 'was unchanged'
    old_trash = next(row for row in old['per_class'] if row['class'] == 'trash')
    new_trash = next(row for row in new['per_class'] if row['class'] == 'trash')
    return (
        f"Validation macro-F1 **{direction}** from {old['macro_f1']:.4f} to {new['macro_f1']:.4f} "
        f"(change {delta:+.4f}). Trash F1 changed from {old_trash['f1']:.4f} to {new_trash['f1']:.4f}; "
        f"accuracy changed from {old['accuracy']:.2%} to {new['accuracy']:.2%}. "
        'This describes the observed validation result, not a statistically established effect. '
        'Inspect precision/recall and other class changes before judging the trade-off. '
        'The comparison uses one seed and only 20 trash validation examples; complete object-level '
        'independence is not guaranteed by the original split. The test set remains unused.'
    )
