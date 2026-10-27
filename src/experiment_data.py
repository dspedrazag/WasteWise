"""Read the approved split; image access is limited to train and validation."""

import hashlib
import json
from pathlib import Path

import pandas as pd
from PIL import Image

from .data_preparation import file_sha256, get_project_root, load_dataset_records


def rows_sha256(rows):
    """Identify examples and labels in a stable order, without reading pixels."""
    text = rows.sort_values('path')[['path', 'label', 'group_id']].to_csv(index=False, lineterminator='\n')
    return hashlib.sha256(text.encode()).hexdigest()


def load_experiment_data(root=None):
    root = Path(root or get_project_root()).resolve()
    records = load_dataset_records(root)
    metadata = json.loads((root / 'manifests/split_metadata.json').read_text())
    manifest = root / 'manifests/split_manifest.csv'
    digest = file_sha256(manifest)
    if digest != metadata['manifest_sha256'] or digest != records['metadata']['split']['manifest_sha256']:
        raise ValueError('The approved split checksum does not match. No split will be regenerated.')
    split = pd.read_csv(manifest, dtype=str)
    if list(split.columns) != ['path', 'label', 'subset', 'group_id'] or split.isna().any().any():
        raise ValueError('Invalid split columns or missing assignments.')
    if not split['path'].is_unique or set(split['path']) != {row['path'] for row in records['eligible']}:
        raise ValueError('Split membership differs from the approved eligible population.')
    labels = {row['path']: row['label'] for row in records['eligible']}
    if not split['path'].map(labels).eq(split['label']).all():
        raise ValueError('Source labels and split labels differ.')
    if set(split['subset']) != {'train', 'validation', 'test'}:
        raise ValueError('Unexpected subset names.')
    if not split.groupby('group_id')['subset'].nunique().eq(1).all():
        raise ValueError('An accepted related-image group crosses subsets.')
    # Keep only the permitted rows in the image-facing interface.
    train = split.loc[split['subset'] == 'train'].sort_values('path').reset_index(drop=True)
    validation = split.loc[split['subset'] == 'validation'].sort_values('path').reset_index(drop=True)
    return {
        'root': root,
        'source_root': root / records['metadata']['source_root'],
        'classes': records['metadata']['class_order'],
        'train': train,
        'validation': validation,
        'allowed_paths': {'train': frozenset(train['path']), 'validation': frozenset(validation['path'])},
        'manifest_sha256': digest,
        'train_rows_sha256': rows_sha256(train),
        'validation_rows_sha256': rows_sha256(validation),
    }


def subset_rows(data, subset):
    if subset not in ('train', 'validation'):
        raise ValueError('Only training and validation images may be accessed during this homework.')
    return data[subset].copy()


def image_path(data, relative_path, subset):
    if subset not in ('train', 'validation'):
        raise ValueError('Only training and validation images may be accessed during this homework.')
    if relative_path not in data['allowed_paths'][subset]:
        raise ValueError(f'Image is not in the approved {subset} subset: {relative_path}')
    path = (data['source_root'] / relative_path).resolve()
    if not path.is_relative_to(data['source_root'].resolve()) or not path.is_file():
        raise FileNotFoundError(f'Expected an existing source image: {relative_path}')
    return path


def open_image(data, relative_path, subset):
    """Return an in-memory RGB image; never save or alter the source."""
    with Image.open(image_path(data, relative_path, subset)) as image:
        return image.convert('RGB')


def dataset_table(data):
    return pd.DataFrame({
        'Training': data['train']['label'].value_counts().reindex(data['classes'], fill_value=0),
        'Validation': data['validation']['label'].value_counts().reindex(data['classes'], fill_value=0),
    }).rename_axis('Category')
