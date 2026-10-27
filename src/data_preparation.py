"""Read-only TrashNet inspection. This module deliberately has no split operation.

Run with the project's scientific Python interpreter. Source images and the ZIP
are only opened for reading. Generated tables and review thumbnails live outside
data/. Perceptual hashes flag candidates; they do not establish object identity.
"""

from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image, ImageDraw, ImageFont


CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png'}
EXPECTED_COUNTS = dict(zip(CLASSES, [403, 501, 410, 594, 482, 137]))
REVIEW_ARCHIVE = Path('manifests/archive/data_review_v1')
APPROVED_CONFLICT_PAIRS = [
    ('glass/glass115.jpg', 'metal/metal91.jpg'),
    ('glass/glass176.jpg', 'plastic/plastic152.jpg'),
    ('glass/glass389.jpg', 'plastic/plastic332.jpg'),
]


def get_project_root():
    """Locate this repository from the module file, independently of kernel cwd."""
    root = Path(__file__).resolve().parents[1]
    if not (root / 'manifests/integrity_before.json').is_file():
        raise FileNotFoundError(f'WasteWise project marker is missing under {root}.')
    return root


def file_sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def source_checksums(source_dir):
    """Fingerprint all source files, including non-images, in stable path order."""
    source_dir = Path(source_dir)
    rows = []
    for path in sorted(source_dir.rglob('*')):
        if path.is_symlink():
            raise ValueError(f'Source symlink requires review: {path}')
        if path.is_file():
            rows.append({
                'path': path.relative_to(source_dir).as_posix(),
                'bytes': path.stat().st_size,
                'sha256': file_sha256(path),
            })
    return rows


def csv_bytes(rows, fieldnames):
    """Canonical CSV: UTF-8, explicit column order, LF line endings."""
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode('utf-8')


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def load_dataset_records(root):
    """Read the final eligibility decision; never create assignments or files."""
    root = Path(root)
    metadata = json.loads((root / 'manifests/dataset_metadata.json').read_text())
    inventory = read_csv(root / 'manifests/image_inventory.csv')
    exclusions = read_csv(root / 'manifests/duplicate_review.csv')
    for name, expected in metadata['authoritative_sha256'].items():
        if file_sha256(root / 'manifests' / name) != expected:
            raise ValueError(f'Authoritative record changed: {name}')
    paths = {row['path'] for row in inventory}
    excluded = {row['path'] for row in exclusions}
    expected_excluded = {path for pair in APPROVED_CONFLICT_PAIRS for path in pair}
    if len(paths) != len(inventory) or excluded != expected_excluded or len(exclusions) != 6:
        raise ValueError('Inventory or six-image exclusion decision is inconsistent.')
    if not excluded <= paths or any(row['decision'] != 'exclude' for row in exclusions):
        raise ValueError('Invalid exclusion record.')
    if any(row['original_label'] != next(item['label'] for item in inventory
                                        if item['path'] == row['path']) for row in exclusions):
        raise ValueError('An exclusion label differs from the source inventory.')
    eligible = [row for row in inventory if row['path'] not in excluded]
    counts = []
    for label in CLASSES:
        source_count = sum(row['label'] == label for row in inventory)
        excluded_count = sum(row['original_label'] == label for row in exclusions)
        counts.append({'class': label, 'source_images': source_count,
                       'excluded_images': excluded_count,
                       'eligible_images': source_count - excluded_count})
    if (len(inventory), len(exclusions), len(eligible)) != (2527, 6, 2521):
        raise ValueError('Final inclusion counts differ from 2527 / 6 / 2521.')
    if counts != metadata['class_counts'] or metadata['inclusion_status'] != 'final':
        raise ValueError('Dataset metadata and eligibility records disagree.')
    return {'metadata': metadata, 'inventory': inventory, 'exclusions': exclusions,
            'eligible': eligible, 'class_counts': counts}


def load_related_review(root):
    """Summarize archived visual observations without adopting group assignments."""
    root = Path(root)
    archive = root / REVIEW_ARCHIVE
    metadata = json.loads((root / 'manifests/dataset_metadata.json').read_text())
    for name, expected in metadata['related_image_review']['evidence_sha256'].items():
        if file_sha256(archive / name) != expected:
            raise ValueError(f'Archived review evidence changed: {name}')
    groups = read_csv(archive / 'related_groups.csv')
    candidates = read_csv(archive / 'near_duplicate_review.csv')
    assessments = json.loads((archive / 'group_review_assessments.json').read_text())
    by_group = {row['proposed_group_id']: row for row in assessments}
    if len(by_group) != len(assessments) or set(by_group) != {row['proposed_group_id'] for row in groups}:
        raise ValueError('Archived group memberships and observations disagree.')
    if len({row['path'] for row in groups}) != len(groups):
        raise ValueError('An image belongs to multiple archived proposals.')
    if any(row['visual_status'] not in {'supported', 'uncertain'} for row in assessments):
        raise ValueError('Unknown visual-review category.')
    eligible_paths = {row['path'] for row in load_dataset_records(root)['eligible']}
    referenced_paths = {row['path'] for row in groups} | {
        row[key] for row in candidates for key in ('path_a', 'path_b')}
    if not referenced_paths <= {row['path'] for row in read_csv(root / 'manifests/image_inventory.csv')}:
        raise ValueError('Archived evidence refers to an unknown source image.')
    if not {row['path'] for row in groups} <= eligible_paths:
        raise ValueError('An excluded image appears in a related-image proposal.')
    supported = {row['path'] for row in groups
                 if by_group[row['proposed_group_id']]['visual_status'] == 'supported'}
    uncertain = {row['path'] for row in groups
                 if by_group[row['proposed_group_id']]['visual_status'] == 'uncertain'}
    unresolved = [row for row in candidates if row['visual_screen'] == 'needs_further_review']
    pair_paths = {row[key] for row in unresolved for key in ('path_a', 'path_b')}
    summary = [
        {'relationship': 'Visually supported proposals', 'groups_or_pairs': sum(
            row['visual_status'] == 'supported' for row in assessments), 'distinct_images': len(supported)},
        {'relationship': 'Uncertain proposals', 'groups_or_pairs': sum(
            row['visual_status'] == 'uncertain' for row in assessments), 'distinct_images': len(uncertain)},
        {'relationship': 'Unresolved candidate pairs', 'groups_or_pairs': len(unresolved),
         'distinct_images': len(pair_paths)},
        {'relationship': 'Combined unresolved coverage (overlap removed)', 'groups_or_pairs': None,
         'distinct_images': len(uncertain | pair_paths)},
    ]
    return {'summary': summary, 'groups': groups, 'assessments': assessments,
            'candidates': candidates, 'unresolved_pairs': unresolved}


def artifact_path(root, relative_path):
    """Permit outputs only under manifests/ or results/data_review/."""
    root = Path(root).resolve()
    path = root / relative_path
    resolved = path.resolve()
    allowed = [root / 'manifests', root / 'results/data_review']
    if not any(resolved.is_relative_to(directory) for directory in allowed):
        raise ValueError(f'Output must be an audit artifact outside data/: {path}')
    if resolved.is_relative_to((root / 'data').resolve()):
        raise ValueError('Source data is read-only.')
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_artifact(root, relative_path, data):
    """Create a new artifact; refuse to overwrite any existing file."""
    path = artifact_path(root, relative_path)
    with path.open('xb') as stream:
        stream.write(data)
    return path


def write_csv(root, relative_path, rows, fieldnames):
    return write_artifact(root, relative_path, csv_bytes(rows, fieldnames))


def write_json(root, relative_path, value):
    data = (json.dumps(value, indent=2, allow_nan=False) + '\n').encode('utf-8')
    return write_artifact(root, relative_path, data)


def verify_source(root):
    """Recompute hashes against the saved before snapshot; do not write files."""
    root = Path(root)
    before = json.loads((root / 'manifests/integrity_before.json').read_text())
    inventory_path = root / before['inventory_file']
    if file_sha256(inventory_path) != before['inventory_sha256']:
        raise ValueError('The original checksum inventory has changed.')
    rows = source_checksums(root / before['source_root'])
    after_bytes = csv_bytes(rows, ['path', 'bytes', 'sha256'])
    after_hash = hashlib.sha256(after_bytes).hexdigest()
    expected = {row['path']: row for row in read_csv(inventory_path)}
    actual = {row['path']: row for row in rows}
    changed = sorted(path for path in expected.keys() & actual.keys()
                     if expected[path]['sha256'] != actual[path]['sha256']
                     or int(expected[path]['bytes']) != actual[path]['bytes'])
    archive_path = root / before['archive']['path']
    archive_hash = file_sha256(archive_path) if archive_path.is_file() else None
    directories = sorted(p.relative_to(root / before['source_root']).as_posix()
                         for p in (root / before['source_root']).rglob('*') if p.is_dir())
    result = {
        'archive_sha256_before': before['archive']['sha256'],
        'archive_sha256_after': archive_hash,
        'archive_unchanged': archive_hash == before['archive']['sha256'],
        'inventory_sha256_before': before['inventory_sha256'],
        'inventory_sha256_after': after_hash,
        'source_file_count_before': before['source_file_count'],
        'source_file_count_after': len(rows),
        'source_files_unchanged': after_hash == before['inventory_sha256'],
        'source_directories_unchanged': directories == before['source_directories'],
        'added_paths': sorted(actual.keys() - expected.keys()),
        'missing_paths': sorted(expected.keys() - actual.keys()),
        'changed_paths': changed,
    }
    result['original_data_unchanged'] = all(result[key] for key in [
        'archive_unchanged', 'source_files_unchanged', 'source_directories_unchanged'])
    return result, rows


def _bits_to_hex(bits):
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return f'{value:016x}'


def perceptual_hashes(image):
    """64-bit horizontal difference hash and 63-bit low-frequency DCT hash.

    DCT screening uses a 32x32 grayscale thumbnail and the top-left 8x8
    orthonormal DCT-II coefficients, excluding DC, thresholded at their median.
    These are screening heuristics, not a trained model or a duplicate verdict.
    No thumbnail is written back to the source directory.
    """
    gray = image.convert('L')
    small = np.asarray(gray.resize((9, 8), Image.Resampling.LANCZOS), dtype=float)
    dhash = _bits_to_hex((small[:, 1:] > small[:, :-1]).ravel())
    pixels = np.asarray(gray.resize((32, 32), Image.Resampling.LANCZOS), dtype=float)
    frequencies = np.arange(8)[:, None]
    positions = np.arange(32)[None, :]
    basis = np.cos(np.pi * (positions + 0.5) * frequencies / 32) * np.sqrt(2 / 32)
    basis[0] /= np.sqrt(2)
    coefficients = (basis @ pixels @ basis.T).ravel()[1:]
    phash = _bits_to_hex(coefficients > np.median(coefficients))
    return dhash, phash


def inspect_images(source_dir):
    """Fully decode each image and report errors without dropping source files."""
    source_dir = Path(source_dir)
    folders = sorted(path.name for path in source_dir.iterdir() if path.is_dir())
    if folders != sorted(CLASSES):
        raise ValueError(f'Unexpected class folders: {folders}')
    records, errors = [], []
    for class_name in CLASSES:
        for path in sorted((source_dir / class_name).iterdir()):
            if path.is_symlink() or path.is_dir():
                raise ValueError(f'Unexpected image path: {path}')
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            try:
                with Image.open(path) as image:
                    image.load()
                    header = f'{image.mode}|{image.width}x{image.height}|'.encode()
                    pixel_hash = hashlib.sha256(header + image.tobytes()).hexdigest()
                    dhash, phash = perceptual_hashes(image)
                    records.append({
                        'path': path.relative_to(source_dir).as_posix(),
                        'label': class_name,
                        'bytes': path.stat().st_size,
                        'file_sha256': file_sha256(path),
                        'pixel_sha256': pixel_hash,
                        'width': image.width, 'height': image.height,
                        'mode': image.mode, 'format': image.format,
                        'dhash64': dhash, 'phash63': phash,
                    })
            except (OSError, ValueError) as error:
                errors.append({'path': path.relative_to(source_dir).as_posix(), 'error': str(error)})
    return records, errors


def exact_duplicate_groups(records):
    by_pixels = defaultdict(list)
    for record in records:
        by_pixels[record['pixel_sha256']].append(record)
    return [group for group in by_pixels.values() if len(group) > 1]


def find_candidates(records, phash_threshold=8, dhash_threshold=4):
    """Screen every pair across all six classes; exclude exact pixel matches.

    The rule is pHash distance <= 8 OR dHash distance <= 4. This intentionally
    produces false positives. Dissimilar views of one object can still be missed.
    """
    hashes = [(int(row['phash63'], 16), int(row['dhash64'], 16)) for row in records]
    candidates = []
    for i, left in enumerate(records):
        for j in range(i + 1, len(records)):
            right = records[j]
            if left['pixel_sha256'] == right['pixel_sha256']:
                continue
            pd = (hashes[i][0] ^ hashes[j][0]).bit_count()
            dd = (hashes[i][1] ^ hashes[j][1]).bit_count()
            if pd <= phash_threshold or dd <= dhash_threshold:
                candidates.append({
                    'path_a': left['path'], 'path_b': right['path'],
                    'label_a': left['label'], 'label_b': right['label'],
                    'phash_distance': pd, 'dhash_distance': dd,
                    'same_folder_label': left['label'] == right['label'],
                })
    candidates.sort(key=lambda row: (row['phash_distance'], row['dhash_distance'], row['path_a'], row['path_b']))
    return [{'candidate_id': f'candidate_{i:04d}', **row} for i, row in enumerate(candidates, 1)]


def natural_path_key(path):
    return [int(part) if part.isdigit() else part for part in re.split(r'(\d+)', path)]


def contact_sheet(root, paths, relative_output, title, columns=8, cell_width=192, cell_height=170):
    """Create annotated review thumbnails outside data/; originals are read only."""
    rows = (len(paths) + columns - 1) // columns
    canvas = Image.new('RGB', (columns * cell_width, 44 + rows * cell_height), 'white')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=14)
    draw.text((10, 10), title, font=font, fill='black')
    source = (Path(root) / 'data/dataset-resized').resolve()
    for index, relative in enumerate(paths):
        path = source / relative
        if not path.resolve().is_relative_to(source):
            raise ValueError(f'Image path escapes source directory: {relative}')
        x, y = (index % columns) * cell_width, 44 + (index // columns) * cell_height
        with Image.open(path) as image:
            thumbnail = image.convert('RGB')
            thumbnail.thumbnail((cell_width - 8, cell_height - 30))
            canvas.paste(thumbnail, (x + (cell_width - thumbnail.width) // 2, y))
        draw.text((x + 4, y + cell_height - 26), relative, font=font, fill='black')
    buffer = io.BytesIO()
    canvas.save(buffer, format='PNG')
    return write_artifact(root, relative_output, buffer.getvalue())


def candidate_contact_sheets(root, candidates):
    """Show adjacent candidate pairs; page numbers preserve the CSV ordering."""
    for start in range(0, len(candidates), 20):
        page = start // 20 + 1
        paths = [row[key] for row in candidates[start:start+20]
                 for key in ['path_a', 'path_b']]
        contact_sheet(
            root, paths, f'results/data_review/candidates/candidates_{page:02d}.png',
            f'Candidates {start+1}-{min(start+20,len(candidates))}; adjacent pairs; not automatic group assignments',
            columns=8, cell_width=220, cell_height=190)


def run_audit(root):
    """Generate evidence for the six final manifest-only exclusions. Never create a split."""
    root = Path(root)
    integrity, _ = verify_source(root)
    if not integrity['original_data_unchanged']:
        raise ValueError('Source differs from its pre-change snapshot; stop for review.')
    records, errors = inspect_images(root / 'data/dataset-resized')
    if errors:
        raise ValueError(f'Unreadable images require review: {errors}')
    if Counter(row['label'] for row in records) != EXPECTED_COUNTS:
        raise ValueError('Image counts differ from the existing audit.')
    groups = exact_duplicate_groups(records)
    actual_pairs = {tuple(sorted(row['path'] for row in group)) for group in groups}
    expected_pairs = {tuple(sorted(pair)) for pair in APPROVED_CONFLICT_PAIRS}
    if actual_pairs != expected_pairs:
        raise ValueError('Exact duplicate findings differ from the documented exclusions.')
    write_csv(root, 'manifests/image_inventory.csv', records, list(records[0]))
    decisions = []
    records_by_path = {row['path']: row for row in records}
    for number, pair in enumerate(APPROVED_CONFLICT_PAIRS, 1):
        for path in pair:
            decisions.append({
                'path': path, 'original_label': records_by_path[path]['label'],
                'exact_group_id': f'exact_{number:03d}',
                'pixel_sha256': records_by_path[path]['pixel_sha256'],
                'decision': 'exclude',
                'reason': 'Identical decoded image has conflicting source folder labels; exclude both copies from all experimental subsets.',
                'inclusion_status': 'final',
                'source_action': 'none; original file preserved',
            })
    write_csv(root, REVIEW_ARCHIVE / 'duplicate_review.regenerated.csv', decisions, list(decisions[0]))
    candidates = find_candidates(records)
    candidate_fields = ['candidate_id', 'path_a', 'path_b', 'label_a', 'label_b', 'phash_distance', 'dhash_distance', 'same_folder_label']
    write_csv(root, REVIEW_ARCHIVE / 'near_duplicate_candidates.csv', candidates, candidate_fields)
    candidate_contact_sheets(root, candidates)
    sheet_index = []
    for class_name in CLASSES:
        paths = sorted([row['path'] for row in records if row['label'] == class_name], key=natural_path_key)
        for start in range(0, len(paths), 80):
            page = start // 80 + 1
            output = f'results/data_review/contact_sheets/{class_name}_{page:02d}.png'
            contact_sheet(root, paths[start:start+80], output, f'{class_name}: images {start+1}-{min(start+80,len(paths))}; visual screening only')
            sheet_index.append({'sheet': output, 'class': class_name, 'image_count': len(paths[start:start+80]), 'review_status': 'pending'})
    write_csv(root, REVIEW_ARCHIVE / 'contact_sheet_index.csv', sheet_index, list(sheet_index[0]))
    return {'decoded_images': len(records), 'unreadable_images': len(errors),
            'exact_conflicting_groups': len(groups), 'excluded_images': len(decisions),
            'eligible_images': len(records)-len(decisions),
            'near_duplicate_candidate_pairs': len(candidates), 'contact_sheets': len(sheet_index),
            'split_created': False}


def prepare_review_tables(root):
    """Reproduce detailed review tables in the archive; never create a split.

    Existing files are never overwritten. Visual proposals do not establish
    physical-object identity. Eligibility is fixed independently of grouping.
    """
    root = Path(root)
    records = read_csv(root / 'manifests/image_inventory.csv')
    by_path = {row['path']: row for row in records}
    candidates = read_csv(root / REVIEW_ARCHIVE / 'near_duplicate_candidates.csv')
    decisions = json.loads((root / REVIEW_ARCHIVE / 'visual_screen_decisions.json').read_text())
    overview = json.loads((root / REVIEW_ARCHIVE / 'overview_group_proposals.json').read_text())
    proposed = set(decisions['proposed_related_candidate_numbers'])
    uncertain = decisions['uncertain_candidate_notes']
    envelope_numbers = [54, 110, 119, 132, 143, 255, 277, 322, 329, 347,
                        361, 383, 414, 493, 497, 498, 552, 557, 560, 582]
    envelopes = {f'paper/paper{number}.jpg' for number in envelope_numbers}
    reviews, links = [], []
    for number, row in enumerate(candidates, 1):
        a, b = row['path_a'], row['path_b']
        if number in proposed:
            status = 'propose_keep_together'
            reason = 'Matching distinctive printing, folds, object structure or scene in the displayed views; conservative grouping proposal.'
            links.append(([a, b], row['candidate_id']))
        elif (str(number) in uncertain or (a in envelopes and b in envelopes)
              or (row['label_a'] == row['label_b'] == 'cardboard')):
            status = 'needs_further_review'
            reason = uncertain.get(str(number),
                'Low-detail cardboard or envelope views; physical identity cannot be established from thumbnails.')
        else:
            status = 'no_link_identified_in_screen'
            reason = 'Different visible object details or printed content; no link proposed in this thumbnail screen. This is not proof of independence.'
        reviews.append({**row, 'visual_screen': status, 'reason': reason,
            'evidence_sheet': f'results/data_review/candidates/candidates_{(number-1)//20+1:02d}.png',
            'review_method': 'visual comparison of thumbnails', 'grouping_status': 'pending'})
    for proposal in overview:
        links.append((proposal['paths'], proposal['proposal_id']))

    # Merge overlapping proposed links so a path never has two proposed groups.
    parent = {path: path for paths, _ in links for path in paths}
    if not set(parent).issubset(by_path):
        raise ValueError('A proposed group references an unknown source image.')

    def find(path):
        while parent[path] != path:
            parent[path] = parent[parent[path]]
            path = parent[path]
        return path

    for paths, _ in links:
        for path in paths[1:]:
            parent[find(path)] = find(paths[0])
    members = defaultdict(list)
    for path in sorted(parent):
        members[find(path)].append(path)
    group_rows, membership = [], {}
    for paths in sorted(members.values()):
        group_id = 'proposed_' + hashlib.sha256('\n'.join(paths).encode()).hexdigest()[:12]
        refs = sorted(ref for linked_paths, ref in links if set(linked_paths) & set(paths))
        for path in paths:
            membership[path] = group_id
            group_rows.append({'path': path, 'original_label': by_path[path]['label'],
                'proposed_group_id': group_id, 'group_size': len(paths),
                'evidence_refs': ';'.join(refs), 'status': 'proposed; grouping unresolved',
                'scope': 'partial related-image grouping; not a split manifest'})
    exclusions = {row['path']: row for row in read_csv(root / 'manifests/duplicate_review.csv')}
    inclusion = []
    for row in records:
        path = row['path']
        excluded = path in exclusions
        inclusion.append({'path': path, 'original_label': row['label'],
            'decision': 'exclude' if excluded else 'include',
            'reason': exclusions[path]['reason'] if excluded else 'Eligible image; grouping uncertainty does not change inclusion.',
            'proposed_group_id': membership.get(path, ''),
            'inclusion_status': 'final',
            'split': 'unassigned'})
    counts = []
    for class_name in CLASSES:
        total = sum(row['label'] == class_name for row in records)
        excluded = sum(row['original_label'] == class_name for row in exclusions.values())
        counts.append({'class': class_name, 'source_images': total,
                       'excluded_images': excluded, 'eligible_images': total-excluded})
    issues = [
        ('cardboard', 'Repeated printed boxes and unmarked close-ups; generic brown texture cannot identify an object.'),
        ('glass', 'Repeated jars, bottle labels, necks and bases. Some top/bottom views cannot be linked to a labelled bottle from thumbnails.'),
        ('metal', 'Cans before/after crushing, foil, lids and can interiors; some relationships remain uncertain.'),
        ('paper', 'Repeated advertisements, circular pages and plain/torn envelopes; shared printing alone does not establish identity.'),
        ('plastic', 'Repeated water bottles, milk containers, caps and clamshell packages; shared product appearance alone does not establish identity.'),
        ('trash', 'Repeated wrappers, food cups and brown tissue views. Several crumpled/torn forms lack reliable object identifiers.'),
    ]
    issue_rows = [{'class': label, 'finding': finding, 'status': 'unresolved relationship',
                   'required_action': 'Document a bounded grouping policy for credible leakage risks; retain all eligible images.'}
                  for label, finding in issues]
    sheet_reviews = [{**row, 'review_status': 'visual comparison complete; object identities not independently verified'}
                     for row in read_csv(root / REVIEW_ARCHIVE / 'contact_sheet_index.csv')]
    outputs = {
        'near_duplicate_review.csv': reviews,
        'related_groups.csv': group_rows,
        'inclusion_review.csv': inclusion,
        'class_review_counts.csv': counts,
        'related_review_open_issues.csv': issue_rows,
        'contact_sheet_review.csv': sheet_reviews,
    }
    for name, rows in outputs.items():
        write_csv(root, REVIEW_ARCHIVE / name, rows, list(rows[0]))
    summary = {
        'schema_version': 1, 'status': 'inclusion final; grouping and split pending',
        'source_images': len(records), 'unreadable_images': 0,
        'excluded_images': len(exclusions), 'additional_exclusions': 0,
        'eligible_images': len(records)-len(exclusions),
        'class_counts': counts,
        'exact_duplicate_groups': 3, 'exact_groups_with_label_conflicts': 3,
        'candidate_pair_count': len(candidates),
        'candidate_rule': '63-bit DCT hash Hamming distance <= 8 OR 64-bit horizontal difference hash distance <= 4; exact pixel duplicates omitted',
        'candidate_screen_counts': dict(Counter(row['visual_screen'] for row in reviews)),
        'all_candidate_pairs_thumbnail_screened': True,
        'all_source_images_thumbnail_screened': True, 'contact_sheet_count': len(sheet_reviews),
        'overview_group_proposals': len(overview),
        'merged_proposed_groups': len(members), 'images_in_proposed_groups': len(membership),
        'grouping_complete': False, 'inclusion_status': 'final',
        'related_groups_approved': False, 'split_ready': False, 'split_created': False,
        'future_split_plan': {'train': 0.70, 'validation': 0.15, 'test': 0.15, 'seed': 42},
        'limitations': [
            'Perceptual hash thresholds are screening heuristics, not measured duplicate-detection recall.',
            'Visual screening of thumbnails does not establish physical object identity or exhaustive grouping.',
            'Repeated product/printed-artifact families were observed beyond hash candidates; proposed groups are partial.',
            'No final membership or train/validation/test assignments are approved or generated.',
        ],
    }
    write_json(root, REVIEW_ARCHIVE / 'data_review_summary.json', summary)
    return summary


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['audit', 'review', 'verify'])
    arguments = parser.parse_args()
    project_root = get_project_root()
    if arguments.action == 'audit':
        print(json.dumps(run_audit(project_root), indent=2))
    elif arguments.action == 'review':
        print(json.dumps(prepare_review_tables(project_root), indent=2))
    else:
        report, _ = verify_source(project_root)
        print(json.dumps(report, indent=2))
        if not report['original_data_unchanged']:
            raise SystemExit(1)
