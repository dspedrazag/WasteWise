"""Reproducible, group-aware TrashNet splitting; no model or image processing.

Importing this module writes nothing. ``build_split`` works in memory;
``load_split`` verifies the saved result. Only ``create_split`` writes files,
and it refuses to replace either existing split artifact.
"""

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
from itertools import combinations, product
import json
from pathlib import Path
import platform
import random

from . import data_preparation as data


SUBSETS = ('train', 'validation', 'test')
CONFIG = {
    'seed': 42,
    'ratio_numerators': [14, 3, 3],
    'ratio_denominator': 20,
    'algorithm': 'globally_balanced_class_rounding_then_weighted_whole_group_assignment_v1',
    'grouping_policy': 'recorded_visual_evidence_with_bounded_conservative_links_v1',
}
FIELDS = ['path', 'label', 'subset', 'group_id']

# These choices apply the existing evidence; uncertain identity is not promoted
# to confirmed identity. Commercial design alone is never a grouping criterion.
CONSERVATIVE_PROPOSALS = {
    'G019': 'Compatible irregular fracture shapes under rotation make a same-object link plausible; the fracture correspondence remains incomplete.',
    'G025': 'Compatible tray deformations support a possible handling/view sequence; correspondence across the underside and interior is uncertain.',
    'G028': 'The recorded possible can-deformation sequence is kept together conservatively; this is a handling-sequence precaution, not a brand-based identity claim.',
    'G059': 'Compatible orange seam fragments and folded transparent portions suggest a possible wrapper-handling sequence; a matching tear is unverified.',
    'G077': 'Compatible padded-mailer layers and a torn state suggest a possible handling sequence; full cross-surface correspondence is uncertain.',
    'G091': 'The pack, foil blisters, and tray may belong to one unpacking sequence; shared capture provenance remains unverified.',
    'G095': 'The torn envelope has a plausible front/reverse link and a crumpled view that obscures the tear; keep the possible handling sequence together.',
}
PARTIAL_PROPOSALS = {
    'G044': {
        'paths': ['paper/paper459.jpg', 'paper/paper466.jpg'],
        'evidence': 'candidate_0003 and G044',
        'reason': 'The existing G044 assessment identifies a clear near-identical pair; candidate_0003 identifies its members. The other advertisement images are not grouped on artwork alone.',
    },
    'G093': {
        'paths': ['trash/trash130.jpg', 'trash/trash131.jpg'],
        'evidence': 'G093; bounded check of its three existing member images before splitting',
        'reason': 'The two rear views share the lower-seam deformation, dark mark, and closely corresponding framing. The front view is not linked solely by product branding.',
    },
}
CONSERVATIVE_PAIRS = {
    'candidate_0029': 'The review describes foil at different stages of crumpling, a plausible same-object handling sequence with uncertain identity.',
    'candidate_0160': 'The review describes an envelope and a torn envelope as a possible before/after handling pair; keep the pair together without claiming confirmed identity.',
}


def _digest(payload):
    return hashlib.sha256(payload).hexdigest()


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + '\n').encode()


def input_hashes(root):
    names = ['manifests/image_inventory.csv', 'manifests/duplicate_review.csv',
             'manifests/integrity_before.json', 'manifests/source_checksums_before.csv',
             'src/data_preparation.py', 'src/data_splitting.py']
    names += [str(data.REVIEW_ARCHIVE / name) for name in
              ['related_groups.csv', 'near_duplicate_review.csv', 'group_review_assessments.json']]
    return {name: data.file_sha256(root / name) for name in names}


def grouping_decisions(root, eligible):
    """Record a treatment for all 96 proposals and all 65 unresolved pairs."""
    review = data.load_related_review(root)
    paths_by_proposal = defaultdict(list)
    for row in review['groups']:
        paths_by_proposal[row['proposed_group_id']].append(row['path'])
    links, decisions = [], []
    for assessment in sorted(review['assessments'], key=lambda row: row['review_id']):
        key = assessment['review_id']
        members = sorted(paths_by_proposal[assessment['proposed_group_id']])
        chosen = []
        if assessment['visual_status'] == 'supported':
            treatment, chosen = 'accept_visually_supported', members
            reason = assessment['evidence_or_uncertainty']
        elif key in CONSERVATIVE_PROPOSALS:
            treatment, chosen = 'accept_conservative_uncertain', members
            reason = CONSERVATIVE_PROPOSALS[key]
        elif key in PARTIAL_PROPOSALS:
            treatment, chosen = 'accept_supported_subgroup_only', PARTIAL_PROPOSALS[key]['paths']
            reason = PARTIAL_PROPOSALS[key]['reason']
        else:
            treatment = 'do_not_accept_full_proposal'
            reason = ('Existing evidence does not justify this full membership. Retain all images; '
                      'do not group on shared design or generic appearance. Unresolved identity remains a leakage limitation.')
        if chosen:
            if not set(chosen) <= set(members):
                raise ValueError(f'Subgroup outside proposal {key}.')
            links.append({'evidence_id': key, 'status': treatment,
                          'paths': sorted(chosen), 'reason': reason})
        decisions.append({'evidence_id': key, 'proposed_group_id': assessment['proposed_group_id'],
                          'original_paths': members, 'accepted_paths': sorted(chosen),
                          'treatment': treatment, 'original_observation': assessment['evidence_or_uncertainty'],
                          'reason': reason})
    for row in sorted(review['unresolved_pairs'], key=lambda row: row['candidate_id']):
        key = row['candidate_id']
        paths = sorted([row['path_a'], row['path_b']])
        accepted = key in CONSERVATIVE_PAIRS
        treatment = 'accept_conservative_uncertain' if accepted else 'leave_unresolved_without_group_link'
        reason = CONSERVATIVE_PAIRS.get(key, 'Insufficient evidence beyond generic appearance or hash similarity; retain eligibility and record residual uncertainty.')
        if accepted:
            links.append({'evidence_id': key, 'status': treatment, 'paths': paths, 'reason': reason})
        decisions.append({'evidence_id': key, 'original_paths': paths,
                          'accepted_paths': paths if accepted else [], 'treatment': treatment,
                          'original_observation': row['reason'], 'reason': reason})
    # Connected components ensure overlapping accepted links cannot cross subsets.
    parent = {row['path']: row['path'] for row in eligible}

    def find(path):
        while parent[path] != path:
            parent[path] = parent[parent[path]]
            path = parent[path]
        return path

    for link in links:
        if not set(link['paths']) <= parent.keys():
            raise ValueError('Accepted link includes an ineligible image.')
        first = link['paths'][0]
        for other in link['paths'][1:]:
            parent[find(other)] = find(first)
    components = defaultdict(list)
    for path in sorted(parent):
        components[find(path)].append(path)
    group_for_path, groups = {}, []
    labels = {row['path']: row['label'] for row in eligible}
    for paths in sorted(components.values()):
        kind = 'related' if len(paths) > 1 else 'singleton'
        group_id = kind + '_' + _digest('\n'.join(paths).encode())[:16]
        if len({labels[path] for path in paths}) != 1:
            raise ValueError('A cross-label related group needs an explicit allocation policy.')
        evidence = [link['evidence_id'] for link in links if set(link['paths']) <= set(paths)]
        groups.append({'group_id': group_id, 'label': labels[paths[0]], 'paths': paths,
                       'size': len(paths), 'evidence_ids': evidence})
        group_for_path.update({path: group_id for path in paths})
    if len({row['group_id'] for row in groups}) != len(groups):
        raise ValueError('Group identifier collision.')
    return group_for_path, groups, links, decisions


def class_targets(class_counts, config):
    """Round each class within one image while meeting rounded global totals.

    With six classes and three subsets there are at most 3**6 combinations of
    floor/ceiling row allocations. Choose the least squared rounding error among
    those meeting the global totals; seeded hashes resolve otherwise equal ties.
    """
    weights, denominator = config['ratio_numerators'], config['ratio_denominator']
    total = sum(class_counts.values())
    global_targets = [total * w // denominator for w in weights]
    ranking = sorted(range(3), key=lambda i: (-(total * weights[i] % denominator), i))
    for i in ranking[:total - sum(global_targets)]:
        global_targets[i] += 1
    options = []
    for label in data.CLASSES:
        n = class_counts[label]
        floors = [n * w // denominator for w in weights]
        remainder = n - sum(floors)
        options.append([tuple(value + int(i in extras) for i, value in enumerate(floors))
                        for extras in combinations(range(3), remainder)])
    feasible = [table for table in product(*options)
                if [sum(row[i] for row in table) for i in range(3)] == global_targets]
    if not feasible:
        raise ValueError('No class-balanced floor/ceiling allocation fits the global totals.')

    def score(table):
        error = sum((denominator * n - class_counts[label] * weights[i]) ** 2
                    for label, row in zip(data.CLASSES, table) for i, n in enumerate(row))
        tie = _digest(f"{config['seed']}|{table}".encode())
        return error, tie

    best = min(feasible, key=score)
    return {label: dict(zip(SUBSETS, row)) for label, row in zip(data.CLASSES, best)}, dict(zip(SUBSETS, global_targets))


def build_split(root, config=None):
    """Build rows and evidence deterministically in memory; never read pixels."""
    root = Path(root)
    config = dict(CONFIG if config is None else config)
    if config != CONFIG:
        raise ValueError('This recorded experiment uses exactly the saved seed-42 configuration.')
    records = data.load_dataset_records(root)
    eligible = sorted(records['eligible'], key=lambda row: row['path'])
    group_for_path, groups, links, decisions = grouping_decisions(root, eligible)
    targets, totals = class_targets(Counter(row['label'] for row in eligible), config)
    rng = random.Random(config['seed'])
    assignments = {}
    for label in data.CLASSES:
        remaining = dict(targets[label])
        class_groups = sorted([g for g in groups if g['label'] == label], key=lambda g: g['group_id'])
        rng.shuffle(class_groups)
        class_groups.sort(key=lambda group: -group['size'])
        for group in class_groups:
            # Weighted by unfilled image capacity, so related groups can appear
            # in every subset instead of systematically going into training.
            choices = [name for name in SUBSETS if remaining[name] >= group['size']]
            if not choices:
                raise ValueError('Whole groups cannot fit the remaining class quotas; do not split groups or drop images.')
            draw = rng.randrange(sum(remaining[name] for name in choices))
            for subset in choices:
                if draw < remaining[subset]:
                    assignments[group['group_id']] = subset
                    remaining[subset] -= group['size']
                    break
                draw -= remaining[subset]
        if any(remaining.values()):
            raise ValueError('Class allocation did not fill its target counts.')
    rows = [{'path': row['path'], 'label': row['label'],
             'subset': assignments[group_for_path[row['path']]],
             'group_id': group_for_path[row['path']]} for row in eligible]
    related = [{**group, 'subset': assignments[group['group_id']]} for group in groups if group['size'] > 1]
    decision_paths = {path for decision in decisions for path in decision['original_paths']}
    unresolved_paths = {path for decision in decisions
                        if decision['treatment'] in {'do_not_accept_full_proposal', 'leave_unresolved_without_group_link', 'accept_supported_subgroup_only'}
                        for path in decision['original_paths']}
    subsets_by_path = {row['path']: row['subset'] for row in rows}
    unresolved_crossings = Counter()
    for decision in decisions:
        decision['assigned_subsets'] = sorted({subsets_by_path[path] for path in decision['original_paths']})
        decision['crosses_subsets'] = len(decision['assigned_subsets']) > 1
        if decision['crosses_subsets'] and decision['treatment'] in {
                'do_not_accept_full_proposal', 'leave_unresolved_without_group_link',
                'accept_supported_subgroup_only'}:
            kind = 'candidate_pairs' if decision['evidence_id'].startswith('candidate_') else 'broad_proposals'
            unresolved_crossings[kind] += 1
    details = {
        'config': config, 'class_order': data.CLASSES, 'subset_order': list(SUBSETS),
        'target_counts': totals, 'class_counts': targets,
        'grouping': {'policy': config['grouping_policy'], 'accepted_links': links,
                     'decisions': decisions, 'related_groups': related,
                     'related_group_count': len(related), 'related_image_count': sum(g['size'] for g in related),
                     'singleton_count': sum(g['size'] == 1 for g in groups),
                     'total_group_count': len(groups), 'largest_related_group': max(g['size'] for g in related),
                     'reviewed_coverage_images': len(decision_paths),
                     'unresolved_relationship_coverage_images': len(unresolved_paths),
                     'unaccepted_relationships_crossing_subsets': dict(unresolved_crossings)},
    }
    return rows, details


def validate_rows(rows, eligible, expected_rows, targets):
    """Validate saved assignments against eligibility and reconstructed groups."""
    expected_labels = {row['path']: row['label'] for row in eligible}
    expected_groups = {row['path']: row['group_id'] for row in expected_rows}
    paths = [row['path'] for row in rows]
    group_subsets = defaultdict(set)
    for row in rows:
        group_subsets[row['group_id']].add(row['subset'])
    checks = {
        'exactly_2521_eligible_rows': len(rows) == 2521 and set(paths) == set(expected_labels),
        'unique_image_paths': len(paths) == len(set(paths)),
        'original_labels_preserved': all(row['label'] == expected_labels.get(row['path']) for row in rows),
        'valid_subset_names': all(row['subset'] in SUBSETS for row in rows),
        'group_identifiers_match_evidence': all(row['group_id'] == expected_groups.get(row['path']) for row in rows),
        'accepted_groups_stay_in_one_subset': all(len(values) == 1 for values in group_subsets.values()),
        'class_counts_match_targets': all(sum(row['label'] == label and row['subset'] == subset for row in rows) == targets[label][subset]
                                         for label in data.CLASSES for subset in SUBSETS),
        'seed_42_reproduces_every_assignment': rows == expected_rows,
    }
    return checks


def load_split(root, verify_source=False):
    """Load and verify the saved split; inspect metadata/hashes, never test pixels."""
    root = Path(root)
    rows = data.read_csv(root / 'manifests/split_manifest.csv')
    metadata = json.loads((root / 'manifests/split_metadata.json').read_text())
    records = data.load_dataset_records(root)
    expected, details = build_split(root, metadata['config'])
    checks = validate_rows(rows, records['eligible'], expected, details['class_counts'])
    checks['split_manifest_sha256_matches'] = data.file_sha256(root / 'manifests/split_manifest.csv') == metadata['manifest_sha256']
    checks['manifest_columns_match_schema'] = bool(rows) and list(rows[0]) == FIELDS
    checks['subset_checksums_match'] = metadata['subset_manifest_sha256'] == {
        subset: _digest(data.csv_bytes([row for row in rows if row['subset'] == subset], FIELDS)) for subset in SUBSETS}
    checks['inputs_and_code_match_recorded_checksums'] = input_hashes(root) == metadata['input_sha256']
    checks['grouping_and_counts_match_recorded_metadata'] = all(metadata[key] == value for key, value in details.items())
    excluded = {row['path'] for row in records['exclusions']}
    checks['six_exclusions_absent'] = excluded.isdisjoint(row['path'] for row in rows)
    checks['population_counts_match_metadata'] = (metadata['source_images'], metadata['excluded_images'], metadata['eligible_images']) == (2527, 6, len(rows))
    checks['class_stratification_within_one_image_of_ideal'] = all(
        abs(details['class_counts'][label][subset] - sum(row['label'] == label for row in rows) * weight / 20) < 1
        for label in data.CLASSES for subset, weight in zip(SUBSETS, CONFIG['ratio_numerators']))
    pixels = defaultdict(set)
    for row in records['eligible']:
        pixels[row['pixel_sha256']].add(row['path'])
    subsets = {row['path']: row['subset'] for row in rows}
    checks['exact_pixel_duplicates_do_not_cross_subsets'] = all(len({subsets[p] for p in paths}) <= 1 for paths in pixels.values())
    integrity = None
    if verify_source:
        integrity, _ = data.verify_source(root)
        checks['original_zip_and_source_files_unchanged'] = integrity['original_data_unchanged']
    if not all(checks.values()):
        raise ValueError('Split verification failed: ' + ', '.join(key for key, passed in checks.items() if not passed))
    return rows, metadata, checks, integrity


def create_split(root):
    """Explicit one-time write of the two artifacts, after all checks pass."""
    root = Path(root)
    for name in ('split_manifest.csv', 'split_metadata.json'):
        if (root / 'manifests' / name).exists():
            raise FileExistsError(f'Refusing to overwrite {name}.')
    before, _ = data.verify_source(root)
    if not before['original_data_unchanged']:
        raise ValueError('Original dataset does not match its baseline.')
    rows, details = build_split(root)
    records = data.load_dataset_records(root)
    reproduced, _ = build_split(root)
    checks = validate_rows(rows, records['eligible'], reproduced, details['class_counts'])
    if not all(checks.values()):
        raise ValueError(f'Split checks failed before writing: {checks}')
    payload = data.csv_bytes(rows, FIELDS)
    after, _ = data.verify_source(root)
    if not after['original_data_unchanged']:
        raise ValueError('Source changed during split construction.')
    metadata = {
        'schema_version': 1, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'created_and_verified', 'source_images': 2527, 'excluded_images': 6,
        'eligible_images': 2521, 'source_root': 'data/dataset-resized',
        'manifest_path': 'manifests/split_manifest.csv', 'manifest_sha256': _digest(payload),
        'manifest_format': 'UTF-8 CSV; LF; rows sorted by source-relative path; header path,label,subset,group_id',
        'python_version': platform.python_version(), 'rng': 'Python random.Random(seed=42), sorted inputs, fixed class/subset order',
        'input_sha256': input_hashes(root), **details,
        'subset_manifest_sha256': {subset: _digest(data.csv_bytes([row for row in rows if row['subset'] == subset], FIELDS)) for subset in SUBSETS},
        'actual_proportions': {subset: details['target_counts'][subset] / len(rows) for subset in SUBSETS},
        'deviation_from_target_percentage_points': {subset: 100 * (details['target_counts'][subset] / len(rows) - weight / 20)
                                                     for subset, weight in zip(SUBSETS, CONFIG['ratio_numerators'])},
        'verification_before_write': checks,
        'source_integrity_before': before, 'source_integrity_after': after,
        'limitations': [
            'Accepted visual and conservative relationships are not independently verified physical-object identities.',
            'Broad unverified families and generic look-alike pairs were not automatically grouped. Their residual relationships may cross subsets; singleton IDs do not certify independence.',
            'Some unaccepted broad proposals contain plausible subrelationships whose boundaries are unestablished. The finite existing review is not exhaustive and cannot guarantee absence of all leakage.',
            'Class/global quotas are rounded to whole images. Larger groups in a future dataset could make these exact quotas infeasible; this implementation then stops rather than splitting a group.',
            'Earlier pre-split exploration inspected source images; the test holdout begins at this saved assignment. No test images are used for subsequent inspection, model selection, or tuning.',
        ],
        'test_policy': 'Reserved and unused for model evaluation or tuning throughout the homework; only manifest statistics and byte-level integrity checks are permitted.',
    }
    data.write_artifact(root, 'manifests/split_manifest.csv', payload)
    data.write_artifact(root, 'manifests/split_metadata.json', _json_bytes(metadata))
    load_split(root, verify_source=True)
    return rows, metadata


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['create', 'verify'])
    args = parser.parse_args()
    project_root = data.get_project_root()
    if args.action == 'create':
        _, result = create_split(project_root)
        print(json.dumps({'counts': result['target_counts'], 'class_counts': result['class_counts'],
                          'related_groups': result['grouping']['related_group_count'],
                          'related_images': result['grouping']['related_image_count']}, indent=2))
    else:
        _, _, checks, _ = load_split(project_root, verify_source=True)
        print(json.dumps(checks, indent=2))
