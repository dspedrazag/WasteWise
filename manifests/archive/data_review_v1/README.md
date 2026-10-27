# Retained data-review evidence

The final dataset contains **2,521 eligible images and six manifest-only
exclusions**. The seven main manifests hold the approved population, split,
grouping decisions, and integrity baseline. Original images remain unchanged.

The following files remain published because existing code depends on them:

| Records | Purpose |
| --- | --- |
| `related_groups.csv`, `near_duplicate_review.csv`, `group_review_assessments.json` | Original grouping evidence, checked against hashes in the main metadata and used by `src/data_splitting.py` to reconstruct and verify the approved split. |
| `near_duplicate_candidates.csv`, `visual_screen_decisions.json`, `overview_group_proposals.json`, `contact_sheet_index.csv` | Inputs required by the legacy `src/data_preparation.py review` command. This command is not needed to load the completed experiments. |

Eight superseded review records were moved to an external local backup; the redundant
`cleanup_verification.json` and `source_checksums_after.csv` were removed. The
latter was identical to the retained `manifests/source_checksums_before.csv`.
Historical galleries are preserved under `data_review/` within that backup and
must be restored to their original paths to follow the links in the retained
evidence tables. These relocated records and galleries are not inputs to the
completed experiment loaders or split verifier.

Playground reads only the seven main manifests. Split reconstruction additionally
requires the three grouping-evidence files above. Archived proposals are evidence,
not independent confirmation of object identity; the approved assignments and
remaining leakage limitations are recorded in `manifests/split_metadata.json`.
