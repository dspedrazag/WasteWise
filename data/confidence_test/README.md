# WasteWise external-image confidence test

20 Wikimedia Commons candidates, downloaded and visually reviewed before inference on 2026-11-18. This directory is separate from the original TrashNet data and all training, validation and test manifests. The existing Git ignore rules exclude it; no ignore rules were changed.

## Files and provenance

- `labels.csv`: original supplied manifest, byte-for-byte unchanged, including provisional labels and source URLs.
- `download_images.py`: supplied downloader, adapted for request pacing, HTTP 429/503 `Retry-After`, official resized renditions, and checksum-verified reuse of completed downloads.
- `sources.csv`: source page, original and downloaded URLs, author, license, attribution/credit, dates, transformation, local SHA-256 and final download status.
- `review.csv`: original and reviewed labels/descriptions/scope, review notes, date and image SHA-256. Used by the notebook.
- `duplicate_checks.csv`: actual downloaded dimensions, file/pixel hashes and nearest TrashNet perceptual-hash distances.
- `images/`: the 20 files listed below. Results and the labeled gallery are saved in `../../playground.ipynb`.

```text
images/
  plastic_01.jpg    plastic_02.jpg    plastic_03.jpg
  metal_01.jpg      metal_02.jpg      metal_03.jpg
  cardboard_01.jpg  cardboard_02.png  cardboard_03.jpg
  glass_01.jpg      glass_02.jpg      glass_03.jpg
  ood_01.jpg        ood_02.jpg        ood_03.jpg
  ambiguous_01.jpg  ambiguous_02.png  ambiguous_03.jpg
  scope_01.jpg      scope_02.jpg
```

From the repository root, use the existing scientific environment:
```sh
python data/confidence_test/download_images.py
```
Completed files are verified against saved provenance and not overwritten. A failed or mismatched file stops a complete evaluation; no silent substitution or reduced sample is allowed. Keep source attribution and each file's license with reused images. `ood_03.jpg` is credited to Doggo19292 under the attribution-only permission on its [source page](https://commons.wikimedia.org/wiki/File:Red_Apple1.jpg#Licensing); the API supplied no separate license URL.

## Download and visual review

Original-file requests repeatedly returned HTTP 429 after back-off. The same sources were downloaded as official Wikimedia resized renditions: 18 files are 1280 pixels wide, `ood_01.jpg` is 500 pixels wide and `ambiguous_02.png` is 250 pixels wide. Requested widths can differ from the server's returned size; actual dimensions are in `duplicate_checks.csv`. The final pass verified **20/20 readable images**, with no unresolved failed downloads. No local image edits were made. The notebook applies the existing RGB conversion, resize, center crop and normalization; transparent PNGs receive no new background-compositing treatment.

All 14 supplied nonblank material targets were retained after visual review. Six OOD/mixed-material scenes retain blank targets and are excluded from material accuracy, rather than assigned to the literal `trash` class. Visual labels are provisional, not physical material verification.

Corrections and clarifications:
- `plastic_02.jpg`: visible liquid/residue contradicts the supplied “empty” description.
- `cardboard_03.jpg`: an open box on several flattened boxes; changed scene type to `multiple_items` and guidance scope to `withhold`. Cardboard material remains scorable.
- `glass_03.jpg`: a drinking tumbler, despite the source filename “Glass bottle”.
- `glass_02.jpg`: a glass jar with a metal lid; the target material describes the body.
- `metal_01.jpg`: only the top of an apparently unopened can is visible.
- `scope_01.jpg` and `scope_02.jpg`: reusable-style plastic bottle and plastic cup; neither label establishes eligible bottle-type guidance.
- Some candidates have dim lighting, blur, backgrounds or cutout/collage composition. All per-image observations are in `review.csv`.

## Duplicate screen

Compared all 20 images against the **2,527 rows** in `manifests/image_inventory.csv`, without changing or evaluating TrashNet images. Checked file and decoded-pixel SHA-256, then existing `src.data_preparation.perceptual_hashes` (63-bit pHash and 64-bit dHash). Each external image was compared in eight rotation/mirror variants; candidate limits match the existing project screen: pHash distance ≤ 8 or dHash distance ≤ 4. Also checked exact and perceptual matches within the external set in its original orientations.

**No exact matches or near-duplicate candidates were found.** Closest TrashNet distances were pHash 10 and dHash 6, above those limits. The nearest paths in the CSV are audit references, not detected duplicates. This heuristic may miss crops or substantial edits and does not prove independence from TrashNet or the model's original ImageNet pretraining.

## Frozen 85% result

9/20 accepted (45% coverage), 11 rejected. Correct predictions: 7 accepted, 4 rejected. Incorrect: 1 accepted, 2 rejected. Not applicable: 1 accepted, 5 rejected. Accepted accuracy is **7/8 = 87.50% among labeled images**; all 14 labeled images give 11/14 = 78.57% without rejection.

The plastic bottle `plastic_02.jpg` was accepted incorrectly as glass (86.99%). The mixed-material collage `ambiguous_02.png` passed as metal (91.34%); the multiple-box scene `cardboard_03.jpg` passed as cardboard (99.70%) but remains an invalid single-item input. All three designated OOD images were rejected. The correctly classified drinking tumbler also passed (89.96%), illustrating why material correctness does not establish item-level recycling eligibility.

Softmax is uncalibrated. This small convenience sample has no labeled paper/trash examples and does not represent UF camera use. Download resizing, visual-label uncertainty and image composition limit generalization. The threshold was not tuned. A confidence pass does not replace item selection, condition questions or verified applicable local rules. No application, dataset, split or checkpoint was changed.
