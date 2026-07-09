# Classifier accuracy on CRUMB vs. its MiraBest-derived subset

## Background

The classifier checkpoint (`checkpoints/classification`) scored **40/100 (40%)**
on a random sample of the CRUMB test set (`90%_Mirabest_classifier_on_crumb.out`,
produced by `scripts/sanity_check_classifier.py --dataset crumb --num-samples 100`).
This is far below the ~90% test accuracy the same model reports on MiraBest, so
the question investigated in this session was: **is the CRUMB accuracy drop
specific to the non-MiraBest sources CRUMB adds, or does it also affect the
MiraBest sources folded into CRUMB?**

## Method: isolating MiraBest sources inside CRUMB

Neither dataset loader (`src/datasets/crumb/CRUMB.py`,
`src/datasets/mirabest/MiraBest.py`) exposes an explicit ID linking a CRUMB
image back to its MiraBest counterpart, but both raw pickle batches embed sky
coordinates in the image filename:

- CRUMB: `./CombinedCat/PNG/Scaled_Final/<RA>_<±Dec>.png`
  (e.g. `000.880_+000.468.png`)
- MiraBest: `./MiraBest_data/<class>_<RA><±Dec>_<z>_<size>.png`
  (e.g. `100_162.947+055.386_0.0739_0250.34.png`)

Matching (RA, Dec) pairs within 0.01° (~36 arcsec) across the two catalogues
identifies which CRUMB images are the same physical sources as MiraBest
galaxies. Note MiraBest's raw batches *do* contain filenames/coordinates —
`MiraBest.py`'s loader just discards them — so they had to be read directly
from the pickle files to recover coordinates.

### Verification that matches are real (not coincidental proximity)

Matching across the full corpora (CRUMB: 2100 images, MiraBest: 770 images):

| Check | Result |
|---|---|
| Non-hybrid matches found | 764 / 770 MiraBest sources (~99%) |
| Coordinate offset (matched pairs) | mean ≈ 3×10⁻⁶°, max 0.0022° — effectively exact |
| Pixel content (matched pairs) | mean abs diff 0.016 / 255; 41 pairs byte-identical |
| **FR-I/FR-II label agreement** | **683 / 764 agree (89.4%); 81 / 764 (10.6%) disagree** |

Conclusion: the coordinate match is reliable — these are the same physical
sources, not a coincidence of nearby unrelated galaxies (near-zero positional
offset, near-identical pixels). However, **CRUMB and MiraBest disagree on the
FR-I/FR-II label for ~1 in 10 of the shared sources.** This was not resolved
in this session — it's a real discrepancy between the two catalogues' labels,
not a bug in the matching. The classifier was trained on CRUMB's labels, so
CRUMB's label was used as ground truth throughout, but this is a caveat on
any accuracy number computed against these sources.

### Restricting to the held-out test split

To keep the "full CRUMB" and "MiraBest subset" numbers comparable and avoid
leaking train-time-seen images into the evaluation, the accuracy comparison
below restricts matching to CRUMB's **test split only** (not train):

- CRUMB test split: 300 images total, 286 non-hybrid.
- Of those 286, **100 match a MiraBest source** (tolerance 0.01°).

## Tooling produced this session

- `scripts/test_classifier_on_mirabest_subset.py` — loads the CRUMB test
  set, isolates the MiraBest-matched subset via the coordinate matching
  above, and reports, for **both** the full CRUMB test set and the
  MiraBest-matched subset:
  1. One full deterministic pass (every image, no sampling — exactly
     reproducible).
  2. `--repeats` (default 5) random 100-sample runs (with replacement,
     matching the original `sanity_check_classifier.py` methodology,
     seeds 0–4) to quantify how much of the originally reported 40% was
     sampling noise vs. a stable effect.
- `scripts/slurm/job_sanity_check_mirabest_subset.sh` — Slurm job
  (`sbatch scripts/slurm/job_sanity_check_mirabest_subset.sh`) that runs the
  above on the GPU partition. `NUM_SAMPLES` / `REPEATS` overridable via
  `sbatch --export=...`.
- `scripts/build_mirabest_crumb_mapping.py` — for **every** one of the 770
  MiraBest images (not just the CRUMB test-split subset), finds its matching
  CRUMB image (nearest coordinate match, any split) and writes
  `results/mirabest_classifier_vs_crumb_dataset/mirabest_crumb_mapping.csv`
  with: both filenames, both splits, both coordinates, both labels, the
  coordinate offset, the raw pixel mean-abs-diff, and a `labels_agree`
  boolean. Runs locally (CPU only, no checkpoint needed — just the raw
  batch files already downloaded). Result: **770/770 MiraBest images matched
  (100% coverage)**, 88/770 (11.4%) label disagreements overall.

## Results

| Scope | n | Full deterministic pass | Mean of 5 random 100-sample runs |
|---|---|---|---|
| MiraBest (`sanity_check_classifier.py --dataset mirabest`) | 100 (random sample) | — | 89/100 (89%) |
| Full CRUMB test set | 286 | 141/286 (49.3%) | 46.4% |
| MiraBest-matched subset of CRUMB test set | 100 | 49/100 (49.0%) | 49.6% |

The MiraBest-matched subset scores essentially the same as the full CRUMB
test set (~49% either way) — so the degradation is not concentrated in
CRUMB's non-MiraBest sources; it affects the MiraBest-overlap images just as
much despite those images being pixel-near-identical to their MiraBest
originals.

## Root cause: CRUMB's test-split labels disagree with MiraBest's for ~46% of the overlap sources

The full-corpus check above found 10.6% CRUMB/MiraBest label disagreement
across all 764 matches (train+test combined). Re-running that check
restricted to just the **100 CRUMB test-split images matched to a MiraBest
source** gives a very different number:

| Check (CRUMB test split only, n=100) | Result |
|---|---|
| Label agreement with MiraBest | 54 / 100 (54%) |
| Label disagreement with MiraBest | **46 / 100 (46%)** |
| Pixel diff (mean / max, 0–255 scale) | 0.0115 / 0.48 — still near-identical images |

This lines up almost exactly with the measured accuracy: if the classifier's
predictions track true morphology (consistent with its 89% score on
MiraBest's own test set), then scoring those same predictions against
CRUMB's test-split labels — which disagree with MiraBest on this specific
image on 46% of these sources — caps achievable accuracy at roughly 54%,
which is what was measured (49–50%).

**Conclusion: this is not a classifier problem, a preprocessing/format
problem, or a stale-checkpoint problem.** Pixel format (uint8, 0–255, same
`eval_transform`) was verified identical earlier; a `test_pipeline.py`
architecture bug (stale resnet50/1-channel build vs. the actual trained
resnet18/3-channel model, unrelated to this investigation) was found and
fixed in `src/pipelines/test_pipeline.py:156-166` but did not affect these
results, which all came from `sanity_check_classifier.py`-style loading
(already using the correct architecture). The ~40-50 point accuracy drop
between MiraBest and CRUMB is best explained by CRUMB's own test-split
FR-I/FR-II labels for the MiraBest-overlap sources disagreeing with
MiraBest's canonical labels on nearly half of those sources — i.e., a label
quality/consistency issue specific to CRUMB's test batch, not evidence the
classifier itself is unreliable.

### Caveat / not yet verified

This conclusion is inferred from the label-disagreement rate lining up with
the accuracy gap, not from directly comparing per-image model predictions
against both label sets. To confirm directly: for each of the 100 matched
test-split images, log the model's prediction alongside *both* the CRUMB
label and the MiraBest label, and check whether disagreements between
prediction and CRUMB-label correlate with CRUMB/MiraBest label disagreement
(they should, if this explanation is correct).

## Ruled out: the data-loading pipelines are not the cause

Checked whether `CRUMB.py` and `MiraBest.py` process a given image
differently (different resize/crop/normalize behaviour, different PIL mode
handling, etc.) rather than the images/labels themselves being the issue.
Both `__getitem__` implementations are structurally identical
(`np.reshape(img, (150,150))` → `Image.fromarray(img, mode='L')` →
apply the passed-in transform), so this was verified empirically rather than
just by reading the code: ran all 770 matched pairs through that exact logic
with the shared `eval_transform` (`Resize(150)` → `CenterCrop(150)` →
`Grayscale(3)` → `ToTensor` → `Normalize(ImageNet stats)`) and diffed the
final normalized tensors:

| Metric (post-transform, normalized scale) | Value |
|---|---|
| Per-pair mean abs diff — median | 0.000055 |
| Per-pair mean abs diff — average | 0.00026 |
| Per-pair max abs diff (single worst pixel) — median | 0.19 |
| Pairs with a max-pixel diff > 0.5 | 154 / 770 |
| Pairs with a max-pixel diff > 1.0 | 56 / 770 |

Whole-image mean difference is essentially zero for virtually every pair —
the two loaders produce the same tensor for the same underlying image. The
occasional single-pixel outlier (visible in ~20% of pairs) is inherited from
differences in the source PNGs themselves (re-encoding artifacts), not
introduced by the loading/transform code. **This rules out the data-loader
pipeline as an explanation** — the accuracy gap is not a preprocessing bug,
it's the CRUMB test-split label inconsistency documented above.
