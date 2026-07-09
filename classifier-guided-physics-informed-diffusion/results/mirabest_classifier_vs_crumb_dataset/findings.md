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

## Status

**Not yet run** — this session had no Slurm access (local machine only has
`sbatch`/`squeue` on the cluster). The script and job above are ready to
submit; once run, the actual full-set and repeated-sample accuracy numbers
for both scopes should be appended below.

## Results

_(pending — append `sbatch` output here once the job above has been run)_
