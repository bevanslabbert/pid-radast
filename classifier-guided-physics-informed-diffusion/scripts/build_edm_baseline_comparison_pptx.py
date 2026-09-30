"""Rebuild EDM_baseline_comparison.pptx  -- lean version.

EDM literature baseline (Vicanek Martinez et al. 2024) vs this project's
DDPM `diffusion` model (the CRUMB-dataset baseline) vs
`classifier_guided_diffusion` (CGD). Differences table + all available
results (metric tables, metric-vs-epoch graphs) + sample grids including
real CRUMB ground truth.

Prereqs (run first):
  python scripts/make_crumb_groundtruth_grid.py
  python scripts/make_edm_comparison_metric_plot.py
"""
import os

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "EDM_baseline_comparison.pptx")

NAVY = RGBColor(0x1F, 0x2A, 0x44)
ACCENT = RGBColor(0x2E, 0x6F, 0x9E)
GREY = RGBColor(0x55, 0x55, 0x55)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]


def slide():
    return prs.slides.add_slide(BLANK)


def title(s, text, sub=None):
    tb = s.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(12.3), Inches(1.0))
    p = tb.text_frame.paragraphs[0]
    r = p.add_run(); r.text = text
    r.font.size = Pt(28); r.font.bold = True; r.font.color.rgb = NAVY
    if sub:
        p2 = tb.text_frame.add_paragraph()
        r = p2.add_run(); r.text = sub
        r.font.size = Pt(13); r.font.color.rgb = GREY


def table(s, rows, top, left=0.5, width=12.3, height=3.0, col_widths=None, font=12):
    nr, nc = len(rows), len(rows[0])
    gt = s.shapes.add_table(nr, nc, Inches(left), Inches(top), Inches(width), Inches(height)).table
    if col_widths:
        for i, w in enumerate(col_widths):
            gt.columns[i].width = Inches(w)
    for ci in range(nc):
        for ri in range(nr):
            cell = gt.cell(ri, ci)
            cell.text = str(rows[ri][ci])
            para = cell.text_frame.paragraphs[0]
            para.alignment = PP_ALIGN.LEFT if ci == 0 else PP_ALIGN.CENTER
            for r in para.runs:
                r.font.size = Pt(font)
                if ri == 0:
                    r.font.bold = True; r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                else:
                    r.font.color.rgb = NAVY
            if ri == 0:
                cell.fill.solid(); cell.fill.fore_color.rgb = ACCENT
            else:
                cell.fill.solid()
                cell.fill.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF) if ri % 2 else RGBColor(0xEE, 0xF2, 0xF7)
    return gt


def pic(s, path, left, top, width, height=None):
    if os.path.exists(path):
        if height is not None:
            s.shapes.add_picture(path, Inches(left), Inches(top), height=Inches(height))
        else:
            s.shapes.add_picture(path, Inches(left), Inches(top), width=Inches(width))
    else:
        tb = s.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(0.4))
        tb.text_frame.paragraphs[0].add_run().text = "[missing: %s]" % os.path.basename(path)


def caption(s, text, top=7.0):
    tb = s.shapes.add_textbox(Inches(0.5), Inches(top), Inches(12.3), Inches(0.4))
    r = tb.text_frame.paragraphs[0].add_run(); r.text = text
    r.font.size = Pt(10); r.font.italic = True; r.font.color.rgb = GREY


RES = os.path.join(ROOT, "results")

# ---------------------------------------------------------------- 1 title
s = slide()
tb = s.shapes.add_textbox(Inches(0.8), Inches(2.8), Inches(11.7), Inches(2.0))
tf = tb.text_frame; tf.word_wrap = True
r = tf.paragraphs[0].add_run()
r.text = "EDM Baseline  vs  CRUMB DDPM  vs  Classifier-Guided Diffusion"
r.font.size = Pt(32); r.font.bold = True; r.font.color.rgb = NAVY
p = tf.add_paragraph(); r = p.add_run()
r.text = "Class-conditional FR-I / FR-II radio-galaxy generation  ·  CRUMB FITS, 150x150"
r.font.size = Pt(15); r.font.color.rgb = GREY

# ---------------------------------------------------------------- 2 differences
s = slide()
title(s, "Key differences")
table(s, [
    ["", "EDM baseline", "diffusion (DDPM)", "CGD"],
    ["Role", "literature reference\n(Vicanek Martinez 2024)", "this project's\nCRUMB-dataset baseline", "DDPM + external steer"],
    ["U-Net", "UNet2DModel, ~29.8M", "UNet2DConditionModel, ~95.9M", "same as DDPM"],
    ["Conditioning", "additive class embedding", "cross-attention (dim 256)", "cross-attention (dim 256)"],
    ["Noise process", "continuous sigma (Karras)", "1000-step discrete, linear beta", "1000-step discrete, linear beta"],
    ["Target / loss", "clean image / sigma-weighted L2", "noise (epsilon) / plain MSE", "noise (epsilon) / plain MSE"],
    ["Sampler", "Heun ODE, 25 steps, EMA", "ancestral DDPM, 50 steps", "ancestral DDPM, 50 steps"],
    ["Guidance", "CFG (3.0)", "CFG (7.5)", "CFG + noise-aware classifier\ngradient (strength 3)"],
], top=1.5, height=4.6, col_widths=[1.9, 3.5, 3.6, 3.3], font=12)
caption(s, "Shared: task, CRUMB FITS pipeline, symmetric-log-SNR normalisation, CFG-style class conditioning, FID/KID/PDF evaluation.")

# ---------------------------------------------------------------- 3 results tables
s = slide()
title(s, "Results")
tb = s.shapes.add_textbox(Inches(0.5), Inches(1.35), Inches(6), Inches(0.3))
tb.text_frame.paragraphs[0].add_run().text = "Distributional metrics (best over training)"
tb.text_frame.paragraphs[0].runs[0].font.bold = True
tb.text_frame.paragraphs[0].runs[0].font.size = Pt(13)
table(s, [
    ["Model", "FID ↓", "KID ↓", "pixel-PDF W ↓", "Epochs"],
    ["EDM baseline  (mean ± std, n=5: seeds 42-46)", "87.3 ± 5.0", "0.048 ± 0.005", "0.0036 ± 0.0009", "400\n(complete)"],
    ["diffusion (DDPM+CFG)  (mean ± std, n=4: seeds 42-45)", "73.4 ± 1.4", "0.021 ± 0.003", "0.0077 ± 0.0021", "400\n(complete)"],
    ["CGD*  (mean ± std, n=5: seeds 42-46)", "74.4 ± 5.2", "0.024 ± 0.005", "0.0078 ± 0.0009", "400\n(complete)"],
], top=1.7, height=1.8, col_widths=[4.6, 1.8, 2.0, 2.4, 1.5], font=12)
tb = s.shapes.add_textbox(Inches(0.5), Inches(3.55), Inches(11.5), Inches(0.35))
r = tb.text_frame.paragraphs[0].add_run()
r.text = ("EDM and CGD have all 5 seeds; DDPM has 4 (seed 46 hit its time limit at epoch 260). Std is the sample std across seeds.  "
          "*CGD sampled with its original, broken guidance classifier - effectively CFG only (see slide 6).")
r.font.size = Pt(10); r.font.italic = True; r.font.color.rgb = GREY

tb = s.shapes.add_textbox(Inches(0.5), Inches(4.3), Inches(9), Inches(0.3))
tb.text_frame.paragraphs[0].add_run().text = "CRUMB VQ-VAE reconstruction fidelity  (VQ-VAE trained only on real CRUMB)"
tb.text_frame.paragraphs[0].runs[0].font.bold = True
tb.text_frame.paragraphs[0].runs[0].font.size = Pt(13)
table(s, [
    ["Passed through the CRUMB VQ-VAE", "N", "Recon MSE ↓", "Recon NCC ↑"],
    ["Real held-out CRUMB  (reference)", "178", "0.00377", "0.652"],
    ["EDM-generated (latest, final)", "32", "0.00271", "0.763"],
    ["DDPM-generated", "16", "0.00174", "0.771"],
    ["CGD-generated (latest, ep 290)", "32", "0.00300", "0.725"],
], top=4.65, height=2.0, col_widths=[5.2, 1.3, 3.0, 3.0], font=12)
caption(s, "DDPM and (unguided) CGD beat EDM on FID/KID; EDM keeps the best pixel-PDF W. DDPM best on VQ-VAE reconstruction fidelity. "
           "Working-guidance CGD results: slides 6-8.")

# ---------------------------------------------------------------- 3b classifier on generated samples
s = slide()
title(s, "Classifier accuracy on generated samples",
      "Does the model generate the class it was asked for?  128 samples per class per seed, scored by a held-out classifier")
table(s, [
    ["Model", "Seeds", "Class accuracy", "FR-I", "FR-II", "Mean confidence"],
    ["Real CRUMB test images  (reference)", "-", "78.1 %", "84.6 %", "73.0 %", "-"],
    ["EDM baseline", "5", "80.7 ± 1.8 %", "71.6 ± 6.5 %", "89.8 ± 3.5 %", "0.743 ± 0.014"],
    ["diffusion (DDPM+CFG)", "4", "96.5 ± 1.5 %", "93.0 ± 2.9 %", "100.0 ± 0.0 %", "0.910 ± 0.015"],
    ["CGD*  (broken guidance classifier)", "5", "97.1 ± 2.4 %", "95.3 ± 2.9 %", "98.9 ± 2.0 %", "0.891 ± 0.049"],
], top=1.6, height=2.4, col_widths=[3.6, 0.9, 2.1, 1.9, 1.9, 1.9], font=13)
tb = s.shapes.add_textbox(Inches(0.5), Inches(4.2), Inches(12.3), Inches(2.7))
tb.text_frame.word_wrap = True
for txt in [
    "A model matching real data would score like the reference row, not 100%. DDPM/CGD overshoot on both classes: their samples are more clear-cut than real galaxies.",
    "EDM is close to real overall, but every model reverses the real per-class pattern (real FR-I 85% > FR-II 73%) at every guidance scale - EDM most strongly (FR-I 72%, FR-II 90%).",
    "*CGD here is effectively CFG only: its guidance classifier was broken (slide 6). With a working one it improves (slides 7-8).",
    "Confound: EDM samples at CFG 3.0, DDPM/CGD at 7.5 - stronger guidance exaggerates class features.",
]:
    p = tb.text_frame.add_paragraph(); r = p.add_run(); r.text = "- " + txt
    r.font.size = Pt(13); r.font.color.rgb = GREY; p.space_after = Pt(8)
caption(s, "Scored by eval_fits_bs8_seed42 (SimpleCNN, crumb_fits domain, never used to guide CGD; real-test reference from its confusion matrix, n=178). "
           "Mean ± sample std across seeds. DDPM seed 46 incomplete.")

# ---------------------------------------------------------------- 3c matched-guidance sweep
s = slide()
title(s, "Matched guidance: accuracy and FID vs CFG scale",
      "Same checkpoints re-sampled at CFG 1.0 / 3.0 / 7.5 - separates architecture from guidance strength")
pic(s, os.path.join(RES, "edm_baseline/guidance_sweep.png"), 1.15, 1.35, 11.0)
tb = s.shapes.add_textbox(Inches(0.5), Inches(5.55), Inches(12.3), Inches(1.4))
tb.text_frame.word_wrap = True
for txt in [
    "At every matched scale DDPM beats EDM on FID (50.9 vs 65.5, 50.4 vs 68.4, 61.0 vs 94.1) and on class accuracy up to 3.0 - the lead is architectural, not guidance.",
    "CFG 7.5 costs every model FID; 3.0 is the better operating point (best FID, accuracy nearest real). CGD line: broken guidance classifier (effectively CFG only).",
]:
    p = tb.text_frame.add_paragraph(); r = p.add_run(); r.text = "- " + txt
    r.font.size = Pt(12); r.font.color.rgb = GREY; p.space_after = Pt(4)
caption(s, "FID from 256 generated images per seed vs the 357-image crumb_fits val split - not comparable in absolute terms to the training-time FID table.",
        top=7.05)

# ---------------------------------------------------------------- 3d CGD guidance classifier diagnosis
s = slide()
title(s, "CGD's guidance classifier was broken",
      "Guidance queries the classifier on noisy, half-formed images at every sampling step - it must work under noise")
table(s, [
    ["Guidance classifier", "Trained on", "Noise-aware", "Clean acc", "Acc @ t=100", "Acc @ t=200", "Behaviour during sampling"],
    ["scratch  (original CGD)", "linear crumb  (wrong domain)", "no", "42 %", "42 %", "42 %", "predicts FR-I for every image: gradient ~0"],
    ["SimpleCNN, crumb_fits  (quick fix)", "crumb_fits", "no", "77 %", "58 %", "58 %", "predicts FR-II for all t > 50: last ~2 steps only"],
    ["ResNet50 + timestep  (robust)", "crumb_fits + DDPM noise", "yes", "75 %", "73 %", "63 %", "tracks the requested class, 0.5 -> 0.9"],
], top=1.6, height=2.3, col_widths=[2.9, 2.2, 1.1, 1.0, 1.1, 1.1, 2.9], font=11)
tb = s.shapes.add_textbox(Inches(0.5), Inches(4.3), Inches(12.3), Inches(2.4))
tb.text_frame.word_wrap = True
for txt in [
    "All earlier CGD results used 'scratch': collapsed to a constant prediction on crumb_fits images, so its gradient was 0.01-0.1% of the update - CGD was effectively DDPM + CFG.",
    "A correct-domain classifier trained on clean images still collapses once noise is added (t >= 50 of 1000), so it can only steer the final steps.",
    "Fix: noise-aware classifier (Dhariwal & Nichol 2021) - ResNet50 trained on crumb_fits images noised with the diffusion schedule, given t as input. Clean test acc 83.2 %.",
    "CGD checkpoints are unchanged - guidance acts only at sampling time, so no diffusion retraining was needed.",
]:
    p = tb.text_frame.add_paragraph(); r = p.add_run(); r.text = "- " + txt
    r.font.size = Pt(12); r.font.color.rgb = GREY; p.space_after = Pt(6)
caption(s, "Accuracy on real crumb_fits validation images noised to timestep t (diagnose_cgd_guidance.py). Chance ~ 56 % (majority class).")

# ---------------------------------------------------------------- 3e CGD guidance strength
s = slide()
title(s, "CGD with a working guidance classifier",
      "Robust classifier guidance at CFG 3.0 - classifier-guidance strength sweep")
pic(s, os.path.join(RES, "edm_baseline/cgd_guidance_strength.png"), 1.15, 1.3, 11.0)
tb = s.shapes.add_textbox(Inches(0.5), Inches(5.55), Inches(12.3), Inches(1.4))
tb.text_frame.word_wrap = True
for txt in [
    "Strength 3: class accuracy 84.3 -> 91.4 %, FR-I 74.1 -> 84.1 % (real 84.6 %), FID 58.0 -> 56.9, KID 0.039 -> 0.037 - no quality cost.",
    "Plateau at 3-10; FID degrades from 30; at 100 guidance overwhelms the model (FID 84.9) and accuracy falls back to unguided.",
]:
    p = tb.text_frame.add_paragraph(); r = p.add_run(); r.text = "- " + txt
    r.font.size = Pt(12); r.font.color.rgb = GREY; p.space_after = Pt(4)
caption(s, "Guide (ResNet50, noise-aware) and scorer (SimpleCNN) are independent architectures, so the gain is not the guide fooling a copy of itself.",
        top=7.05)

# ---------------------------------------------------------------- 3f comparison at matched settings
s = slide()
title(s, "All models at CFG 3.0",
      "Same scoring classifier, same real reference, 128 samples per class per seed")
table(s, [
    ["Model", "Seeds", "Class acc", "FR-I", "FR-II", "FID ↓", "KID ↓"],
    ["Real CRUMB test images  (reference)", "-", "78.1 %", "84.6 %", "73.0 %", "-", "-"],
    ["EDM baseline", "5", "80.7 ± 1.8 %", "71.6 %", "89.8 %", "68.4 ± 3.7", "0.056"],
    ["DDPM", "4", "87.5 ± 2.9 %", "79.5 %", "95.5 %", "50.4 ± 6.2", "0.033"],
    ["DDPM + robust guidance (strength 10)  - prelim.", "1 of 4", "96.9 %  (89.8)", "93.8 %  (85.9)", "100 %  (93.8)", "55.6  (55.9)", "0.031  (0.036)"],
    ["CGD, unguided", "5", "84.3 ± 6.0 %", "74.1 %", "94.5 %", "58.0 ± 7.6", "0.039"],
    ["CGD + robust guidance (strength 3)", "5", "91.4 ± 3.3 %", "84.1 %", "98.8 %", "56.9 ± 6.3", "0.037"],
    ["DDPM at CFG 7.5  (raising the CFG scale instead)", "4", "96.5 ± 1.5 %", "93.0 %", "100 %", "61.0 ± 5.2", "0.037"],
], top=1.6, height=3.4, col_widths=[4.3, 0.9, 1.6, 1.5, 1.4, 1.4, 1.2], font=11)
tb = s.shapes.add_textbox(Inches(0.5), Inches(5.2), Inches(12.3), Inches(1.7))
tb.text_frame.word_wrap = True
for txt in [
    "Classifier guidance raises class fidelity without the FID cost of a higher CFG scale: CGD +7 pts at -1 FID, vs DDPM CFG 3 -> 7.5 +9 pts at +10.6 FID.",
    "CGD's remaining FID gap to DDPM is inherited from its weaker checkpoints, not caused by guidance. DDPM + guidance (seed 42) keeps DDPM's FID - remaining seeds pending.",
]:
    p = tb.text_frame.add_paragraph(); r = p.add_run(); r.text = "- " + txt
    r.font.size = Pt(12); r.font.color.rgb = GREY; p.space_after = Pt(6)
caption(s, "Prelim. row: one seed; bracketed values are the same seed unguided. Mean ± sample std across seeds. FID from 256 samples/seed vs 357 real val images.")

# ---------------------------------------------------------------- 4 comparison graph
s = slide()
title(s, "Metrics vs epoch")
pic(s, os.path.join(RES, "edm_baseline/metric_comparison.png"), 0.4, 2.1, 12.6)

# ---------------------------------------------------------------- 6 reconstruction overview + metrics
s = slide()
title(s, "CRUMB VQ-VAE reconstruction",
      "Generated images passed through a VQ-VAE trained only on real CRUMB - does their structure lie on the CRUMB manifold?")
pic(s, os.path.join(RES, "edm_baseline/recon_overview.png"), 0.35, 1.55, 0, height=4.1)
table(s, [
    ["", "MSE ↓", "NCC ↑"],
    ["Real CRUMB (reference)", "0.00377", "0.652"],
    ["EDM-generated (latest, final)", "0.00271", "0.763"],
    ["DDPM-generated", "0.00174", "0.771"],
    ["CGD-generated (latest, ep 290)", "0.00300", "0.725"],
], top=6.0, left=0.5, width=7.0, height=1.3, col_widths=[3.4, 1.8, 1.8], font=11)
tb = s.shapes.add_textbox(Inches(8.0), Inches(2.0), Inches(4.9), Inches(4.5))
tb.text_frame.word_wrap = True
for txt in [
    "One example pair per model (col 1 input, col 2 reconstruction).",
    "All models reconstruct at least as well as real held-out CRUMB.",
    "DDPM has the highest NCC this round - most CRUMB-like morphology.",
    "MSE flatters all models: generated fields are smoother than real CRUMB.",
    "Per-model example galleries on the following slides.",
]:
    p = tb.text_frame.add_paragraph(); r = p.add_run(); r.text = "- " + txt
    r.font.size = Pt(12); r.font.color.rgb = GREY; p.space_after = Pt(8)

# ---------------------------------------------------------------- 6b per-model reconstruction galleries
for key, lbl in [("crumb", "Real CRUMB"), ("edm", "EDM-generated"),
                 ("ddpm", "DDPM-generated"), ("cgd", "CGD-generated")]:
    s = slide()
    title(s, "Reconstruction examples - %s" % lbl,
          "col 1 = input   ·   col 2 = CRUMB VQ-VAE reconstruction")
    pic(s, os.path.join(RES, "edm_baseline/recon_%s.png" % key), 3.4, 1.35, 0, height=6.0)

# ---------------------------------------------------------------- 7 samples
s = slide()
title(s, "Samples  (left cols FR-I  ·  right cols FR-II)")
rows = [
    ("Real CRUMB  (ground truth)", os.path.join(RES, "edm_baseline/crumb_groundtruth_samples.png")),
    ("EDM baseline (latest, 400 ep) - epoch 320", os.path.join(RES, "edm_baseline/20260917_161506_my_run_tag_913384/comparison_epoch_320.png")),
    ("DDPM (latest, 400 ep) - epoch 390", os.path.join(RES, "diffusion/20260917_161506_diffusion_crumb_fits_913386/comparison_epoch_390.png")),
    ("CGD (latest, 400 ep) - epoch 390", os.path.join(RES, "classifier_guided_diffusion/20260920_172407_cls_guided_diffusion_crumb_fits_seed42_921074/comparison_epoch_390.png")),
]
pos = [(0.9, 1.25), (7.1, 1.25), (0.9, 4.6), (7.1, 4.6)]
for (label, path), (x, yy) in zip(rows, pos):
    tb = s.shapes.add_textbox(Inches(x), Inches(yy), Inches(5.3), Inches(0.3))
    r = tb.text_frame.paragraphs[0].add_run(); r.text = label
    r.font.size = Pt(12); r.font.bold = True; r.font.color.rgb = NAVY
    pic(s, path, x, yy + 0.3, 5.3)

prs.save(OUT)
print("wrote", OUT)
