"""Rebuild EDM_baseline_comparison.pptx  -- lean version.

EDM literature baseline (Vicanek Martinez et al. 2024) vs this project's DDPM
`diffusion` model vs CGD = the same DDPM checkpoints sampled with noise-aware
classifier guidance (robust guidance classifier, strength 3, CFG 3.0).

Prereqs (run first):
  python scripts/make_crumb_groundtruth_grid.py
  python scripts/make_edm_comparison_metric_plot.py
  python scripts/make_guidance_sweep_plot.py
  python scripts/make_cgd_strength_plot.py
  python scripts/make_recon_grid.py
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


EVAL = os.path.join(RES, "generation_classifier_eval")


def bullets(s, items, top, size=14, height=1.5):
    tb = s.shapes.add_textbox(Inches(0.5), Inches(top), Inches(12.3), Inches(height))
    tb.text_frame.word_wrap = True
    for txt in items:
        p = tb.text_frame.add_paragraph(); r = p.add_run(); r.text = "- " + txt
        r.font.size = Pt(size); r.font.color.rgb = GREY; p.space_after = Pt(6)


# ---------------------------------------------------------------- 1 title
s = slide()
tb = s.shapes.add_textbox(Inches(0.8), Inches(2.8), Inches(11.7), Inches(2.0))
tf = tb.text_frame; tf.word_wrap = True
r = tf.paragraphs[0].add_run()
r.text = "EDM Baseline  vs  DDPM  vs  Classifier-Guided Diffusion"
r.font.size = Pt(32); r.font.bold = True; r.font.color.rgb = NAVY
p = tf.add_paragraph(); r = p.add_run()
r.text = "Class-conditional FR-I / FR-II radio-galaxy generation  ·  CRUMB FITS, 150x150"
r.font.size = Pt(15); r.font.color.rgb = GREY

# ---------------------------------------------------------------- 2 models
s = slide()
title(s, "The three models")
table(s, [
    ["", "EDM baseline", "DDPM", "CGD"],
    ["What it is", "literature reference\n(Vicanek Martinez 2024)", "this project's baseline", "DDPM + classifier guidance\nat sampling time"],
    ["Network", "UNet2DModel, ~29.8M", "UNet2DConditionModel, ~95.9M", "same checkpoints as DDPM"],
    ["Sampler", "Heun ODE, 25 steps", "DDPM, 50 steps", "DDPM, 50 steps"],
    ["Class steering", "CFG", "CFG", "CFG + gradient of a frozen,\nnoise-aware classifier"],
], top=1.5, height=3.6, col_widths=[2.0, 3.4, 3.4, 3.5], font=14)
caption(s, "All results use CFG 3.0 unless noted. CGD needs no retraining - guidance only changes sampling.", top=5.6)

# ---------------------------------------------------------------- 3 training-time quality
s = slide()
title(s, "Image quality during training")
t = table(s, [
    ["Model  (best over 400 epochs)", "FID ↓", "KID ↓", "pixel-PDF W ↓"],
    ["EDM baseline  (5 seeds)", "87.3 ± 5.0", "0.048 ± 0.005", "0.0036 ± 0.0009"],
    ["DDPM  (4 seeds)", "73.4 ± 1.4", "0.021 ± 0.003", "0.0077 ± 0.0021"],
    ["CGD", "same training as DDPM - guidance is added at sampling (slide 7)", "", ""],
], top=1.3, height=1.45, col_widths=[4.6, 2.4, 2.6, 2.7], font=13)
t.cell(3, 1).merge(t.cell(3, 3))
pic(s, os.path.join(RES, "edm_baseline/metric_comparison.png"), 0.85, 2.95, 11.6)
caption(s, "DDPM wins FID/KID; EDM matches the pixel-intensity distribution best.")

# ---------------------------------------------------------------- 4 CFG scale
s = slide()
title(s, "CFG scale: class fidelity vs image quality")
pic(s, os.path.join(RES, "edm_baseline/guidance_sweep.png"), 1.15, 1.3, 11.0)
bullets(s, [
    "Higher CFG = more class-typical samples, but worse FID. CFG 3.0 is the best trade-off.",
    "DDPM beats EDM on FID at every matched scale.",
], top=5.6)

# ---------------------------------------------------------------- 5 guidance classifier
s = slide()
title(s, "Classifier guidance needs a noise-aware classifier",
      "The classifier is queried on noisy, half-formed images at every sampling step")
table(s, [
    ["Guidance classifier", "Works on noisy images?", "Effect on samples"],
    ["Original (wrong data domain)", "no - always predicts FR-I", "none"],
    ["Clean-image classifier", "no - collapses once noise is added", "last 1-2 steps only"],
    ["Noise-aware ResNet50 (used)", "yes - 73 % at t=100, 63 % at t=200", "steers the second half of sampling"],
], top=1.7, height=2.3, col_widths=[4.0, 4.4, 3.9], font=14)
bullets(s, [
    "Earlier CGD results used the broken original classifier, so they were effectively plain DDPM.",
], top=4.5)

# ---------------------------------------------------------------- 6 guidance strength
s = slide()
title(s, "Choosing the guidance strength")
pic(s, os.path.join(RES, "edm_baseline/cgd_guidance_strength.png"), 1.15, 1.3, 11.0)
bullets(s, [
    "Accuracy plateaus at 3-10 with no FID cost; too strong (100) breaks the samples. Chosen: 3.",
], top=5.6)
caption(s, "Swept on a second, independently trained set of DDPM checkpoints.", top=6.6)

# ---------------------------------------------------------------- 7 main results
s = slide()
title(s, "Results at CFG 3.0", "128 samples per class per seed, scored by a separate held-out classifier")
table(s, [
    ["Model", "Seeds", "Class accuracy", "FR-I", "FR-II", "FID ↓", "KID ↓"],
    ["Real CRUMB images  (target)", "-", "78.1 %", "84.6 %", "73.0 %", "-", "-"],
    ["EDM baseline", "5", "80.7 ± 1.8 %", "71.6 %", "89.8 %", "68.4 ± 3.7", "0.056"],
    ["DDPM", "5", "89.0 ± 4.2 %", "81.5 %", "96.4 %", "50.6 ± 5.4", "0.033"],
    ["CGD  (DDPM + guidance)", "5", "92.8 ± 3.4 %", "87.4 %", "98.3 %", "50.5 ± 5.3", "0.031"],
], top=1.7, height=2.5, col_widths=[3.8, 0.9, 2.0, 1.4, 1.4, 1.6, 1.2], font=14)
bullets(s, [
    "Guidance improves every seed (+3.8 pts, FR-I +5.9) at no FID cost.",
    "Raising CFG to 7.5 instead buys similar accuracy but costs +10.6 FID.",
    "Same result on a second training run: 84.3 -> 91.4 %.",
], top=4.5)

# ---------------------------------------------------------------- 8 guidance samples
s = slide()
title(s, "What guidance changes", "Same model and starting noise - only the guidance differs (DDPM seed 44)")
# DDPM seed 44, CFG 3.0, classifier_scale 0 vs 3 (eval_ddpm_cfg3.0_robcls{0,3}_v2), greyscale
for label, fig, top in [("DDPM\n(no guidance)", "guidance_samples_off.png", 1.45),
                        ("CGD\n(guided)", "guidance_samples_on.png", 4.2)]:
    tb = s.shapes.add_textbox(Inches(0.4), Inches(top + 1.0), Inches(2.3), Inches(0.5))
    r = tb.text_frame.paragraphs[0].add_run(); r.text = label
    r.font.size = Pt(14); r.font.bold = True; r.font.color.rgb = NAVY
    pic(s, os.path.join(RES, "edm_baseline", fig), 2.9, top, 9.8)
caption(s, "Guidance refines details (e.g. sharper FR-II cores) rather than redrawing the galaxy.")

# ---------------------------------------------------------------- 9 samples vs real
s = slide()
title(s, "Samples vs real")
panels = [  # (label, path, x, y, width)
    ("Real CRUMB  (left cols FR-I · right cols FR-II)", os.path.join(RES, "edm_baseline/crumb_groundtruth_samples.png"), 1.0, 1.15, 5.0),
    ("EDM baseline  (left cols FR-I · right cols FR-II)", os.path.join(RES, "edm_baseline/20260917_161506_my_run_tag_913384/comparison_epoch_320.png"), 7.2, 1.15, 5.0),
    ("DDPM  (top FR-I · bottom FR-II)", os.path.join(RES, "edm_baseline/guidance_samples_off.png"), 0.6, 4.75, 6.0),
    ("CGD  (top FR-I · bottom FR-II)", os.path.join(RES, "edm_baseline/guidance_samples_on.png"), 6.8, 4.75, 6.0),
]
for label, path, x, yy, w in panels:
    tb = s.shapes.add_textbox(Inches(x), Inches(yy), Inches(w), Inches(0.3))
    r = tb.text_frame.paragraphs[0].add_run(); r.text = label
    r.font.size = Pt(12); r.font.bold = True; r.font.color.rgb = NAVY
    pic(s, path, x, yy + 0.3, w)

# ---------------------------------------------------------------- 10 VQ-VAE reconstruction
s = slide()
title(s, "Do samples look like CRUMB?", "Passed through a VQ-VAE trained only on real CRUMB")
pic(s, os.path.join(RES, "edm_baseline/recon_overview.png"), 0.65, 1.5, 12.0)
table(s, [
    ["", "Recon MSE ↓", "Recon NCC ↑"],
    ["Real CRUMB  (reference)", "0.00377", "0.652"],
    ["EDM-generated", "0.00271", "0.763"],
    ["DDPM-generated", "0.00174", "0.771"],
], top=4.7, left=3.15, width=7.0, height=1.6, col_widths=[3.4, 1.8, 1.8], font=13)
caption(s, "Both reconstruct at least as well as real CRUMB (generated fields are smoother). Seed 42, unguided.")

# ---------------------------------------------------------------- 11 takeaways
s = slide()
title(s, "Takeaways")
bullets(s, [
    "DDPM beats the EDM baseline on FID and KID; EDM matches pixel intensities best.",
    "CFG 3.0 is the best operating point - higher CFG trades image quality for class fidelity.",
    "Classifier guidance only works with a noise-aware classifier.",
    "With one, CGD raises class fidelity (FR-I closest to real) at no cost in image quality.",
    "Every model still over-produces clear-cut FR-II and finds FR-I hardest.",
], top=1.6, size=18, height=4.5)

prs.save(OUT)
print("wrote", OUT)
