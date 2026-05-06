# Project Progress

## Models

### classification
- ResNet50 fine-tuned on MiraBest (FR-I vs FR-II binary classification)
- Dataset: CRUMB (CIFAR-style PNG batches)
- No FITS support
- Known bug: `model.loa. _state_dict(...)` typo at `train_pipeline.py:67` — crashes on resume

### robust_classification
- `TimeDependentResNet`: ResNet50 with sinusoidal timestep embedding, enabling noise-level conditioning
- Training: warmup (noisy images only) → transition (progressive PGD epsilon) → full adversarial (PGD on noisy images)
- Dataset: PNG only (`mirabest` or `mirabest_fits_png`)
- Config currently says `mirabest_fits` — this is wrong. `train_robust_classifier_guided_diffusion` applies a `fits_to_linear` conversion before passing images to the classifier, which assumes the classifier was trained on linear-normalised PNG images. Training on FITS would break that assumption.
- Saves both latest and best-val-acc checkpoints separately

### diffusion
- `UNet2DConditionModel` (HuggingFace diffusers) + `DDPMScheduler`
- Class-conditional with classifier-free guidance (15% label dropout, `guidance_scale=7.5`)
- Datasets: `mirabest` / `mirabest_fits` / `crumb` — full FITS + PNG support
- Saves generated images every 5 epochs; saves FITS files if trained on `mirabest_fits`
- `FrechetInceptionDistance` is imported and `fid_history` is tracked but FID is never actually computed

### pid (physics-informed diffusion)
- Extends `diffusion` with training-time physics constraints applied to the Tweedie x_0 estimate
- Physics losses (defined in `src/models/pid.py`):
  - `symmetry_loss`: H-flip + V-flip MSE — enforces bilateral point-symmetry of AGN jets
  - `nonnegativity_loss`: ReLU penalty on sub-zero pixels — enforces non-negative flux
  - `loss = MSE + λ_sym·symmetry + λ_neg·nonnegativity`
- Post-sampling: `apply_nonnegativity` clamps generated images to >= 0
- Tracks physics compliance on generated samples every 5 epochs (`pct_negative_history`, `sym_score_history`)
- Datasets: full FITS + PNG support, same as diffusion

### robust_classifier_guided_diffusion (not yet wired up)
- Extends `diffusion`: frozen `TimeDependentResNet` penalises the UNet when `x_0_pred` is misclassified
- `loss = MSE + λ_cls·CrossEntropy(classifier(x_0_pred), labels)`
- Not registered in the `train_model` dispatcher — dead code
- Has a double-init bug: `build_diffusion_components` is called then `scheduler`, `class_emb`, and `optimizer` are immediately overwritten

---

## Open Questions

### Generated Image Quality
No evaluation metric is currently in use. Options considered:

- **FID (Inception)**: unsuitable — InceptionV3 is ImageNet-pretrained, no radio morphology prior
- **FID with fine-tuned ResNet50 features**: better domain fit, but still partially ImageNet-biased
- **FID with scratch-trained encoder**: cleanest, but MiraBest (~150 images) is too small for a stable Gaussian fit
- **Classifier confidence on generated images**: does the fine-tuned ResNet50 correctly classify generated FR-I/FR-II at high confidence? Directly meaningful, already half-wired in `robust_classifier_guided_diffusion`
- **Physics compliance metrics**: `pct_negative_history`, `sym_score_history` — already implemented in `train_pid`, grounded in physical constraints
- **Pixel-space distribution statistics**: integrated flux, source extent (second moment), core-to-lobe flux ratio — compare histograms of real vs generated

### Physics Constraints
Current constraints (symmetry, non-negativity) are morphological. Potential additions:
- Flux conservation: total integrated flux of generated images should match real distribution
- PSF consistency: generated images should be consistent with the instrument beam
- Spectral index constraints (if multi-frequency data is available)

### Next Steps
- [ ] Wire up `robust_classifier_guided_diffusion` in the dispatcher
- [ ] Fix double-init bug in `train_robust_classifier_guided_diffusion`
- [ ] Combine physics constraints (pid) + classifier guidance into a single `train_pid_classifier_guided` pipeline
- [ ] Implement at least one quality metric before comparing pipelines
