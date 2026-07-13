# Classifier architecture: ResNet18 transfer learning vs. training from scratch

## Background

`train_classification` originally fine-tuned a pretrained `resnet18` (ImageNet
weights, 3-channel input) on CRUMB's ~1.4k-image training split. Validation
loss diverged from very early in training (best epoch ~6 of 100) while
training loss kept falling — classic overfitting for a dataset this small
relative to the model's capacity.

## Attempt 1: stronger regularization on ResNet18

Raised `weight_decay` (0.001 → 0.01) and `dropout` (0.2 → 0.5) on the
fine-tuned `fc` head. Best val loss and test accuracy were essentially
unchanged (0.6104 vs. 0.6086 baseline) — regularizing the head alone had
almost no effect, because the actual overfitting was coming from `layer4`
(millions of parameters) still being fully trainable against ~1.4k images.

## Attempt 2: freeze the entire ResNet18 backbone

Froze all of `layer1`–`layer4`, leaving only the `fc` head trainable —
effectively a linear probe on frozen ImageNet features.

| | layer4 unfrozen | fully frozen |
|---|---|---|
| Best val loss | 0.6104 | **0.6272** (worse) |
| Test accuracy | 73.68% | **71.23%** (worse) |

This *reduced* the train/val gap but made both val loss and test accuracy
worse — i.e. it traded overfitting for underfitting. With every convolutional
layer frozen, there wasn't enough task-specific capacity left to fit CRUMB's
actual morphology, even with the overfitting problem gone. **Conclusion: a
pretrained ResNet18 is a poor fit for this dataset in either direction** — too
many trainable parameters relative to ~1.4k images if unfrozen, and not
enough freedom to learn the task if frozen. Generic ImageNet features
(edges/textures from natural photos) are also a loose match for grayscale
radio-galaxy morphology to begin with.

## Attempt 3: train a small CNN from scratch instead

Switched to `SimpleCNN` (`src/models/simple_cnn.py`) — a lightweight
from-scratch architecture, no pretrained weights, no frozen layers. Went
through several rounds of tuning:

1. **2 conv blocks (16→32 channels)**, LR carried over from the old
   ResNet18-fc-tuning config (`0.000134`) — badly underfit; training loss
   barely moved (0.708→0.668, hovering near chance-level `ln(2)=0.693`).
   Root cause: that LR was tuned for fine-tuning a linear head, not for
   training conv filters from random init, which needs an LR an order of
   magnitude higher.
2. Raised LR to `0.001` — fixed the stuck-training-loss problem, but
   introduced huge validation-loss spikes (up into the teens). Root cause:
   `weight_decay` was being applied to `BatchNorm`'s `gamma`/`beta`
   parameters via a single undifferentiated `Adam(model.parameters(), ...)`
   call, which destabilizes normalization.
3. Dropped weight decay entirely (dropout is `SimpleCNN`'s only
   regularizer) — spikes shrank but didn't disappear.
4. **3 conv blocks (16→32→64 channels)** — confirmed capacity, not weight
   decay, was the earlier limiting factor: train loss now dropped properly
   (0.70→0.55) and best val loss improved to 0.5809. But spikes got *worse*
   (peak 16.15), ruling out capacity as the spike's cause and pointing at
   `BatchNorm` itself: with a small batch size (16), an unusual batch
   periodically corrupts BatchNorm's running mean/var, and every eval-mode
   validation pass uses those corrupted stats until the next recovery —
   exactly the spike-then-recover pattern observed.
5. **Replaced `BatchNorm2d` with `GroupNorm(4, channels)`** in all 3 blocks
   — normalizes within each image, no batch statistics, no running average,
   no train/eval mismatch. This eliminated the spikes outright (max val loss
   across a full 300-epoch run dropped from double digits to ~0.76, smooth
   throughout).
6. **Raised `batch_size` 16 → 32** — reduced remaining gradient/GroupNorm
   noise and smoothed the validation curve further.

### Best result

3-block `SimpleCNN` (16→32→64 channels, `GroupNorm`, no weight decay,
dropout 0.6, `lr=0.001`, `batch_size=32`, 300 epochs):

- Best val loss: **0.5780**
- Test accuracy: **76.84%**

This is the best result obtained across every configuration tried,
**comparable to (and slightly better than) the original overfit ResNet18
transfer-learning run (74.74%)**, using a far smaller model with no
pretrained weights at all — supporting the conclusion that ResNet18's
capacity was mismatched to this dataset in both the unfrozen and frozen
configurations, and a lightweight from-scratch CNN is at least as effective.

Verified stable across 5 seeds (`--runs 5`) rather than resting on a single
lucky run — see `results/classification/` for the per-seed logs.

## Summary

| Configuration | Best val loss | Test accuracy | Note |
|---|---|---|---|
| ResNet18, layer4 unfrozen (baseline) | 0.6086 | 74.74% | Overfits — val diverges from epoch ~6 |
| ResNet18, layer4 unfrozen, +weight decay/dropout | 0.6104 | 73.68% | Regularizing the head alone did nothing |
| ResNet18, fully frozen (linear probe) | 0.6272 | 71.23% | Underfits — not enough task-specific capacity |
| SimpleCNN, 2 blocks, low LR | — | — | Underfits — LR too low for from-scratch training |
| SimpleCNN, 2 blocks, LR fixed, weight decay on BN | — | 65.61% | Huge val-loss spikes (BatchNorm + weight decay) |
| SimpleCNN, 3 blocks, BatchNorm | 0.5809 | 72.98% | Spikes persisted/worsened (BatchNorm + small batch) |
| SimpleCNN, 3 blocks, GroupNorm | 0.5986 | 67.02% | Spikes fixed; mild train/val divergence returns at 300 epochs |
| **SimpleCNN, 3 blocks, GroupNorm, batch_size=32** | **0.5780** | **76.84%** | Best result |

Related: [`crumb_test_label_corruption.md`](./crumb_test_label_corruption.md)
documents why CRUMB's test-split labels needed correcting before any of
these accuracy numbers were meaningful — all experiments in this document
were run after that fix was in place.
