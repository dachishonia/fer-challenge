# Facial Expression Recognition Challenge

**Kaggle:** [Challenges in Representation Learning: FER2013](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge)  
**WandB:** [dshon23/fer-challenge](https://wandb.ai/dshon23/fer-challenge)

---

## Dataset

- **48×48 grayscale images**, 7 emotion classes
- 28,709 training / 3,589 validation / 3,589 test samples
- Class imbalance: `Disgust` has ~10× fewer samples than `Happy` → fixed with class-weighted loss

| Label | Emotion  |
|-------|----------|
| 0     | Angry    |
| 1     | Disgust  |
| 2     | Fear     |
| 3     | Happy    |
| 4     | Sad      |
| 5     | Surprise |
| 6     | Neutral  |

---

## Architecture Progression

The key principle: **start small and add capacity/regularization iteratively**. Each architecture fixes a specific problem found in the previous one.

### v1 — TinyCNN (Underfitting Baseline)

```
Conv(1→8) → ReLU → MaxPool
Conv(8→16) → ReLU → MaxPool
Linear(2304→7)
```

**Decision:** Start with the smallest possible model (2 conv layers, ~50K params) to establish an underfitting baseline and verify the training pipeline is correct.

**Result:** ~35-40% val accuracy. Train and val loss are both high — classic underfitting. The model lacks capacity to learn discriminative facial features.

**Diagnosis:** The bottleneck is model capacity, not regularization. Moving to more conv layers is the right fix.

---

### v2 — SmallCNN (Overfitting Demonstration)

```
Conv(1→32) → ReLU → MaxPool ×4
Linear(1152→512) → Linear(512→7)
```

**Decision:** 4 conv layers, ~1.2M params, **no regularization** (no BatchNorm, no Dropout). Goal is to intentionally demonstrate overfitting.

**Result:** Train accuracy climbs to ~75%, but val accuracy plateaus at ~45-50%. Large gap = overfitting.

**Diagnosis:** The model memorizes training samples instead of generalizing. Fix: add BatchNorm and Dropout.

---

### v3 — MediumCNN (Regularization)

```
[Conv → BatchNorm → ReLU → MaxPool → Dropout2d] ×4
Linear(2304→512) → Dropout → Linear(512→128) → Linear(128→7)
```

**Decision:** Add BatchNorm to stabilize internal covariate shift (faster convergence, acts as mild regularizer) and Dropout to prevent neuron co-adaptation. Also add data augmentation (horizontal flip, rotation, crop).

**Tested:** dropout=0.3 vs 0.5 vs 0.4+cosine LR schedule.

**Result:** ~55-60% val accuracy. Train/val gap is much smaller. Dropout=0.4 with cosine LR works best.

**Diagnosis:** Regularization works. Cosine annealing helps avoid local minima late in training.

---

### v4 — ResNetStyleCNN (Residual Connections)

```
Stem: Conv(1→32)
Stage1: ResBlock(32) → Pool
Stage2: Conv(32→64) → ResBlock(64) → Pool
Stage3: Conv(64→128) → ResBlock(128) → Pool
Stage4: Conv(128→256) → ResBlock(256) → Pool
GlobalAvgPool → Linear(256→7)
```

**Decision:** Skip connections allow gradients to flow directly to early layers, solving vanishing gradient in deep networks. GlobalAvgPool replaces large FC layers (fewer parameters, better generalization).

**Result:** ~60-65% val accuracy. Training is more stable with smaller oscillations in loss curve.

**Diagnosis:** Residual connections clearly help. The gradient flow visualization in WandB confirms early layers now receive meaningful gradients.

---

### v5 — Transfer Learning (MobileNetV2)

```
MobileNetV2 (ImageNet pretrained)
  → last 3-6 blocks unfrozen
  → AdaptiveAvgPool
  → Dropout → Linear(1280→256) → Linear(256→7)
```

**Decision:** Facial texture and edge features learned on ImageNet transfer well to face expression recognition. The 48×48 images are resized to 224×224 and converted to 3-channel for compatibility.

**Tested:** (a) freeze all but last 3 blocks vs (b) fine-tune 6 blocks with lower LR.

**Result:** ~65-70% val accuracy. Fine-tuning more layers with lower LR (1e-4) outperforms shallow fine-tuning.

**Diagnosis:** ImageNet pretrained features are significantly better than training from scratch for this dataset size. The key insight is using a very low LR when fine-tuning to avoid destroying pretrained weights.

---

## WandB Runs Summary

All 12 runs are logged at [wandb.ai/dshon23/fer-challenge](https://wandb.ai/dshon23/fer-challenge).

Each run logs:
- `train/loss`, `train/acc` per epoch
- `val/loss`, `val/acc` per epoch
- `lr` (learning rate schedule)
- Gradient flow plot every 10 epochs
- Confusion matrix (final)
- 16 sample predictions (final)
- Forward/backward sanity check metrics

---

## Repository Structure

```
fer-facial-expression-recognition/
├── notebooks/
│   └── fer_experiments.ipynb   # Main Colab notebook
├── src/
│   ├── dataset.py              # FERDataset, transforms, dataloaders
│   ├── models.py               # All 5 architectures + registry
│   ├── train.py                # Training loop + WandB logging
│   └── utils.py                # Sanity checks, confusion matrix, grad flow
├── configs/
│   └── hyperparams.yaml        # All experiment configs
├── requirements.txt
└── README.md
```

---

## How to Run

1. Open `notebooks/fer_experiments.ipynb` in Google Colab (T4 GPU)
2. Fill in Kaggle credentials in Cell 3
3. Run `wandb.login()` with your API key in Cell 4
4. Run all cells sequentially — each section is one architecture

Total estimated runtime: ~4-6 hours on T4 GPU for all 12 runs.
