import numpy as np
import torch
import torch.nn as nn
import wandb
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

from dataset import EMOTION_LABELS


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ── Sanity Checks ─────────────────────────────────────────────────────────────

def forward_sanity_check(model, loader, device):
    """
    Verify the forward pass:
    1. Output shape is (batch, 7)
    2. Initial loss is near -log(1/7) ≈ 1.946 (random model)
    3. No NaNs in output
    """
    model.eval()
    images, labels = next(iter(loader))
    images, labels = images.to(device), labels.to(device)

    with torch.no_grad():
        outputs = model(images)

    assert outputs.shape[1] == 7, f"Bad output shape: {outputs.shape}"
    assert not torch.isnan(outputs).any(), "NaN in forward pass!"

    criterion = nn.CrossEntropyLoss()
    loss = criterion(outputs, labels).item()
    expected = -np.log(1 / 7)

    print(f"  [Forward Check]")
    print(f"    Output shape : {tuple(outputs.shape)}")
    print(f"    Initial loss : {loss:.4f}  (expected ~{expected:.4f} for random model)")
    print(f"    NaN detected : False")

    return {
        "sanity/initial_loss": loss,
        "sanity/expected_loss": expected,
        "sanity/loss_ratio": loss / expected,
    }


def backward_sanity_check(model, loader, device):
    """
    Verify gradient flow:
    1. Every trainable parameter receives a gradient
    2. No gradients are exactly zero (dead layers)
    3. Log per-layer gradient norms
    """
    model.train()
    images, labels = next(iter(loader))
    images, labels = images.to(device), labels.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    optimizer.zero_grad()
    loss = criterion(model(images), labels)
    loss.backward()

    grad_stats = {}
    dead_layers = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.grad is None:
            dead_layers.append(name)
            continue
        norm = param.grad.norm().item()
        grad_stats[f"gradients/{name}"] = norm
        if norm == 0.0:
            dead_layers.append(name)

    norms = list(grad_stats.values())
    print(f"  [Backward Check]")
    print(f"    Layers checked  : {len(grad_stats)}")
    print(f"    Dead layers     : {dead_layers if dead_layers else 'None'}")
    if norms:
        print(f"    Grad norm min   : {min(norms):.6f}")
        print(f"    Grad norm max   : {max(norms):.6f}")

    optimizer.zero_grad()
    return grad_stats


def run_sanity_checks(model, loader, device, run=None):
    print("\n─── Sanity Checks ───────────────────────────────────────────")
    fwd = forward_sanity_check(model, loader, device)
    bwd = backward_sanity_check(model, loader, device)
    print("─────────────────────────────────────────────────────────────\n")

    if run is not None:
        run.log({**fwd, **bwd})

    return fwd, bwd


# ── WandB Helpers ─────────────────────────────────────────────────────────────

def log_confusion_matrix(all_labels, all_preds, run):
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=EMOTION_LABELS,
                yticklabels=EMOTION_LABELS, cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    run.log({"confusion_matrix": wandb.Image(fig)})
    plt.close(fig)


def log_sample_predictions(images, labels, preds, run, n=16):
    imgs = []
    for i in range(min(n, len(images))):
        img = images[i].cpu()
        if img.shape[0] == 3:
            img = img.permute(1, 2, 0).numpy()
            img = (img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])).clip(0, 1)
        else:
            img = img.squeeze().numpy()
            img = (img * 0.5 + 0.5).clip(0, 1)
        true_label = EMOTION_LABELS[labels[i]]
        pred_label = EMOTION_LABELS[preds[i]]
        caption = f"T:{true_label} | P:{pred_label}"
        imgs.append(wandb.Image(img, caption=caption))
    run.log({"sample_predictions": imgs})


def log_grad_flow(model, run, step):
    """Log mean absolute gradient per layer as a bar chart to WandB."""
    named_grads = {}
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            named_grads[name] = param.grad.abs().mean().item()

    if not named_grads:
        return

    fig, ax = plt.subplots(figsize=(max(6, len(named_grads) * 0.4), 4))
    ax.bar(range(len(named_grads)), named_grads.values())
    ax.set_xticks(range(len(named_grads)))
    ax.set_xticklabels(named_grads.keys(), rotation=90, fontsize=6)
    ax.set_title(f"Gradient Flow (step {step})")
    plt.tight_layout()
    run.log({"grad_flow": wandb.Image(fig)}, step=step)
    plt.close(fig)
