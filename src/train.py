import torch
import torch.nn as nn
from tqdm import tqdm
import wandb

from dataset import EMOTION_LABELS
from utils import log_confusion_matrix, log_sample_predictions, log_grad_flow

WANDB_PROJECT = "fer-challenge"
WANDB_ENTITY  = "dshon23-free-university-of-tbilisi"


# ── Epoch helpers ─────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device, scheduler=None):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in tqdm(loader, leave=False, desc="train"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        # Gradient clipping — prevents exploding gradients in deeper nets
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, return_preds=False):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_labels, all_preds, sample_imgs = [], [], []
    collected_sample = False

    for images, labels in tqdm(loader, leave=False, desc="eval"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        preds = outputs.argmax(1)
        total_loss += loss.item() * labels.size(0)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if return_preds:
            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            if not collected_sample:
                sample_imgs = (images[:16].cpu(), labels[:16].cpu().tolist(), preds[:16].cpu().tolist())
                collected_sample = True

    if total == 0:
            if return_preds:
                return 0.0, 0.0, [], [], []
            return 0.0, 0.0
    return total_loss / total, correct / total


# ── Optimizer / Scheduler factory ─────────────────────────────────────────────

def get_optimizer(model, cfg):
    params = filter(lambda p: p.requires_grad, model.parameters())
    if cfg.get('optimizer', 'adam').lower() == 'sgd':
        return torch.optim.SGD(params, lr=cfg['lr'], momentum=0.9, weight_decay=1e-4)
    return torch.optim.Adam(params, lr=cfg['lr'], weight_decay=cfg.get('weight_decay', 1e-4))


def get_scheduler(optimizer, cfg):
    sched = cfg.get('scheduler', 'plateau')
    if sched == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epochs'])
    if sched == 'step':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                       patience=5, factor=0.5)


# ── Main training function ─────────────────────────────────────────────────────

def train_model(model, cfg, train_loader, val_loader, device, class_weights=None):
    """
    Trains `model` according to `cfg` and logs everything to WandB.

    cfg keys:
        run_name, architecture, epochs, lr, optimizer, scheduler,
        batch_size, dropout, augment, weight_decay
    """
    run = wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=cfg['run_name'],
        config=cfg,
        reinit=True,
    )
    wandb.watch(model, log='all', log_freq=50)

    weight = class_weights.to(device) if class_weights is not None else None
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = get_optimizer(model, cfg)
    scheduler = get_scheduler(optimizer, cfg)
    plateau = isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)

    best_val_acc = 0.0
    best_state = None
    patience_counter = 0
    early_stop_patience = cfg.get('early_stop_patience', 15)

    for epoch in range(1, cfg['epochs'] + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        current_lr = optimizer.param_groups[0]['lr']

        run.log({
            'epoch':      epoch,
            'train/loss': train_loss,
            'train/acc':  train_acc,
            'val/loss':   val_loss,
            'val/acc':    val_acc,
            'lr':         current_lr,
        })

        # Log gradient flow every 10 epochs
        if epoch % 10 == 0:
            log_grad_flow(model, run, epoch)

        if plateau:
            scheduler.step(val_loss)
        else:
            scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        print(f"Epoch {epoch:3d}/{cfg['epochs']} | "
              f"train loss={train_loss:.4f} acc={train_acc:.3f} | "
              f"val loss={val_loss:.4f} acc={val_acc:.3f} | "
              f"lr={current_lr:.6f}")

        if patience_counter >= early_stop_patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    # Restore best weights and run final evaluation
    if best_state:
        model.load_state_dict(best_state)

    _, _, all_labels, all_preds, sample = evaluate(
        model, val_loader, criterion, device, return_preds=True)

    log_confusion_matrix(all_labels, all_preds, run)
    if sample:
        imgs, lbls, preds = sample
        log_sample_predictions(imgs, lbls, preds, run)

    run.summary['best_val_acc'] = best_val_acc
    run.finish()

    return model, best_val_acc


# ── Quick overfit test ─────────────────────────────────────────────────────────

def overfit_single_batch(model, loader, device, steps=50):
    """
    Try to overfit on a single batch.
    If loss doesn't drop to ~0, model or data pipeline has a bug.
    """
    model.train()
    images, labels = next(iter(loader))
    images, labels = images.to(device), labels.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    print("Single-batch overfit test:")
    for step in range(steps):
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        if (step + 1) % 10 == 0:
            acc = (model(images).argmax(1) == labels).float().mean().item()
            print(f"  step {step+1:3d} | loss={loss.item():.4f} | acc={acc:.3f}")

    model.train()
