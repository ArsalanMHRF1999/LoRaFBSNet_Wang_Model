import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Subset, DataLoader
from spikingjelly.activation_based import functional, neuron

from src.utils.paths import WEIGHTS_ROOT, OUTPUT_ROOT
from src.utils.dataset_configs import BATCH_SIZE, NUM_WORKERS
from src.dataset.transformer import safe_video_collate
from src.dataset.dataloaders import create_dataloaders
from src.train_feedback.load_weights import create_model_with_feedback
from src.utils.train_helper import configure_feedback_training, set_feedback_train_mode

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 101

# 1. Build model with (4 -> 3) feedback
model = create_model_with_feedback(
    WEIGHTS_ROOT,
    feedback_connection=[(4, 3)],
    num_classes=NUM_CLASSES,
    device=DEVICE,
)
functional.set_step_mode(model.feedback_modules, step_mode="s")
trainable_params = configure_feedback_training(model)

# 2. Small fixed subsets for fast Kaggle iteration
full_train_ds, _ = create_dataloaders("train")
full_val_ds, _ = create_dataloaders("val")

g = torch.Generator().manual_seed(42)
train_idx = torch.randperm(len(full_train_ds), generator=g)[:256].tolist()
val_idx = torch.randperm(len(full_val_ds), generator=g)[:128].tolist()

small_train_loader = DataLoader(
    Subset(full_train_ds, train_idx),
    batch_size=min(BATCH_SIZE, 16),
    shuffle=True,
    num_workers=min(NUM_WORKERS, 2),
    pin_memory=True,
    collate_fn=safe_video_collate,
)
small_val_loader = DataLoader(
    Subset(full_val_ds, val_idx),
    batch_size=min(BATCH_SIZE, 16),
    shuffle=False,
    num_workers=min(NUM_WORKERS, 2),
    pin_memory=True,
    collate_fn=safe_video_collate,
)

# 3. Diagnostics storage
diag = {
    "train_loss": [],
    "val_loss": [],
    "train_acc": [],
    "val_acc": [],
    "fb_weight_norm": [],
    "raw_grad_norm": [],
    "fb_act_std": [],
    "orig_act_std": [],
    "spike_rate_l3": [],
}

# Hook target stage (layer 3) input/output stats
act_cache = {}

def hook_layer3_in(module, inp, out):
    # inp[0] is x_t, inp[1] is external_feedback (if passed)
    act_cache["x_std"] = inp[0].detach().float().std().item()
    if len(inp) > 1 and inp[1] is not None and module.feedback_pwconv is not None:
        with torch.no_grad():
            fb_proj = module.feedback_bn(module.feedback_pwconv(inp[1]))
            act_cache["fb_std"] = fb_proj.float().std().item()
    else:
        act_cache["fb_std"] = 0.0

def hook_sn_out(module, inp, out):
    act_cache["l3_sn_rate"] = out.detach().float().mean().item()

h1 = model.recurrent_layer3.register_forward_hook(hook_layer3_in)
h2 = model.recurrent_layer3.sn.register_forward_hook(hook_sn_out)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(trainable_params, lr=1e-4, weight_decay=1e-3)

for epoch in range(10):
    set_feedback_train_mode(model)
    tr_loss, tr_correct, tr_total = 0.0, 0, 0
    epoch_raw_grads, epoch_fb_stds, epoch_x_stds, epoch_rates = [], [], [], []

    for clips, labels in small_train_loader:
        functional.reset_net(model)
        clips = clips.to(DEVICE, non_blocking=True).permute(1, 0, 2, 3, 4).contiguous()
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(clips).mean(0)
        loss = criterion(outputs, labels)
        loss.backward()

        raw_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in trainable_params if p.grad is not None],
            max_norm=1.0,
        ).item()
        optimizer.step()

        bs = labels.size(0)
        tr_loss += loss.item() * bs
        tr_correct += outputs.argmax(dim=1).eq(labels).sum().item()
        tr_total += bs

        epoch_raw_grads.append(raw_norm)
        epoch_fb_stds.append(act_cache.get("fb_std", 0.0))
        epoch_x_stds.append(act_cache.get("x_std", 0.0))
        if "l3_sn_rate" in act_cache:
            epoch_rates.append(act_cache["l3_sn_rate"])

    model.eval()
    va_loss, va_correct, va_total = 0.0, 0, 0
    with torch.no_grad():
        for clips, labels in small_val_loader:
            functional.reset_net(model)
            clips = clips.to(DEVICE, non_blocking=True).permute(1, 0, 2, 3, 4).contiguous()
            labels = labels.to(DEVICE, non_blocking=True)
            outputs = model(clips).mean(0)
            loss = criterion(outputs, labels)
            bs = labels.size(0)
            va_loss += loss.item() * bs
            va_correct += outputs.argmax(dim=1).eq(labels).sum().item()
            va_total += bs

    fb_w_norm = torch.sqrt(
        sum((p.detach().float() ** 2).sum() for p in trainable_params)
    ).item()

    diag["train_loss"].append(tr_loss / tr_total)
    diag["val_loss"].append(va_loss / va_total)
    diag["train_acc"].append(100.0 * tr_correct / tr_total)
    diag["val_acc"].append(100.0 * va_correct / va_total)
    diag["fb_weight_norm"].append(fb_w_norm)
    diag["raw_grad_norm"].append(float(np.mean(epoch_raw_grads)))
    diag["fb_act_std"].append(float(np.mean(epoch_fb_stds)))
    diag["orig_act_std"].append(float(np.mean(epoch_x_stds)))
    diag["spike_rate_l3"].append(float(np.mean(epoch_rates)) if epoch_rates else 0.0)

    print(
        f"Ep {epoch+1:02d} | "
        f"Tr Loss: {diag['train_loss'][-1]:.4f} Acc: {diag['train_acc'][-1]:.1f}% | "
        f"Val Loss: {diag['val_loss'][-1]:.4f} Acc: {diag['val_acc'][-1]:.1f}% | "
        f"||W_fb||: {fb_w_norm:.4f} | fb/x std: {diag['fb_act_std'][-1]:.4f}/{diag['orig_act_std'][-1]:.4f}"
    )

h1.remove()
h2.remove()

# 4. Plot diagnostic dashboard
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
ep_axis = np.arange(1, len(diag["train_loss"]) + 1)

axes[0, 0].plot(ep_axis, diag["train_loss"], label="Train Loss")
axes[0, 0].plot(ep_axis, diag["val_loss"], label="Val Loss")
axes[0, 0].set_title("Loss (Overfit Check)")
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

axes[0, 1].plot(ep_axis, diag["train_acc"], label="Train Acc %")
axes[0, 1].plot(ep_axis, diag["val_acc"], label="Val Acc %")
axes[0, 1].set_title("Accuracy")
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

axes[1, 0].plot(ep_axis, diag["fb_act_std"], label="Feedback Proj Std")
axes[1, 0].plot(ep_axis, diag["orig_act_std"], label="Layer 3 Input Std")
axes[1, 0].set_title("Feedback vs. Feedforward Scale")
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

axes[1, 1].plot(ep_axis, diag["fb_weight_norm"], label="Trainable ||W||_2")
axes[1, 1].plot(ep_axis, diag["raw_grad_norm"], label="Raw Grad Norm")
axes[1, 1].plot(ep_axis, diag["spike_rate_l3"], label="Layer 3 SN Spike Rate")
axes[1, 1].set_title("Weights, Gradients & Spiking")
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_ROOT / "kaggle_overfit_diagnostics.png", dpi=200)
plt.show()