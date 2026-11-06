import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from spikingjelly.activation_based import functional
import matplotlib.pyplot as plt
from src.utils.paths import OUTPUT_ROOT
from src.utils.functional import set_step_mode



def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.eval()
    model.feedback_modules.train()  # Set feedback modules to training mode
    # Keep feedback modules in training mode.
    # This allows their BatchNorm layers to update.
    model.feedback_modules.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (clips, labels) in enumerate(train_loader):

        functional.reset_net(model)
        clips = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # B,T,C,H,W -> T,B,C,H,W
        clips = clips.permute(1, 0, 2, 3, 4).contiguous()
        optimizer.zero_grad(set_to_none=True)

        # Forward
        outputs = model(clips)

        # Temporal average
        outputs = outputs.mean(0)
        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()

        # Clip only feedback gradients
        torch.nn.utils.clip_grad_norm_(
            model.feedback_modules.parameters(),
            max_norm=1.0
        )
        optimizer.step()

        # Metrics
        batch_size = labels.size(0)
        predictions = outputs.argmax(dim=1)
        correct += predictions.eq(labels).sum().item()
        total += batch_size
        total_loss += loss.item() * batch_size

        if batch_idx % 20 == 0:
            print(
                f"Train [{batch_idx+1}/{len(train_loader)}] "
                f"Loss: {loss.item():.4f} "
                f"Acc: {100.0 * correct / total:.2f}%",
                flush=True
            )

    epoch_loss = total_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc

@torch.no_grad()
def validate_one_epoch(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for clips, labels in val_loader:
        functional.reset_net(model)
        clips = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # B,T,C,H,W -> T,B,C,H,W
        clips = clips.permute(1, 0, 2, 3, 4).contiguous()
        outputs = model(clips)

        # Temporal average
        outputs = outputs.mean(0)
        loss = criterion(outputs, labels)
        predictions = outputs.argmax(dim=1)
        batch_size = labels.size(0)
        correct += predictions.eq(labels).sum().item()
        total += batch_size
        total_loss += loss.item() * batch_size

    val_loss = total_loss / total
    val_acc = 100.0 * correct / total
    return val_loss, val_acc


def train_model(model, train_loader, val_loader, num_epochs, device):
    functional.set_step_mode(model.feedback_modules, step_mode='s')
    # ==========================================
    # 1. Freeze pretrained model
    # ==========================================
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze feedback modules only
    for param in model.feedback_modules.parameters():
        param.requires_grad = True

    # Verify trainable parameters
    trainable_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append((name, param.numel()))
    print("\n========== PARAMETER STATUS ==========")
    for name, count in trainable_params:
        print(f"{name}: {count:,}")
    print(
        "\nTotal trainable parameters:",
        sum(count for _, count in trainable_params)
    )
    print(
        "Total frozen parameters:",
        sum(p.numel() for p in model.parameters()
            if not p.requires_grad)
    )
    print("======================================")

    # ==========================================
    # 2. Loss, optimizer, scheduler
    # ==========================================
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.feedback_modules.parameters(),
        lr=1e-3,
        weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=num_epochs)

    # ==========================================
    # 3. Training history
    # ==========================================

    best_val_acc = -1.0
    best_val_loss = float("inf")

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    # ==========================================
    # 4. Training loop
    # ==========================================

    for epoch in range(num_epochs):

        print(f"\n{'='*55}")
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"{'='*55}")

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        val_loss, val_acc = validate_one_epoch(
            model,
            val_loader,
            criterion,
            device
        )

        scheduler.step()

        # Store metrics
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"\nTrain Loss: {train_loss:.4f}")
        print(f"Train Acc:  {train_acc:.2f}%")
        print(f"Val Loss:   {val_loss:.4f}")
        print(f"Val Acc:    {val_acc:.2f}%")

        print(f"LR: {optimizer.param_groups[0]['lr']:.7f}")

        # ==========================================
        # 5. Save best model
        # ==========================================

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss

            torch.save(
                model.state_dict(),
                OUTPUT_ROOT / "best_feedback_model.pth"
            )

            torch.save(
                model.feedback_modules.state_dict(),
                OUTPUT_ROOT / "best_feedback_connection.pth"
            )

            print("\n*** Best feedback model saved! ***")

        print("-" * 55)

    # ==========================================
    # 6. Save training history
    # ==========================================

    np.savez(
        OUTPUT_ROOT / "training_history.npz",
        **{
            k: np.asarray(v)
            for k, v in history.items()
        }
    )

    # ==========================================
    # 7. Plot accuracy and loss
    # ==========================================

    epochs = np.arange(1, num_epochs + 1)

    # -------- Accuracy plot --------

    plt.figure(figsize=(10, 6))
    plt.plot(
        epochs,
        history["train_acc"],
        marker="o",
        linewidth=2,
        label="Training Accuracy"
    )
    plt.plot(
        epochs,
        history["val_acc"],
        marker="s",
        linewidth=2,
        label="Validation Accuracy"
    )
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.title(
        "Feedback Connection Training and Validation Accuracy",
        fontsize=14,
        fontweight="bold"
    )
    plt.xticks(epochs)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(
        OUTPUT_ROOT / "feedback_accuracy_curve.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()
    plt.close()

    # -------- Loss plot --------
    plt.figure(figsize=(10, 6))
    plt.plot(
        epochs,
        history["train_loss"],
        marker="o",
        linewidth=2,
        label="Training Loss"
    )
    plt.plot(
        epochs,
        history["val_loss"],
        marker="s",
        linewidth=2,
        label="Validation Loss"
    )
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Cross Entropy Loss", fontsize=12)
    plt.title(
        "Feedback Connection Training and Validation Loss",
        fontsize=14,
        fontweight="bold"
    )
    plt.xticks(epochs)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(
        OUTPUT_ROOT / "feedback_loss_curve.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()
    plt.close()

    # ==========================================
    # 8. Final report
    # ==========================================

    print("\nTraining completed.")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"\nResults saved in: {OUTPUT_ROOT}")

    return history