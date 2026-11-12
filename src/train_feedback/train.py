import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from spikingjelly.activation_based import functional
import matplotlib.pyplot as plt
from src.utils.paths import OUTPUT_ROOT
from src.utils.train_helper import set_feedback_train_mode, configure_feedback_training
from src.model.lorafb_snet import ConvRecurrentContainer

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    set_feedback_train_mode(model)
    print(model.recurrent_layer3.sn.training)   # should now print True
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

        # Clip all trainable feedback gradients
        torch.nn.utils.clip_grad_norm_(
            [
                p for p in model.parameters()
                if p.requires_grad and p.grad is not None
            ],
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

    if total == 0:
        raise RuntimeError("Training loader is empty!")

    return total_loss / total, 100.0 * correct / total

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

    if total == 0:
        raise RuntimeError("Validation loader is empty!")

    return total_loss / total, 100.0 * correct / total



def train_model(
    model,
    train_loader,
    val_loader,
    num_epochs,
    device,
    patience=7,
    min_delta_loss=1e-4,
    min_delta_acc=0.1
):

    # ==========================================
    # 1. Configure feedback-only training
    # ==========================================

    functional.set_step_mode(model.feedback_modules, step_mode='s')

    trainable_params = configure_feedback_training(model)

    trainable = [
        n for n, p in model.named_parameters()
        if p.requires_grad
    ]

    print("Trainable parameters:", len(trainable))
    print(trainable[:10])

    # ==========================================
    # 2. Loss, optimizer, scheduler
    # ==========================================

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        trainable_params,
        lr=3e-4,
        weight_decay=1e-4
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs
    )

    # ==========================================
    # 3. Training history
    # ==========================================

    best_val_acc = -1.0
    best_val_loss = float("inf")

    # Early stopping trackers
    early_best_loss = float("inf")
    early_best_acc = -1.0

    patience_counter = 0

    # Save best model in memory
    best_model_state = None

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # ==========================================
    # 4. Training loop
    # ==========================================

    for epoch in range(num_epochs):

        print(f"\n{'='*55}")
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"{'='*55}")

        # --------------------------------------
        # Training
        # --------------------------------------

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        # --------------------------------------
        # Validation
        # --------------------------------------

        val_loss, val_acc = validate_one_epoch(
            model,
            val_loader,
            criterion,
            device
        )

        scheduler.step()

        # --------------------------------------
        # Store metrics
        # --------------------------------------

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        current_lr = optimizer.param_groups[0]["lr"]

        print(f"\nTrain Loss: {train_loss:.4f}")
        print(f"Train Acc:  {train_acc:.2f}%")
        print(f"Val Loss:   {val_loss:.4f}")
        print(f"Val Acc:    {val_acc:.2f}%")
        print(f"LR:         {current_lr:.7f}")

        # ==========================================
        # 5. Save best model
        # ==========================================

        is_best = (
            val_acc > best_val_acc
            or (
                val_acc == best_val_acc
                and val_loss < best_val_loss
            )
        )

        if is_best:

            best_val_acc = val_acc
            best_val_loss = val_loss

            # Keep best weights in memory
            best_model_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

            # Save complete model
            torch.save(
                best_model_state,
                OUTPUT_ROOT / "best_feedback_model.pth"
            )

            # Save feedback-related parameters
            feedback_state = {}

            for name, module in model.named_modules():

                if name == "feedback_modules":

                    feedback_state["feedback_modules"] = (
                        module.state_dict()
                    )

                elif isinstance(module, ConvRecurrentContainer):

                    if module.feedback_pwconv is not None:

                        feedback_state[
                            f"{name}.feedback_pwconv"
                        ] = module.feedback_pwconv.state_dict()

                        feedback_state[
                            f"{name}.feedback_bn"
                        ] = module.feedback_bn.state_dict()

            torch.save(
                feedback_state,
                OUTPUT_ROOT / "best_feedback_connection.pth"
            )

            print("\n*** Best feedback model saved! ***")

        # ==========================================
        # 6. Early Stopping
        # ==========================================

        loss_improved = (
            val_loss < early_best_loss - min_delta_loss
        )

        acc_improved = (
            val_acc > early_best_acc + min_delta_acc
        )

        if loss_improved:
            early_best_loss = val_loss

        if acc_improved:
            early_best_acc = val_acc

        if loss_improved or acc_improved:

            patience_counter = 0

            print("Early stopping: Validation improved.")

        else:

            patience_counter += 1

            print(
                f"No significant improvement: "
                f"{patience_counter}/{patience}"
            )

        print("-" * 55)

        # --------------------------------------
        # Stop training if patience is exhausted
        # --------------------------------------

        if patience_counter >= patience:

            print("\nEarly stopping triggered!")
            print(f"Training stopped at epoch {epoch+1}")

            break

    # ==========================================
    # 7. Restore best model
    # ==========================================

    if best_model_state is not None:

        model.load_state_dict(best_model_state)

        print("\nBest model weights restored.")

    # ==========================================
    # 8. Save training history
    # ==========================================

    np.savez(
        OUTPUT_ROOT / "training_history.npz",
        **{
            key: np.asarray(value)
            for key, value in history.items()
        }
    )

    # ==========================================
    # 9. Plot accuracy
    # ==========================================

    actual_epochs = len(history["train_acc"])
    epochs = np.arange(1, actual_epochs + 1)

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

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Feedback Connection Training Accuracy")
    plt.xticks(epochs)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_ROOT / "feedback_accuracy_curve.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()

    # ==========================================
    # 10. Plot loss
    # ==========================================

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

    plt.xlabel("Epoch")
    plt.ylabel("Cross Entropy Loss")
    plt.title("Feedback Connection Training Loss")
    plt.xticks(epochs)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_ROOT / "feedback_loss_curve.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()

    # ==========================================
    # 11. Final report
    # ==========================================

    print("\nTraining completed.")
    print(f"Epochs executed: {actual_epochs}/{num_epochs}")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Results saved in: {OUTPUT_ROOT}")

    return history
