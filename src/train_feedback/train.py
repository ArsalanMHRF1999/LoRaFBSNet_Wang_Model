import json
import signal
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from spikingjelly.activation_based import functional
import matplotlib.pyplot as plt
from src.utils.paths import OUTPUT_ROOT
from src.utils.train_helper import set_feedback_train_mode, configure_feedback_training
from src.model.lorafb_snet import ConvRecurrentContainer


_STOP_REQUESTED = False


def _signal_stop_handler(signum, frame):
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    print(
        f"\n[Signal {signum}] Stop/timeout signal received. "
        "Will save checkpoint, history, and plots safely...",
        flush=True
    )


def _save_plots_and_history(history, out_dir, suffix=""):
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{suffix}" if suffix else ""

    # Save history in both NPZ and JSON formats
    np_payload = {
        key: np.asarray(value)
        for key, value in history.items()
    }
    np.savez(out_dir / f"training_history{tag}.npz", **np_payload)

    json_payload = {
        key: [float(v) for v in value]
        for key, value in history.items()
    }
    with open(out_dir / f"training_history{tag}.json", "w") as f:
        json.dump(json_payload, f, indent=2)

    actual_epochs = len(history.get("train_acc", []))
    if actual_epochs == 0:
        return

    epochs = np.arange(1, actual_epochs + 1)

    # Accuracy plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history["train_acc"], marker="o", linewidth=2, label="Training Accuracy")
    plt.plot(epochs, history["val_acc"], marker="s", linewidth=2, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Feedback Connection Training Accuracy")
    plt.xticks(epochs)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"feedback_accuracy_curve{tag}.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history["train_loss"], marker="o", linewidth=2, label="Training Loss")
    plt.plot(epochs, history["val_loss"], marker="s", linewidth=2, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross Entropy Loss")
    plt.title("Feedback Connection Training Loss")
    plt.xticks(epochs)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"feedback_loss_curve{tag}.png", dpi=300, bbox_inches="tight")
    plt.close()

    # LR plot (if available)
    if "lr" in history and len(history["lr"]) == actual_epochs:
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, history["lr"], marker="o", linewidth=2, label="Learning Rate")
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.title("Learning Rate Schedule")
        plt.xticks(epochs)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"learning_rate_curve{tag}.png", dpi=300, bbox_inches="tight")
        plt.close()


def _save_resume_checkpoint(
    out_dir,
    epoch_idx,
    model,
    optimizer,
    scheduler,
    history,
    best_val_acc,
    best_val_loss,
    early_best_loss,
    early_best_acc,
    patience_counter,
    best_model_state,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "epoch": epoch_idx,
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "history": history,
        "best_val_acc": best_val_acc,
        "best_val_loss": best_val_loss,
        "early_best_loss": early_best_loss,
        "early_best_acc": early_best_acc,
        "patience_counter": patience_counter,
        "best_model_state": best_model_state,
    }
    torch.save(ckpt, out_dir / "latest_resume_checkpoint.pth")


def _try_resume_checkpoint(out_dir, model, optimizer, scheduler, device):
    ckpt_path = out_dir / "latest_resume_checkpoint.pth"
    if not ckpt_path.exists():
        return None

    print(f"\nFound resume checkpoint at {ckpt_path}. Resuming...", flush=True)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler is not None and ckpt.get("scheduler_state") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    return ckpt


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    set_feedback_train_mode(model)
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (clips, labels) in enumerate(train_loader):
        if _STOP_REQUESTED:
            print(
                f"\n[Stop requested] Ending epoch early at batch {batch_idx}/{len(train_loader)}.",
                flush=True
            )
            break

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
        raise RuntimeError("Training loader produced 0 samples (empty loader or immediate stop).")

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


def _build_scheduler(optimizer, num_epochs, warmup_epochs=0):
    cosine = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, num_epochs - warmup_epochs)
    )
    if warmup_epochs <= 0 or num_epochs <= 1:
        return cosine

    warmup_epochs = min(warmup_epochs, num_epochs - 1)
    warmup = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_epochs
    )
    return optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs]
    )


def train_model(
    model,
    train_loader,
    val_loader,
    num_epochs,
    device,
    patience=4,
    min_delta_loss=1e-4,
    min_delta_acc=0.1,
    lr=3e-5,
    weight_decay=1e-2,
    label_smoothing=0.1,
    warmup_epochs=0,
    resume=True,
    show_plots=False,
):
    global _STOP_REQUESTED
    _STOP_REQUESTED = False

    # Register signal handlers for graceful timeout/termination flush
    prev_sigterm = signal.signal(signal.SIGTERM, _signal_stop_handler)
    prev_sigint = signal.signal(signal.SIGINT, _signal_stop_handler)

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
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = optim.AdamW(
        trainable_params,
        lr=lr,
        weight_decay=weight_decay
    )
    scheduler = _build_scheduler(optimizer, num_epochs, warmup_epochs=warmup_epochs)

    # ==========================================
    # 3. Training history & resume state
    # ==========================================
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    best_val_acc = -1.0
    best_val_loss = float("inf")
    early_best_loss = float("inf")
    early_best_acc = -1.0
    patience_counter = 0
    best_model_state = None

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
    }

    if resume:
        ckpt = _try_resume_checkpoint(OUTPUT_ROOT, model, optimizer, scheduler, device)
        if ckpt is not None:
            start_epoch = ckpt.get("epoch", -1) + 1
            history = ckpt.get("history", history)
            best_val_acc = ckpt.get("best_val_acc", best_val_acc)
            best_val_loss = ckpt.get("best_val_loss", best_val_loss)
            early_best_loss = ckpt.get("early_best_loss", early_best_loss)
            early_best_acc = ckpt.get("early_best_acc", early_best_acc)
            patience_counter = ckpt.get("patience_counter", patience_counter)
            best_model_state = ckpt.get("best_model_state", best_model_state)
            print(f"Resuming training from epoch {start_epoch + 1}/{num_epochs}", flush=True)

    # ==========================================
    # 4. Training loop (fault-tolerant)
    # ==========================================
    interrupted_reason = None

    try:
        for epoch in range(start_epoch, num_epochs):
            if _STOP_REQUESTED:
                interrupted_reason = "Stop/SIGTERM signal received before epoch start."
                break

            print(f"\n{'='*55}", flush=True)
            print(f"Epoch {epoch+1}/{num_epochs}", flush=True)
            print(f"{'='*55}", flush=True)

            current_lr = optimizer.param_groups[0]["lr"]

            train_loss, train_acc = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device
            )

            # If stopped mid-epoch with partial batches, save partial history/plots and exit cleanly
            if _STOP_REQUESTED:
                interrupted_reason = f"Stop/SIGTERM signal received mid-epoch {epoch+1}."
                _save_resume_checkpoint(
                    OUTPUT_ROOT,
                    max(0, epoch - 1),
                    model,
                    optimizer,
                    scheduler,
                    history,
                    best_val_acc,
                    best_val_loss,
                    early_best_loss,
                    early_best_acc,
                    patience_counter,
                    best_model_state,
                )
                _save_plots_and_history(history, OUTPUT_ROOT, suffix="partial")
                break

            val_loss, val_acc = validate_one_epoch(
                model,
                val_loader,
                criterion,
                device
            )

            scheduler.step()

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["lr"].append(current_lr)

            print(f"\nTrain Loss: {train_loss:.4f}", flush=True)
            print(f"Train Acc:  {train_acc:.2f}%", flush=True)
            print(f"Val Loss:   {val_loss:.4f}", flush=True)
            print(f"Val Acc:    {val_acc:.2f}%", flush=True)
            print(f"LR:         {current_lr:.7f}", flush=True)

            # Save best model
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

                best_model_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }

                torch.save(
                    best_model_state,
                    OUTPUT_ROOT / "best_feedback_model.pth"
                )

                feedback_state = {}
                for name, module in model.named_modules():
                    if name == "feedback_modules":
                        feedback_state["feedback_modules"] = module.state_dict()
                    elif isinstance(module, ConvRecurrentContainer):
                        if module.feedback_pwconv is not None:
                            feedback_state[f"{name}.feedback_pwconv"] = (
                                module.feedback_pwconv.state_dict()
                            )
                            if module.feedback_bn is not None:
                                feedback_state[f"{name}.feedback_bn"] = (
                                    module.feedback_bn.state_dict()
                                )

                torch.save(
                    feedback_state,
                    OUTPUT_ROOT / "best_feedback_connection.pth"
                )

                print("\n*** Best feedback model saved! ***", flush=True)

            # Early Stopping
            loss_improved = val_loss < early_best_loss - min_delta_loss
            acc_improved = val_acc > early_best_acc + min_delta_acc

            if loss_improved:
                early_best_loss = val_loss
            if acc_improved:
                early_best_acc = val_acc

            if loss_improved or acc_improved:
                patience_counter = 0
                print("Early stopping: Validation improved.", flush=True)
            else:
                patience_counter += 1
                print(
                    f"No significant improvement: "
                    f"{patience_counter}/{patience}",
                    flush=True
                )

            # Save resumable checkpoint + plots + logs after EVERY completed epoch
            _save_resume_checkpoint(
                OUTPUT_ROOT,
                epoch,
                model,
                optimizer,
                scheduler,
                history,
                best_val_acc,
                best_val_loss,
                early_best_loss,
                early_best_acc,
                patience_counter,
                best_model_state,
            )
            _save_plots_and_history(history, OUTPUT_ROOT)

            print("-" * 55, flush=True)

            if patience_counter >= patience:
                print("\nEarly stopping triggered!", flush=True)
                print(f"Training stopped at epoch {epoch+1}", flush=True)
                break

            if _STOP_REQUESTED:
                interrupted_reason = f"Stop/SIGTERM signal received after epoch {epoch+1}."
                break

    except (KeyboardInterrupt, Exception) as exc:
        interrupted_reason = f"{type(exc).__name__}: {exc}"
        print(f"\n[Interrupted] {interrupted_reason}", flush=True)
        print("Saving latest history and plots before exiting...", flush=True)
        _save_plots_and_history(history, OUTPUT_ROOT, suffix="partial")
        raise

    finally:
        # Always flush plots & history if at least 1 epoch was recorded, and restore signal handlers
        if len(history.get("train_acc", [])) > 0:
            _save_plots_and_history(history, OUTPUT_ROOT)
        signal.signal(signal.SIGTERM, prev_sigterm)
        signal.signal(signal.SIGINT, prev_sigint)

    # ==========================================
    # 5. Restore best model
    # ==========================================
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("\nBest model weights restored.", flush=True)

    # ==========================================
    # 6. Final plots & report
    # ==========================================
    _save_plots_and_history(history, OUTPUT_ROOT)

    if show_plots and len(history.get("train_acc", [])) > 0:
        actual_epochs = len(history["train_acc"])
        epochs = np.arange(1, actual_epochs + 1)

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, history["train_acc"], marker="o", linewidth=2, label="Training Accuracy")
        plt.plot(epochs, history["val_acc"], marker="s", linewidth=2, label="Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.title("Feedback Connection Training Accuracy")
        plt.xticks(epochs)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, history["train_loss"], marker="o", linewidth=2, label="Training Loss")
        plt.plot(epochs, history["val_loss"], marker="s", linewidth=2, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross Entropy Loss")
        plt.title("Feedback Connection Training Loss")
        plt.xticks(epochs)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.close()

    actual_epochs = len(history["train_acc"])
    print("\nTraining finished.", flush=True)
    if interrupted_reason:
        print(f"Note: Ended early due to -> {interrupted_reason}", flush=True)
    print(f"Epochs executed: {actual_epochs}/{num_epochs}", flush=True)
    print(f"Best validation accuracy: {best_val_acc:.2f}%", flush=True)
    print(f"Best validation loss: {best_val_loss:.4f}", flush=True)
    print(f"Results saved in: {OUTPUT_ROOT}", flush=True)

    return history