
import torch
import numpy as np
import matplotlib.pyplot as plt
from spikingjelly.activation_based import functional
from pathlib import Path

def evaluate_model(
    model,
    test_loader,
    criterion,
    device,
    num_classes,
    class_names=None,
    plot_dir="evaluation_results"
):
    model.eval()

    print("Model loaded successfully!", flush=True)

    # ==========================================
    # 1. Evaluation variables
    # ==========================================

    total_loss = 0.0
    correct = 0
    total = 0
    top5_correct = 0

    all_predictions = []
    all_labels = []
    all_confidences = []
    all_correctness = []

    batch_losses = []

    # ==========================================
    # 2. Evaluation loop
    # ==========================================

    print("\nStarting evaluation...\n", flush=True)

    with torch.inference_mode():
        for batch_idx, (clips, labels) in enumerate(test_loader):
            functional.reset_net(model)
            clips = clips.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # B,T,C,H,W -> T,B,C,H,W
            clips = clips.permute(1, 0, 2, 3, 4).contiguous()

            # Forward
            outputs = model(clips)

            # Temporal average
            outputs = outputs.mean(0)

            loss = criterion(outputs, labels)

            # Predictions
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = probabilities.max(dim=1)
            batch_size = labels.size(0)

            # Top-1
            batch_correct = predicted.eq(labels)
            correct += batch_correct.sum().item()
            total += batch_size

            # Top-5
            top5_predictions = probabilities.topk(
                min(5, num_classes), dim=1
            ).indices
            top5_correct += top5_predictions.eq(
                labels.unsqueeze(1)
            ).any(dim=1).sum().item()

            # Loss
            total_loss += loss.item() * batch_size
            batch_losses.append(loss.item())

            # Store results
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            all_confidences.extend(confidence.cpu().numpy())
            all_correctness.extend(batch_correct.cpu().numpy())

            if batch_idx % 10 == 0:
                current_acc = 100.0 * correct / total
                print(
                    f"Batch [{batch_idx+1}/{len(test_loader)}] | "
                    f"Loss: {loss.item():.4f} | "
                    f"Running Acc: {current_acc:.2f}%",
                    flush=True
                )

    # ==========================================
    # 3. Calculate final metrics
    # ==========================================

    if total == 0:
        raise ValueError("The test_loader returned no samples.")

    test_loss = total_loss / total
    accuracy = 100.0 * correct / total
    top5_accuracy = 100.0 * top5_correct / total
    predictions = np.asarray(all_predictions)
    labels = np.asarray(all_labels)
    confidences = np.asarray(all_confidences)
    correctness = np.asarray(all_correctness, dtype=bool)

    print("\n========== FINAL RESULTS ==========")
    print(f"Test Loss:     {test_loss:.4f}")
    print(f"Top-1 Acc:     {accuracy:.2f}%")
    print(f"Top-5 Acc:     {top5_accuracy:.2f}%")

    print("===================================\n", flush=True)

    # ==========================================
    # 6. Prepare plot directory
    # ==========================================

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]

    if len(class_names) != num_classes:
        raise ValueError("class_names must contain num_classes entries.")

    # ==========================================
    # 4. Confusion Matrix
    # ==========================================

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (labels, predictions), 1)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_normalized = np.divide(
        cm,
        row_sums,
        out=np.zeros_like(cm, dtype=float),
        where=row_sums != 0
    )

    fig, ax = plt.subplots(figsize=(16, 14))

    im = ax.imshow(
        cm_normalized,
        interpolation="nearest",
        cmap="Blues",
        vmin=0,
        vmax=1
    )

    ax.set_title(
        "LoRaFB-SNet - Normalized Confusion Matrix",
        fontsize=17,
        fontweight="bold",
        pad=18
    )

    ax.set_xlabel("Predicted Class", fontsize=12)
    ax.set_ylabel("True Class", fontsize=12)
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(class_names, rotation=90, fontsize=5)
    ax.set_yticklabels(class_names, fontsize=5)

    fig.colorbar(im, ax=ax, label="Normalized Frequency")
    plt.tight_layout()
    plt.savefig(
        plot_dir / "01_confusion_matrix.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # ==========================================
    # 5. Per-Class Accuracy
    # ==========================================

    class_accuracy = np.divide(
        np.diag(cm),
        row_sums.flatten(),
        out=np.full(num_classes, np.nan),
        where=row_sums.flatten() != 0
    ) * 100

    valid_classes = np.where(~np.isnan(class_accuracy))[0]

    sorted_classes = valid_classes[
        np.argsort(class_accuracy[valid_classes])
    ]

    # Select 20 weakest and 20 strongest classes
    selected = np.concatenate([
        sorted_classes[:20],
        sorted_classes[-20:]
    ])

    fig, ax = plt.subplots(figsize=(13, 12))

    values = class_accuracy[selected]
    names = [class_names[i] for i in selected]

    bars = ax.barh(names, values)

    ax.set_title(
        "Per-Class Recognition Accuracy",
        fontsize=16,
        fontweight="bold"
    )

    ax.set_xlabel("Accuracy (%)")
    ax.set_xlim(0, 105)

    ax.grid(axis="x", alpha=0.25)

    for bar, value in zip(bars, values):
        ax.text(
            value + 1,
            bar.get_y() + bar.get_height()/2,
            f"{value:.1f}%",
            va="center",
            fontsize=8
        )

    plt.tight_layout()
    plt.savefig(
        plot_dir / "02_per_class_accuracy.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # ==========================================
    # 9. Confidence Distribution
    # ==========================================

    correct_conf = confidences[correctness]
    incorrect_conf = confidences[~correctness]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(
        correct_conf,
        bins=30,
        alpha=0.65,
        density=True,
        label="Correct Predictions"
    )

    ax.hist(
        incorrect_conf,
        bins=30,
        alpha=0.65,
        density=True,
        label="Incorrect Predictions"
    )

    ax.set_title(
        "Prediction Confidence Distribution",
        fontsize=16,
        fontweight="bold"
    )

    ax.set_xlabel("Maximum Softmax Probability")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(alpha=0.2)

    plt.tight_layout()

    plt.savefig(
        plot_dir / "03_confidence_distribution.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # ==========================================
    # 10. Batch Loss Curve
    # ==========================================

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        np.arange(1, len(batch_losses) + 1),
        batch_losses,
        linewidth=1.2,
        label="Batch Loss"
    )

    ax.set_title(
        "Evaluation Loss Across Test Batches",
        fontsize=16,
        fontweight="bold"
    )

    ax.set_xlabel("Batch")
    ax.set_ylabel("Cross Entropy Loss")
    ax.legend()
    ax.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(
        plot_dir / "04_batch_loss.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # ==========================================
    # 11. Summary Figure
    # ==========================================

    fig, ax = plt.subplots(figsize=(9, 5))
    metric_names = ["Top-1 Accuracy", "Top-5 Accuracy"]
    metric_values = [accuracy, top5_accuracy]
    bars = ax.bar(metric_names, metric_values)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(
        "LoRaFB-SNet Performance Summary",
        fontsize=16,
        fontweight="bold"
    )

    for bar, value in zip(bars, metric_values):
        ax.text(
            bar.get_x() + bar.get_width()/2,
            value + 2,
            f"{value:.2f}%",
            ha="center",
            fontweight="bold"
        )

    ax.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(
        plot_dir / "05_performance_summary.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # ==========================================
    # 12. Save numerical results
    # ==========================================

    np.savez(
        plot_dir / "evaluation_metrics.npz",
        predictions=predictions,
        labels=labels,
        confidences=confidences,
        correctness=correctness,
        confusion_matrix=cm,
        class_accuracy=class_accuracy,
        batch_losses=np.asarray(batch_losses),
        top1_accuracy=accuracy,
        top5_accuracy=top5_accuracy,
        test_loss=test_loss
    )

    print(f"All plots saved to: {plot_dir.resolve()}", flush=True)

    return accuracy, test_loss
