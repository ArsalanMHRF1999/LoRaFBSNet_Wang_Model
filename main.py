# from src.dataset.run_dataset import dataset_report
# from src.evaluation.evaluate_pretrained import evaluate_model
# from src.utils.paths import WEIGHTS_ROOT
# from torch import nn
# import torch
# from src.dataset.dataloaders import create_dataloaders
# from src.train_feedback.load_weights import create_model_with_feedback, create_model
# from src.train_feedback.train import train_model

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# NUM_CLASSES = 101
# model_43 = create_model(WEIGHTS_ROOT, num_classes=NUM_CLASSES, device=DEVICE)
# model = create_model_with_feedback(WEIGHTS_ROOT, feedback_connection=[(4, 3)], num_classes=NUM_CLASSES, device=DEVICE)
# dataset_report()
# test_dataset, test_video_loader = create_dataloaders("test")
# criterion = nn.CrossEntropyLoss()
# accuracy = evaluate_model(model, test_video_loader, criterion, DEVICE, NUM_CLASSES)

# train_dataset, train_video_loader = create_dataloaders("train")
# val_dataset, val_video_loader = create_dataloaders("val")
# history = train_model(
#     model, 
#     train_video_loader, 
#     val_video_loader, 
#     num_epochs=1, 
#     device=DEVICE
# )


# import torch
# import torch.nn as nn
# import numpy as np
# import matplotlib.pyplot as plt
# from pathlib import Path
# from src.dataset.dataloaders import create_dataloaders
# from src.train_feedback.load_weights import (
#     create_model_with_feedback,
#     create_model
# )

# from src.evaluation.evaluate_pretrained import evaluate_model
# from src.utils.paths import WEIGHTS_ROOT

# from src.utils.paths import FEEDBACK_CONFIGS, COMPARISON_RESULT_ROOT, WEIGHTS_ROOT, FEEDBACK_CHECKPOINTS

# # ==========================================
# # 1. Configuration
# # ==========================================

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# NUM_CLASSES = 101


# # ==========================================
# # 2. Load complete trained feedback model
# # ==========================================

# def load_trained_feedback_model(
#     pretrained_weights,
#     feedback_checkpoint,
#     feedback_connection
# ):

#     model = create_model_with_feedback(
#         weights_path=pretrained_weights,
#         feedback_connection=feedback_connection,
#         num_classes=NUM_CLASSES,
#         device=DEVICE
#     )

#     checkpoint = torch.load(
#         feedback_checkpoint,
#         map_location="cpu",
#         weights_only=True
#     )

#     # Load the complete trained model
#     model.load_state_dict(checkpoint, strict=True)

#     model = model.to(DEVICE)
#     model.eval()

#     print(
#         f"Loaded trained model: {feedback_checkpoint}",
#         flush=True
#     )

#     return model


# # ==========================================
# # 3. Load test dataset
# # ==========================================

# print("\nLoading test dataset...")

# test_dataset, test_loader = create_dataloaders("test")

# criterion = nn.CrossEntropyLoss()

# print("Test samples:", len(test_dataset))


# # ==========================================
# # 4. Define all models
# # ==========================================

# models = {}

# # ------------------------------------------
# # Baseline model (No feedback)
# # ------------------------------------------

# print("\nLoading baseline model...")

# baseline_model = create_model(
#     weights_path=WEIGHTS_ROOT,
#     num_classes=NUM_CLASSES,
#     device=DEVICE
# )

# baseline_model.eval()

# models["Baseline"] = baseline_model


# # ------------------------------------------
# # Feedback models
# # ------------------------------------------

# for model_name, connection in FEEDBACK_CONFIGS.items():

#     print(f"\nLoading {model_name}...")

#     model = load_trained_feedback_model(
#         pretrained_weights=WEIGHTS_ROOT,
#         feedback_checkpoint=FEEDBACK_CHECKPOINTS[model_name],
#         feedback_connection=connection
#     )

#     models[model_name] = model


# # ==========================================
# # 5. Evaluate all models
# # ==========================================

# results = {}

# for model_name, model in models.items():

#     print("\n" + "=" * 60)
#     print(f"Evaluating: {model_name}")
#     print("=" * 60)

#     # Separate directory for each model
#     model_plot_dir = COMPARISON_RESULT_ROOT / model_name.replace(" ", "_")

#     metrics = evaluate_model(
#         model=model,
#         test_loader=test_loader,
#         criterion=criterion,
#         device=DEVICE,
#         num_classes=NUM_CLASSES,
#         class_names=None,
#         plot_dir=model_plot_dir,
#         return_details=True
#     )

#     results[model_name] = metrics

#     print(f"\n{model_name} Results:")
#     print(f"Top-1 Accuracy: {metrics['top1_accuracy']:.2f}%")
#     print(f"Top-5 Accuracy: {metrics['top5_accuracy']:.2f}%")
#     print(f"Test Loss:      {metrics['test_loss']:.4f}")


# # ==========================================
# # 6. Print comparison table
# # ==========================================

# print("\n\n========== MODEL COMPARISON ==========")

# print(
#     f"{'Model':<20}"
#     f"{'Top-1 Acc':<15}"
#     f"{'Top-5 Acc':<15}"
#     f"{'Test Loss':<15}"
# )

# print("-" * 65)

# for name, metrics in results.items():

#     print(
#         f"{name:<20}"
#         f"{metrics['top1_accuracy']:<15.2f}"
#         f"{metrics['top5_accuracy']:<15.2f}"
#         f"{metrics['test_loss']:<15.4f}"
#     )


# # ==========================================
# # 7. Comparative Accuracy Plot
# # ==========================================

# model_names = list(results.keys())

# top1_values = [
#     results[name]["top1_accuracy"]
#     for name in model_names
# ]

# top5_values = [
#     results[name]["top5_accuracy"]
#     for name in model_names
# ]

# x = np.arange(len(model_names))
# width = 0.35

# fig, ax = plt.subplots(figsize=(11, 6))

# bars1 = ax.bar(
#     x - width / 2,
#     top1_values,
#     width,
#     label="Top-1 Accuracy"
# )

# bars2 = ax.bar(
#     x + width / 2,
#     top5_values,
#     width,
#     label="Top-5 Accuracy"
# )

# ax.set_ylabel("Accuracy (%)")
# ax.set_title("Performance Comparison: Feedback Architectures")
# ax.set_xticks(x)
# ax.set_xticklabels(model_names)
# ax.set_ylim(0, 105)

# ax.legend()
# ax.grid(axis="y", alpha=0.25)

# for bars in [bars1, bars2]:
#     for bar in bars:
#         ax.text(
#             bar.get_x() + bar.get_width() / 2,
#             bar.get_height() + 1,
#             f"{bar.get_height():.2f}",
#             ha="center",
#             fontsize=9
#         )

# plt.tight_layout()

# plt.savefig(
#     COMPARISON_RESULT_ROOT / "accuracy_comparison.png",
#     dpi=300,
#     bbox_inches="tight"
# )

# plt.show()
# plt.close()


# # ==========================================
# # 8. Comparative Loss Plot
# # ==========================================

# loss_values = [
#     results[name]["test_loss"]
#     for name in model_names
# ]

# fig, ax = plt.subplots(figsize=(10, 6))

# bars = ax.bar(model_names, loss_values)

# ax.set_ylabel("Cross Entropy Loss")
# ax.set_title("Test Loss Comparison")
# ax.grid(axis="y", alpha=0.25)

# for bar, value in zip(bars, loss_values):
#     ax.text(
#         bar.get_x() + bar.get_width() / 2,
#         bar.get_height(),
#         f"{value:.4f}",
#         ha="center",
#         va="bottom"
#     )

# plt.tight_layout()

# plt.savefig(
#     COMPARISON_RESULT_ROOT / "loss_comparison.png",
#     dpi=300,
#     bbox_inches="tight"
# )

# plt.show()
# plt.close()


# # ==========================================
# # 9. Save numerical comparison
# # ==========================================

# np.savez(
#     COMPARISON_RESULT_ROOT / "model_comparison.npz",

#     model_names=np.array(model_names),

#     top1_accuracy=np.array(top1_values),

#     top5_accuracy=np.array(top5_values),

#     test_loss=np.array(loss_values)
# )


# print("\n====================================")
# print("All evaluations completed!")
# print(f"Results saved in: {COMPARISON_RESULT_ROOT.resolve()}")
# print("====================================")

from pathlib import Path

roots = [
    Path("/kaggle/input"),
    Path("/kaggle/working"),
]

for root in roots:
    print(f"\n{'='*70}")
    print(f"Searching in: {root}")

    for p in root.rglob("*.pth"):
        print(p)

    for p in root.rglob("*.pt"):
        print(p)

