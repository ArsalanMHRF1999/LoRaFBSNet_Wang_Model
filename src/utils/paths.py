from pathlib import Path
import os

# =====================================
# Project configuration
# =====================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# =====================================
# Environment detection
# =====================================

IS_KAGGLE = (
    os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None
    or Path("/kaggle/input").exists()
)

# =====================================
# Storage configuration
# =====================================

if IS_KAGGLE:

    # Kaggle paths
    DATA_ROOT = Path(
        "/kaggle/input/datasets/matthewjansen/ucf101-action-recognition"
    )

    ANNOTATION_ROOT = Path(
        "/kaggle/working/ucf101_annotations"
    )

    WEIGHTS_ROOT = Path(
        "/kaggle/input/models/mohammadmehranfar/lora-weights/"
        "pytorch/default/1/lorafb_snet18.pth"
    )

    OUTPUT_ROOT = Path(
        "/kaggle/working/feedback_weights"
    )

else:

    # Local Windows paths
    DATA_STORAGE = Path(r"D:\SAEED\DEEP_PROJECT")

    DATA_ROOT = (
        DATA_STORAGE / "datasets" / "UCF101"
    )

    ANNOTATION_ROOT = (
        DATA_STORAGE / "annotations" / "UCF101"
    )

    WEIGHTS_ROOT = (
        DATA_STORAGE
        / "pretrained_weights"
        / "lorafb_snet18.pth"
    )

    OUTPUT_ROOT = (
        DATA_STORAGE / "outputs" / "feedback_weights"
    )

# =====================================
# Create output directories
# =====================================

ANNOTATION_ROOT.mkdir(parents=True, exist_ok=True)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# =====================================
# Configuration validation
# =====================================

print(f"Running on Kaggle: {IS_KAGGLE}")
print(f"Project root: {PROJECT_ROOT}")
print(f"Dataset root: {DATA_ROOT}")
print(f"Annotation root: {ANNOTATION_ROOT}")
print(f"Weights path: {WEIGHTS_ROOT}")
print(f"Output root: {OUTPUT_ROOT}")
