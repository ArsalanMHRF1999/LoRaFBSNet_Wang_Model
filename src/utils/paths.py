
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

    DATA_ROOT = Path(
        "/kaggle/input/datasets/matthewjansen/ucf101-action-recognition"
    )

    ANNOTATION_ROOT = DATA_ROOT

    WEIGHTS_ROOT = Path(
        "/kaggle/input/models/mohammadmehranfar/lora-weights/"
        "pytorch/default/1/lorafb_snet18.pth"
    )

    OUTPUT_ROOT = Path("/kaggle/working/feedback_weights")

else:

    DATA_STORAGE = Path(r"D:\Arsalan_project")

    # Extracted UCF101 dataset
    DATA_ROOT = DATA_STORAGE / "datasets" / "UCF101"

    # CSV annotations are already inside DATA_ROOT
    ANNOTATION_ROOT = DATA_ROOT

    # LoRaFB-SNet18 weights
    WEIGHTS_ROOT = (
        DATA_STORAGE
        / "pretrained_weights"
        / "ucf101"
        / "lorafb_snet18.pth"
    )

    # New training outputs
    OUTPUT_ROOT = (
        DATA_STORAGE
        / "outputs"
        / "feedback_weights"
    )

# =====================================
# Create output directories
# =====================================

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# =====================================
# Configuration validation
# =====================================

print("=" * 50)
print(f"Running on Kaggle: {IS_KAGGLE}")
print(f"Project root: {PROJECT_ROOT}")
print(f"Dataset root: {DATA_ROOT}")
print(f"Annotation root: {ANNOTATION_ROOT}")
print(f"Weights path: {WEIGHTS_ROOT}")
print(f"Output root: {OUTPUT_ROOT}")

print("=" * 50)

print("Dataset exists:", DATA_ROOT.exists())
print("Train CSV exists:", (DATA_ROOT / "train.csv").exists())
print("Validation CSV exists:", (DATA_ROOT / "val.csv").exists())
print("Test CSV exists:", (DATA_ROOT / "test.csv").exists())
print("Model weights exist:", WEIGHTS_ROOT.is_file())
print("Output directory exists:", OUTPUT_ROOT.exists())
