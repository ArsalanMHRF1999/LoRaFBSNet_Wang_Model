
from pathlib import Path
import os

FEEDBACK_CONFIGS = {
    "Feedback 4-3": [(4, 3)],
    "Feedback 3-2": [(3, 2)],
    "Feedback 3-1": [(3, 1)],
}

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

    ANNOTATION_ROOT = Path("kaggle/working/annotations")

    COMPARISON_RESULT_ROOT = Path(
        "/kaggle/working/compare_results"
    )

    WEIGHTS_ROOT = Path(
        "/kaggle/input/models/mohammadmehranfar/lora-weights/"
        "pytorch/default/1/lorafb_snet18.pth"
    )

    FEEDBACK_CHECKPOINTS = {
        "Feedback 4-3": Path("/kaggle/input/models/mohammadmehranfar/4_3_feedback/pytorch/default/1/best_feedback_model.pth"),
        "Feedback 3-2": Path("/kaggle/input/models/mohammadmehranfar/3_2_feedback/pytorch/default/1/best_feedback_model.pth"),
        "Feedback 3-1": Path("/kaggle/input/models/mohammadmehranfar/3_1_feedback_weight/pytorch/default/1/best_feedback_model.pth"),
    }

    OUTPUT_ROOT = Path(
        "/kaggle/working/feedback_weights"
    )


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

    FEEDBACK_CHECKPOINTS = {
        "Feedback 4-3": Path(".../.../.../feedback_weights/weights/feedback_4_3/best_feedback_model.pth"),
        "Feedback 3-2": Path(".../.../.../feedback_weights/weights/feedback_3_2/best_feedback_model.pth"),
        "Feedback 3-1": Path(".../.../.../feedback_weights/weights/feedback_3_1/best_feedback_model.pth"),
    }

    OUTPUT_ROOT = (
        DATA_STORAGE
        / "outputs"
        / "feedback_weights"
    )

# =====================================
# Create output directories
# =====================================

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

COMPARISON_RESULT_ROOT = Path("comparison_results")
COMPARISON_RESULT_ROOT.mkdir(parents=True, exist_ok=True)

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
print(f"Compare_Output root: {COMPARISON_RESULT_ROOT}")
