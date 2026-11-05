from pathlib import Path

# =====================================
# Project configuration
# =====================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# =====================================
# Dataset paths
# =====================================
DATA_ROOT = Path("/kaggle/input/datasets/matthewjansen/ucf101-action-recognition")
ANNOTATION_ROOT = Path("/kaggle/working/ucf101_annotations")
WEIGHTS_ROOT = Path("/kaggle/input/models/mohammadmehranfar/lora-weights/pytorch/default/1/lorafb_snet18.pth")
OUTPUT_ROOT = Path("/kaggle/working/feedback_weights")

# Create output directory
ANNOTATION_ROOT.mkdir(parents=True, exist_ok=True)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

