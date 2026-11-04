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


# Create output directory
ANNOTATION_ROOT.mkdir(parents=True, exist_ok=True)

