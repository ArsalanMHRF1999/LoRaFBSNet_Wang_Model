
import os
from pathlib import Path

# =====================================
# Project configuration
# =====================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

IS_KAGGLE = Path("/kaggle/input").exists()

# =====================================
# Dataset paths
# =====================================

if IS_KAGGLE:
    DEFAULT_DATA_ROOT = Path(
        "/kaggle/input/datasets/matthewjansen/ucf101-action-recognition"
    )

    DEFAULT_ANNOTATION_ROOT = Path(
        "/kaggle/working/ucf101_annotations"
    )

else:
    DEFAULT_DATA_ROOT = Path.home() / "Datasets" / "UCF101"

    DEFAULT_ANNOTATION_ROOT = (
        PROJECT_ROOT / "data" / "annotations"
    )


DATA_ROOT = Path(
    os.getenv("UCF101_DATA_ROOT", str(DEFAULT_DATA_ROOT))
)

ANNOTATION_ROOT = Path(
    os.getenv(
        "UCF101_ANNOTATION_ROOT",
        str(DEFAULT_ANNOTATION_ROOT)
    )
)

# Create output directory
ANNOTATION_ROOT.mkdir(parents=True, exist_ok=True)


# =====================================
# Diagnostic
# =====================================

if __name__ == "__main__":

    print("=" * 50)
    print("PROJECT CONFIGURATION")
    print("=" * 50)

    print(f"Project root:      {PROJECT_ROOT}")
    print(f"Kaggle:            {IS_KAGGLE}")
    print(f"Dataset root:      {DATA_ROOT}")
    print(f"Annotation root:   {ANNOTATION_ROOT}")

    print("=" * 50)

    print("Dataset exists:", DATA_ROOT.exists())
