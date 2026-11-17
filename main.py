
from multiprocessing import freeze_support

from src.dataset.run_dataset import dataset_report
from src.evaluation.evaluate_pretrained import evaluate_model
from src.utils.paths import WEIGHTS_ROOT
from torch import nn
import torch

from src.dataset.dataloaders import create_dataloaders
from src.train_feedback.load_weights import (
    create_model_with_feedback,
    create_model
)
from src.train_feedback.train import train_model


def main():

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_CLASSES = 101

    print("=" * 50)
    print("Device:", DEVICE)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    print("=" * 50)

    # Create model with pretrained weights and feedback
    model = create_model_with_feedback(
        WEIGHTS_ROOT,
        feedback_connection=[(4, 3)],
        num_classes=NUM_CLASSES,
        device=DEVICE
    )

    # Load datasets
    train_dataset, train_video_loader = create_dataloaders("train")
    val_dataset, val_video_loader = create_dataloaders("val")

    print("Train videos:", len(train_dataset))
    print("Validation videos:", len(val_dataset))

    # Fine-tuning: one epoch for the first test
    history = train_model(
        model,
        train_video_loader,
        val_video_loader,
        num_epochs=1,
        device=DEVICE
    )

    print("Fine-tuning completed!")
    print("History:", history)


if __name__ == "__main__":
    freeze_support()
    main()
