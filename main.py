from src.dataset.run_dataset import dataset_report
from src.evaluation.evaluate_pretrained import evaluate_model
from src.utils.paths import WEIGHTS_ROOT
from torch import nn
import torch
from src.dataset.dataloaders import create_dataloaders
from src.train_feedback.load_weights import create_model_with_feedback, create_model
from src.train_feedback.train import train_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 101
# model_43 = create_model(WEIGHTS_ROOT, num_classes=NUM_CLASSES, device=DEVICE)
model = create_model_with_feedback(WEIGHTS_ROOT, feedback_connection=[(4, 3)], num_classes=NUM_CLASSES, device=DEVICE)
# dataset_report()
# test_dataset, test_video_loader = create_dataloaders("test")
# criterion = nn.CrossEntropyLoss()
# accuracy = evaluate_model(model, test_video_loader, criterion, DEVICE, NUM_CLASSES)

train_dataset, train_video_loader = create_dataloaders("train")
val_dataset, val_video_loader = create_dataloaders("val")
history = train_model(
    model, 
    train_video_loader, 
    val_video_loader, 
    num_epochs=10, 
    device=DEVICE
)
