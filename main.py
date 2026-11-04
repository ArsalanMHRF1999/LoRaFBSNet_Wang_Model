from src.dataset.run_dataset import dataset_report
from src.evaluation.evaluate_pretrained import evaluate_model
from src.utils.paths import WEIGHTS_ROOT
from torch import nn
from src.dataset.dataloaders import create_dataloaders

dataset_report()
test_dataset, test_video_loader = create_dataloaders("test")
criterion = nn.CrossEntropyLoss()
accuracy = evaluate_model(test_video_loader, criterion, WEIGHTS_ROOT, num_classes=101)