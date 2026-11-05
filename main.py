from src.dataset.run_dataset import dataset_report
from src.evaluation.evaluate_pretrained import evaluate_model
from src.utils.paths import WEIGHTS_ROOT
from torch import nn
from src.dataset.dataloaders import create_dataloaders
from src.train_feedback.load_weights import create_model_with_feedback

model = create_model_with_feedback(WEIGHTS_ROOT, num_classes=101)

# dataset_report()
# test_dataset, test_video_loader = create_dataloaders("test")
# criterion = nn.CrossEntropyLoss()
# accuracy = evaluate_model(test_video_loader, criterion, WEIGHTS_ROOT, num_classes=101)