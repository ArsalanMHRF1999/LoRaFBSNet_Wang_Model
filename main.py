from src.dataset.dataloaders import train_dataset, train_video_loader, test_dataset, test_video_loader, val_dataset, val_video_loader
from src.utils.paths import DATA_ROOT
from src.utils.visualize_dataset import show_ucf_clip_sample_with_class
import torch

print("Number of videos:")
print("Train:", len(train_dataset.samples))
print("Validation:", len(val_dataset.samples))
print("Test:", len(test_dataset.samples))

print("\nNumber of generated clips:")
print("Train:", len(train_dataset))
print("Validation:", len(val_dataset))
print("Test:", len(test_dataset))

clips, labels = next(iter(test_video_loader))

print("\nBatch verification:")
print("Clips shape:", clips.shape)
print("Labels shape:", labels.shape)
print("Data type:", clips.dtype)
print("Min:", clips.min().item())
print("Max:", clips.max().item())

assert clips.shape[1:] == (16, 3, 224, 224)
assert clips.dtype == torch.float32

classes = sorted([
    p.name
    for p in (DATA_ROOT / "train").iterdir()
    if p.is_dir()
])

idx_to_class = {
    idx: class_name
    for idx, class_name in enumerate(classes)
}

print("Number of classes:", len(idx_to_class))
print("First 10 classes:")

for idx in range(10):
    print(idx, ":", idx_to_class[idx])

# Test visualization on any sample index
show_ucf_clip_sample_with_class(
    train_video_loader.dataset,
    idx_to_class,
    sample_idx=10,
    n_show=8,
    title_prefix="Full UCF101 clip"
)