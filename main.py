from pathlib import Path

print("=" * 60)
print("KAGGLE DATASET DIAGNOSTICS")
print("=" * 60)

input_dir = Path("/kaggle/input")

for item in sorted(input_dir.iterdir()):
    print(f"\nDataset: {item}")

    if item.is_dir():
        for sub in sorted(item.iterdir())[:20]:
            print("   ", sub.name)

print("=" * 60)

# from src.dataset.dataloaders import (
#     train_dataset,
#     train_video_loader,
#     test_dataset,
#     test_video_loader,
#     val_dataset,
#     val_video_loader
# )