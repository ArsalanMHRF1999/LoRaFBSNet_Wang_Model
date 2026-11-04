from pathlib import Path
import os

print("\n" + "=" * 70)
print("KAGGLE ENVIRONMENT DIAGNOSTICS")
print("=" * 70)

print("\nCurrent working directory:")
print(os.getcwd())

print("\nKaggle input directory:")

input_dir = Path("/kaggle/input")

print("Exists:", input_dir.exists())

if input_dir.exists():

    print("Contents:", list(input_dir.iterdir()))

    for item in sorted(input_dir.iterdir()):

        print(f"\nDataset: {item.name}")

        if item.is_dir():

            for root, dirs, files in os.walk(item):

                print("Directory:", root)
                print("Subdirectories:", dirs[:10])
                print("Files:", files[:5])

                dirs[:] = dirs[:3]

print("\nKaggle working directory:")
print(list(Path("/kaggle/working").iterdir()))

for item in sorted(input_dir.iterdir()):
    print(f"\nDataset: {item}")

    if item.is_dir():
        for sub in sorted(item.iterdir())[:20]:
            print("   ", sub.name)

print("=" * 60)

print("\n" + "=" * 70)
print("DIAGNOSTICS COMPLETED")
print("=" * 70)

# from src.dataset.dataloaders import (
#     train_dataset,
#     train_video_loader,
#     test_dataset,
#     test_video_loader,
#     val_dataset,
#     val_video_loader
# )