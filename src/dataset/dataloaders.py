from src.utils.dataset_configs import FRAMES_PER_CLIP, STEP_BETWEEN_CLIPS, BATCH_SIZE, NUM_WORKERS       
from src.dataset.transformer import get_baseline_transforms, UCF101WithVideoID, safe_video_collate
from src.utils.paths import DATA_ROOT, ANNOTATION_ROOT
from src.dataset.annotation import create_annotations
from torch.utils.data import DataLoader


def create_dataloaders(split):
    create_annotations(split)

    common_loader_args = {
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "pin_memory": True,
        "collate_fn": safe_video_collate,
        "persistent_workers": NUM_WORKERS > 0,
    }

    if NUM_WORKERS > 0:
        common_loader_args["prefetch_factor"] = 2

    if split == "train":
        dataset = UCF101WithVideoID(
            root=str(DATA_ROOT / "train"),
            annotation_path=str(ANNOTATION_ROOT / "train"),
            frames_per_clip=FRAMES_PER_CLIP,
            step_between_clips=STEP_BETWEEN_CLIPS,
            fold=1,
            train=True,
            transform=get_baseline_transforms(train=True)
        )

        loader = DataLoader(
            dataset,
            shuffle=True,
            **common_loader_args
        )

    elif split == "val":
        dataset = UCF101WithVideoID(
            root=str(DATA_ROOT / "val"),
            annotation_path=str(ANNOTATION_ROOT / "val"),
            frames_per_clip=FRAMES_PER_CLIP,
            step_between_clips=STEP_BETWEEN_CLIPS,
            fold=1,
            train=False,
            transform=get_baseline_transforms(train=False)
        )

        loader = DataLoader(
            dataset,
            shuffle=False,
            **common_loader_args
        )

    elif split == "test":
        dataset = UCF101WithVideoID(
            root=str(DATA_ROOT / "test"),
            annotation_path=str(ANNOTATION_ROOT / "test"),
            frames_per_clip=FRAMES_PER_CLIP,
            step_between_clips=STEP_BETWEEN_CLIPS,
            fold=1,
            train=False,
            transform=get_baseline_transforms(train=False)
        )

        loader = DataLoader(
            dataset,
            shuffle=False,
            **common_loader_args
        )

    else:
        raise ValueError(f"Unknown split: {split}")

    return dataset, loader

