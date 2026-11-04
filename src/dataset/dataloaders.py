from src.utils.dataset_configs import FRAMES_PER_CLIP, STEP_BETWEEN_CLIPS, BATCH_SIZE, NUM_WORKERS       
from src.dataset.transformer import get_baseline_transforms, UCF101WithVideoID
from src.utils.paths import DATA_ROOT, ANNOTATION_ROOT
from torch.utils.data import DataLoader

train_dataset = UCF101WithVideoID(
    root=str(DATA_ROOT / "train"),
    annotation_path=str(ANNOTATION_ROOT / "train"),
    frames_per_clip=FRAMES_PER_CLIP,
    step_between_clips=STEP_BETWEEN_CLIPS,
    fold=1,
    train=True,
    transform=get_baseline_transforms(train=True)
)

val_dataset = UCF101WithVideoID(
    root=str(DATA_ROOT / "val"),
    annotation_path=str(ANNOTATION_ROOT / "val"),
    frames_per_clip=FRAMES_PER_CLIP,
    step_between_clips=STEP_BETWEEN_CLIPS,
    fold=1,
    train=False,
    transform=get_baseline_transforms(train=False)
)

test_dataset = UCF101WithVideoID(
    root=str(DATA_ROOT / "test"),
    annotation_path=str(ANNOTATION_ROOT / "test"),
    frames_per_clip=FRAMES_PER_CLIP,
    step_between_clips=STEP_BETWEEN_CLIPS,
    fold=1,
    train=False,
    transform=get_baseline_transforms(train=False)
)

train_video_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

val_video_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

test_video_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)