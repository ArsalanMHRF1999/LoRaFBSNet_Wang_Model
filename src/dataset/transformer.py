import torchvision
import torchvision.transforms as transforms
import torch
from torch.utils.data._utils.collate import default_collate
from src.utils.dataset_configs import baseline_configs


class UCF101WithVideoID(torchvision.datasets.UCF101):

    def __getitem__(self, idx):

        # Decode video clip
        try:
            video, audio, info, video_idx = self.video_clips.get_clip(idx)

        except Exception as e:
            print(
                f"[VIDEO DECODE ERROR] "
                f"Clip index: {idx} | "
                f"Error: {type(e).__name__}: {e}"
            )
            return None

        # Retrieve the correct label
        label = self.samples[self.indices[video_idx]][1]

        # Apply transformations
        if self.transform is not None:
            video = self.transform(video)

        return video, label


def safe_video_collate(batch):

    # Remove failed video clips
    batch = [item for item in batch if item is not None]

    # Entire batch failed
    if len(batch) == 0:
        return None

    return default_collate(batch)


class ConvertBHWCtoBCHW:

    def __call__(self, video):
        return video.permute(0, 3, 1, 2)


def get_baseline_transforms(train=True):

    transform_list = [
        ConvertBHWCtoBCHW(),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Resize(baseline_configs["primary_resize_size"]),
    ]

    if train:
        transform_list.append(
            transforms.RandomCrop(baseline_configs["crop_size"])
        )
    else:
        transform_list.append(
            transforms.CenterCrop(baseline_configs["crop_size"])
        )

    transform_list.extend([
        transforms.Resize(baseline_configs["resize_size"]),
        transforms.Normalize(
            mean=baseline_configs["mean"],
            std=baseline_configs["std"]
        )
    ])

    return transforms.Compose(transform_list)