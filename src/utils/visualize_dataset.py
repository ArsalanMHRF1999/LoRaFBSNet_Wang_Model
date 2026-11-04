import matplotlib.pyplot as plt
import numpy as np
import torch

def show_ucf_clip_sample_with_class(dataset, idx_to_class, sample_idx=0, n_show=6, title_prefix="Sample"):
    clip, label = dataset[sample_idx]
    print(clip.shape)
    T = clip.shape[0]
    frame_ids = np.linspace(0, T - 1, n_show).astype(int)

    mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    std  = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)

    clip_vis = clip * std + mean
    clip_vis = clip_vis.clamp(0, 1)

    fig, axes = plt.subplots(1, len(frame_ids), figsize=(3 * len(frame_ids), 3))
    if len(frame_ids) == 1:
        axes = [axes]

    for ax, t in zip(axes, frame_ids):
        img = clip_vis[t].permute(1, 2, 0).cpu().numpy()
        ax.imshow(img)
        ax.set_title(f"t={t}")
        ax.axis("off")

    class_name = idx_to_class[label]
    plt.suptitle(f"{title_prefix} | label={label} | class={class_name}")
    plt.tight_layout()
    plt.show()