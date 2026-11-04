baseline_configs = {
    "crop_size": (256, 256),
    "resize_size": (224, 224),
    "mean": [0.5, 0.6, 0.6],
    "std": [0.5, 0.5, 0.5],
    "primary_resize_size": (256, 340),
}

FRAMES_PER_CLIP = 16
STEP_BETWEEN_CLIPS = 16

BATCH_SIZE = 4
NUM_WORKERS = 2
