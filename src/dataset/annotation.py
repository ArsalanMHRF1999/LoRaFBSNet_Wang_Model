from src.utils.paths import DATA_ROOT, ANNOTATION_ROOT

SPLITS = ["train", "val", "test"]

def create_annotations(split):

    split_root = DATA_ROOT / split
    annotation_dir = ANNOTATION_ROOT / split
    annotation_dir.mkdir(parents=True, exist_ok=True)

    video_paths = sorted(split_root.glob("*/*.avi"))

    relative_paths = [
        path.relative_to(split_root).as_posix()
        for path in video_paths
    ]

    assert len(relative_paths) > 0, f"No videos found in {split}"

    # The authors' class reads trainlist when train=True
    # and testlist when train=False.
    content = "\n".join(relative_paths) + "\n"

    (annotation_dir / "trainlist01.txt").write_text(content)
    (annotation_dir / "testlist01.txt").write_text(content)

    print(
        f"{split}: {len(relative_paths)} videos | "
        f"Annotations: {annotation_dir}"
    )


for split in SPLITS:
    create_annotations(split)