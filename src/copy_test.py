import shutil
from pathlib import Path
from tqdm import tqdm


def copy_test_files():
    # Create directories if they don't exist
    source_dir = Path("humanml3d-data")
    target_dir = Path("humanml3d-test")
    target_dir.mkdir(parents=True, exist_ok=True)

    source_cliptoken_dir = source_dir / "caption_clip" / "token"
    source_clipseq_dir = source_dir / "caption_clip" / "seq"
    source_caption_dir = source_dir / "caption_raw"
    source_rifke_dir = source_dir / "smpl_rifke"

    target_cliptoken_dir = target_dir / "caption_clip" / "token"
    target_cliptoken_dir.mkdir(parents=True, exist_ok=True)
    target_clipseq_dir = target_dir / "caption_clip" / "seq"
    target_clipseq_dir.mkdir(parents=True, exist_ok=True)
    target_caption_dir = target_dir / "caption_raw"
    target_caption_dir.mkdir(parents=True, exist_ok=True)
    target_rifke_dir = target_dir / "smpl_rifke"
    target_rifke_dir.mkdir(parents=True, exist_ok=True)

    # Read the test split file
    with open(source_dir / "humanml3d_test_split.txt", "r") as f:
        test_clips = f.read().splitlines()

    # Copy files for each clip ID
    for clip_id in tqdm(test_clips):
        filepath = source_cliptoken_dir / f"{clip_id}.npy"
        shutil.copy2(filepath, target_cliptoken_dir / filepath.name)

        filepath = source_clipseq_dir / f"{clip_id}.npy"
        shutil.copy2(filepath, target_clipseq_dir / filepath.name)

        filepath = source_caption_dir / f"{clip_id}.txt"
        shutil.copy2(filepath, target_caption_dir / filepath.name)

        filepath = source_rifke_dir / f"{clip_id}.npy"
        shutil.copy2(filepath, target_rifke_dir / filepath.name)


if __name__ == "__main__":
    copy_test_files()
