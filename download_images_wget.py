#!/usr/bin/env python3
"""
Download images using direct HTTP requests to HuggingFace CDN.
This avoids API rate limits.
"""

import json
import subprocess
from pathlib import Path
from tqdm import tqdm
import time

def main():
    print("=" * 60)
    print("Downloading images via direct HTTP")
    print("=" * 60)

    # Read JSON to get image IDs
    data_file = Path("/nas03/yixuh/vlm-adaptive-resoning/LLaMA-Factory/data/cold_start_9k/train_all.json")
    with open(data_file, 'r') as f:
        data = json.load(f)

    # Extract image indices
    image_indices = set()
    for item in data:
        if 'images' in item:
            for img_path in item['images']:
                # Extract index from path like "cold_start_9k/images/7.jpg"
                filename = img_path.split('/')[-1]
                idx = filename.replace('.jpg', '')
                image_indices.add(idx)

    image_indices = sorted(image_indices, key=lambda x: int(x) if x.isdigit() else 0)

    print(f"\nNeed to download {len(image_indices)} images")

    target_dir = Path("/nas03/yixuh/vlm-adaptive-resoning/LLaMA-Factory/data/cold_start_9k/images")
    target_dir.mkdir(parents=True, exist_ok=True)

    # Check existing
    existing = {f.stem for f in target_dir.glob("*.jpg")}
    print(f"Already have {len(existing)} images")

    to_download = [idx for idx in image_indices if idx not in existing]
    print(f"Need to download {len(to_download)} images")

    # Base URL for HuggingFace resolve endpoint
    repo_id = "LMMs-Lab-Turtle/Vision-SR1-Cold-9K"
    base_url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/cold_start"

    downloaded = 0
    failed = []

    print("\nDownloading...")
    for idx in tqdm(to_download):
        try:
            filename = f"{idx}.jpg"
            url = f"{base_url}/{filename}"
            output_path = target_dir / filename

            # Use wget for download
            cmd = [
                "wget",
                "-q",  # Quiet
                "--tries=3",
                "--timeout=30",
                "-O", str(output_path),
                url
            ]

            result = subprocess.run(cmd, capture_output=True)

            if result.returncode == 0 and output_path.exists():
                # Verify it's a real image, not an LFS pointer
                if output_path.stat().st_size > 1000:  # Real images are larger
                    downloaded += 1
                else:
                    output_path.unlink()  # Remove small file
                    failed.append(idx)
            else:
                failed.append(idx)
                if output_path.exists():
                    output_path.unlink()

            # Small delay to be nice to the server
            if downloaded % 50 == 0:
                time.sleep(1)

        except Exception as e:
            failed.append(idx)
            if len(failed) <= 5:
                print(f"\nError downloading {idx}: {e}")

    print(f"\n{'=' * 60}")
    print(f"Download complete!")
    print(f"Downloaded: {downloaded} images")
    print(f"Failed: {len(failed)} images")
    print(f"Total images: {len(list(target_dir.glob('*.jpg')))}")
    print(f"{'=' * 60}")

    if failed:
        print(f"\nSaving failed indices to failed_images.txt...")
        with open("failed_images.txt", "w") as f:
            f.write("\n".join(str(x) for x in failed))

if __name__ == "__main__":
    main()
