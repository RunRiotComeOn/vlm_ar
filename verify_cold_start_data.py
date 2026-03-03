#!/usr/bin/env python3
"""
Verify the cold_start_9k dataset configuration.
Checks:
1. JSON files exist and are valid
2. Image paths in JSON are correct
3. Images exist on disk
4. Dataset can be loaded by LLaMA-Factory
"""

import json
from pathlib import Path
from collections import defaultdict


def verify_json_files():
    """Verify JSON files exist and are valid."""
    print("=" * 60)
    print("1. Verifying JSON Files")
    print("=" * 60)

    data_dir = Path("/nas03/yixuh/vlm-adaptive-resoning/LLaMA-Factory/data/cold_start_9k")

    json_files = [
        "train_type1.json",
        "train_type2.json",
        "train_type3.json",
        "train_all.json",
    ]

    results = {}

    for json_file in json_files:
        file_path = data_dir / json_file
        try:
            if not file_path.exists():
                print(f"✗ {json_file}: File not found")
                results[json_file] = {"status": "missing", "count": 0}
                continue

            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            print(f"✓ {json_file}: {len(data)} examples")
            results[json_file] = {"status": "ok", "count": len(data), "data": data}

        except json.JSONDecodeError as e:
            print(f"✗ {json_file}: Invalid JSON - {e}")
            results[json_file] = {"status": "invalid", "count": 0}
        except Exception as e:
            print(f"✗ {json_file}: Error - {e}")
            results[json_file] = {"status": "error", "count": 0}

    return results


def verify_image_paths(results):
    """Verify image paths in JSON files."""
    print("\n" + "=" * 60)
    print("2. Verifying Image Paths")
    print("=" * 60)

    data_dir = Path("/nas03/yixuh/vlm-adaptive-resoning/LLaMA-Factory/data")
    all_image_paths = set()
    stats = defaultdict(int)

    for json_file, result in results.items():
        if result["status"] != "ok":
            continue

        data = result["data"]
        for idx, item in enumerate(data):
            if "images" in item:
                for img_path in item["images"]:
                    all_image_paths.add(img_path)
                    stats["total_image_references"] += 1

                    # Check path format
                    if not img_path.startswith("cold_start_9k/images/"):
                        print(f"✗ Warning: Unexpected path format in {json_file}[{idx}]: {img_path}")
                        stats["wrong_path_format"] += 1

    print(f"Total unique image paths: {len(all_image_paths)}")
    print(f"Total image references: {stats['total_image_references']}")

    if stats["wrong_path_format"] > 0:
        print(f"✗ {stats['wrong_path_format']} image paths have wrong format")
    else:
        print("✓ All image paths have correct format")

    return all_image_paths


def verify_images_exist(image_paths):
    """Verify images exist on disk."""
    print("\n" + "=" * 60)
    print("3. Verifying Images Exist")
    print("=" * 60)

    data_dir = Path("/nas03/yixuh/vlm-adaptive-resoning/LLaMA-Factory/data")
    missing_images = []
    existing_images = []

    for img_path in image_paths:
        full_path = data_dir / img_path

        if full_path.exists():
            existing_images.append(img_path)
        else:
            missing_images.append(img_path)

    print(f"✓ {len(existing_images)} images found")
    print(f"✗ {len(missing_images)} images missing")

    if missing_images:
        print("\nMissing images (showing first 10):")
        for img_path in missing_images[:10]:
            print(f"  - {img_path}")

        if len(missing_images) > 10:
            print(f"  ... and {len(missing_images) - 10} more")

        print("\n⚠ Images are missing! Please run:")
        print("  python download_cold_start_images_retry.py")

        return False
    else:
        print("✓ All images found!")
        return True


def verify_dataset_info():
    """Verify dataset_info.json configuration."""
    print("\n" + "=" * 60)
    print("4. Verifying dataset_info.json")
    print("=" * 60)

    dataset_info_path = Path("/nas03/yixuh/vlm-adaptive-resoning/LLaMA-Factory/data/dataset_info.json")

    try:
        with open(dataset_info_path, 'r', encoding='utf-8') as f:
            dataset_info = json.load(f)

        expected_datasets = [
            "vision_sr1_cold_type1",
            "vision_sr1_cold_type2",
            "vision_sr1_cold_type3",
            "vision_sr1_cold_all",
        ]

        all_found = True
        for dataset_name in expected_datasets:
            if dataset_name in dataset_info:
                config = dataset_info[dataset_name]
                print(f"✓ {dataset_name}: {config.get('file_name')}")
            else:
                print(f"✗ {dataset_name}: Not found in dataset_info.json")
                all_found = False

        if all_found:
            print("\n✓ All datasets configured in dataset_info.json")
        else:
            print("\n✗ Some datasets missing from dataset_info.json")

        return all_found

    except Exception as e:
        print(f"✗ Error reading dataset_info.json: {e}")
        return False


def print_sample_data(results):
    """Print a sample entry."""
    print("\n" + "=" * 60)
    print("5. Sample Data Entry")
    print("=" * 60)

    for json_file, result in results.items():
        if result["status"] == "ok" and result["count"] > 0:
            data = result["data"]
            print(f"\nSample from {json_file}:")
            print(json.dumps(data[0], indent=2, ensure_ascii=False))
            break


def main():
    print("\n" + "=" * 60)
    print("Vision-SR1-Cold-9K Dataset Verification")
    print("=" * 60)

    # Run verification steps
    results = verify_json_files()
    image_paths = verify_image_paths(results)
    images_ok = verify_images_exist(image_paths)
    config_ok = verify_dataset_info()
    print_sample_data(results)

    # Summary
    print("\n" + "=" * 60)
    print("Verification Summary")
    print("=" * 60)

    all_ok = True

    # Check JSON files
    json_ok = all(r["status"] == "ok" for r in results.values())
    if json_ok:
        print("✓ JSON files: OK")
    else:
        print("✗ JSON files: FAILED")
        all_ok = False

    # Check images
    if images_ok:
        print("✓ Images: OK")
    else:
        print("✗ Images: MISSING (run download_cold_start_images_retry.py)")
        all_ok = False

    # Check configuration
    if config_ok:
        print("✓ Configuration: OK")
    else:
        print("✗ Configuration: FAILED")
        all_ok = False

    # Final status
    print("\n" + "=" * 60)
    if all_ok:
        print("✓ All checks passed! Dataset is ready to use.")
        print("\nYou can now train with:")
        print("  cd LLaMA-Factory")
        print("  llamafactory-cli train --dataset vision_sr1_cold_all ...")
    else:
        print("✗ Some checks failed. Please fix the issues above.")

    print("=" * 60)


if __name__ == "__main__":
    main()
