
import random
import shutil
import re
import csv
from pathlib import Path

# Set random seed for reproducibility
random.seed(42)

# Project folders
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dataset"

ORIGINAL_DIR = DATA_DIR / "original"
TRAIN_DIR = DATA_DIR / "train"
CANDIDATE_DIR = DATA_DIR / "test_candidate"
TEST_DIR = DATA_DIR / "test"

# Dataset settings
CLASSES = ["tom", "jerry"]
NUM_TRAIN = 800
NUM_TEST = 200

IMAGE_EXTENSIONS = {".jpg"}


def get_frame_number(file):
    # Extract the number from a filename like frame00123.jpg
    match = re.fullmatch(
        r"frame(\d+)",
        file.stem,
        re.IGNORECASE
    )

    if match:
        return int(match.group(1))

    return None


def get_images(folder):
    # Get valid image files
    images = []

    for file in folder.iterdir():

        if not file.is_file():
            continue

        if file.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        frame_number = get_frame_number(file)

        if frame_number is None:
            print(f"Skipped: {file.name}")
            continue

        images.append(file)

    # Sort by frame number
    images.sort(key=lambda x: (get_frame_number(x), x.name))

    return images


def check_empty(folder):
    # Prevent mixing with previous datasets
    if folder.exists() and any(folder.iterdir()):
        raise FileExistsError(
            f"Folder is not empty: {folder}\n"
            "Please clear this folder before running."
        )


# Check folders before copying
for class_name in CLASSES:

    if not (ORIGINAL_DIR / class_name).exists():
        raise FileNotFoundError(
            f"Original folder not found: {class_name}"
        )

    for target_dir in [TRAIN_DIR, CANDIDATE_DIR, TEST_DIR]:
        check_empty(target_dir / class_name)


# Prepare selection records
records = []

# Process each class
for class_name in CLASSES:

    print(f"\n=== Processing {class_name.upper()} ===")

    original_folder = ORIGINAL_DIR / class_name
    train_folder = TRAIN_DIR / class_name
    candidate_folder = CANDIDATE_DIR / class_name
    test_folder = TEST_DIR / class_name

    images = get_images(original_folder)

    if len(images) < NUM_TRAIN + NUM_TEST:
        raise ValueError(
            f"Not enough images for {class_name}."
        )

    # 1. Select the first 800 images for training
    train_images = images[:NUM_TRAIN]

    # 2. Use the remaining images as test candidates
    candidate_images = images[NUM_TRAIN:]

    # 3. Select only odd-numbered frames
    odd_images = [
        image for image in candidate_images
        if get_frame_number(image) % 2 == 1
    ]

    if len(odd_images) < NUM_TEST:
        raise ValueError(
            f"{class_name}: Only {len(odd_images)} "
            f"odd-numbered images are available. "
            f"At least {NUM_TEST} are required."
        )

    # 4. Randomly select 200 odd-numbered images
    test_images = random.sample(odd_images, NUM_TEST)

    # Create output folders
    for folder in [train_folder, candidate_folder, test_folder]:
        folder.mkdir(parents=True, exist_ok=True)

    # Copy training images
    for image in train_images:
        shutil.copy2(image, train_folder / image.name)

        records.append([
            class_name,
            image.name,
            "train"
        ])

    # Copy candidate images
    for image in candidate_images:
        shutil.copy2(image, candidate_folder / image.name)

    # Copy test images
    for image in test_images:
        shutil.copy2(image, test_folder / image.name)

        records.append([
            class_name,
            image.name,
            "test"
        ])

    # Print summary
    print(f"Original: {len(images)}")
    print(f"Train: {len(train_images)}")
    print(f"Test candidates: {len(candidate_images)}")
    print(f"Odd candidates: {len(odd_images)}")
    print(f"Test: {len(test_images)}")


# Save selected image filenames
record_path = DATA_DIR / "dataset_split.csv"

with open(record_path, "w", newline="", encoding="utf-8") as f:

    writer = csv.writer(f)

    writer.writerow([
        "class",
        "filename",
        "split"
    ])

    writer.writerows(records)


print("\nDataset creation completed!")
print(f"Split information saved to: {record_path}")