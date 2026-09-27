import os
import shutil
from sklearn.model_selection import train_test_split

# ============================================================
# PATHS
# ============================================================

DATASET_PATH = r"D:\project\AgriShield-AI\PlantVillage-Dataset-master"

OUTPUT_PATH = r"D:\project\AgriShield-AI\PlantVillage_21_Split"

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# ============================================================
# SPLIT SETTINGS
# ============================================================

TRAIN_SIZE = 0.70
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15


# ============================================================
# CREATE OUTPUT FOLDERS
# ============================================================

train_path = os.path.join(OUTPUT_PATH, "train")
val_path = os.path.join(OUTPUT_PATH, "validation")
test_path = os.path.join(OUTPUT_PATH, "test")

os.makedirs(train_path, exist_ok=True)
os.makedirs(val_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)


# ============================================================
# FIND CLASS FOLDERS
# ============================================================

classes = sorted([
    folder
    for folder in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, folder))
    and folder not in ["train", "validation", "test"]
])


print("\n" + "=" * 70)
print("PLANTVILLAGE DATASET SPLITTING")
print("=" * 70)

print(f"\nNumber of classes: {len(classes)}")


# ============================================================
# PROCESS EACH CLASS
# ============================================================

for class_name in classes:

    class_path = os.path.join(DATASET_PATH, class_name)

    images = [
        file
        for file in os.listdir(class_path)
        if file.lower().endswith(IMAGE_EXTENSIONS)
    ]

    print(f"\nProcessing: {class_name}")
    print(f"Total images: {len(images)}")

    # --------------------------------------------------------
    # FIRST SPLIT
    # 70% TRAIN
    # 30% TEMPORARY
    # --------------------------------------------------------

    train_images, temp_images = train_test_split(
        images,
        test_size=(VALIDATION_SIZE + TEST_SIZE),
        random_state=42,
        shuffle=True
    )

    # --------------------------------------------------------
    # SECOND SPLIT
    # 15% VALIDATION
    # 15% TEST
    # --------------------------------------------------------

    val_images, test_images = train_test_split(
        temp_images,
        test_size=0.5,
        random_state=42,
        shuffle=True
    )

    # Create class folders
    train_class_path = os.path.join(train_path, class_name)
    val_class_path = os.path.join(val_path, class_name)
    test_class_path = os.path.join(test_path, class_name)

    os.makedirs(train_class_path, exist_ok=True)
    os.makedirs(val_class_path, exist_ok=True)
    os.makedirs(test_class_path, exist_ok=True)

    # --------------------------------------------------------
    # COPY TRAIN IMAGES
    # --------------------------------------------------------

    for image in train_images:

        source = os.path.join(class_path, image)
        destination = os.path.join(train_class_path, image)

        shutil.copy2(source, destination)

    # --------------------------------------------------------
    # COPY VALIDATION IMAGES
    # --------------------------------------------------------

    for image in val_images:

        source = os.path.join(class_path, image)
        destination = os.path.join(val_class_path, image)

        shutil.copy2(source, destination)

    # --------------------------------------------------------
    # COPY TEST IMAGES
    # --------------------------------------------------------

    for image in test_images:

        source = os.path.join(class_path, image)
        destination = os.path.join(test_class_path, image)

        shutil.copy2(source, destination)

    print(
        f"Train: {len(train_images)} | "
        f"Validation: {len(val_images)} | "
        f"Test: {len(test_images)}"
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("DATASET SPLIT COMPLETE")
print("=" * 70)

print("\nOutput location:")
print(OUTPUT_PATH)

print("\nDataset structure:")

print("""
PlantVillage_21_Split/
│
├── train/
│   ├── 21 classes
│
├── validation/
│   ├── 21 classes
│
└── test/
    ├── 21 classes
""")

print("Split ratio: 70% Train / 15% Validation / 15% Test")