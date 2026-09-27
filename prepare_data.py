import os
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# ============================================================
# PATHS
# ============================================================

BASE_PATH = r"D:\project\AgriShield-AI\PlantVillage_21_Split"

TRAIN_PATH = os.path.join(BASE_PATH, "train")
VAL_PATH = os.path.join(BASE_PATH, "validation")
TEST_PATH = os.path.join(BASE_PATH, "test")


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42


print("\n" + "=" * 70)
print("AGRI SHIELD AI - DATA PREPARATION")
print("=" * 70)


# ============================================================
# LOAD TRAINING DATA
# ============================================================

print("\nLoading training dataset...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_PATH,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)


# ============================================================
# LOAD VALIDATION DATA
# ============================================================

print("Loading validation dataset...")

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_PATH,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# LOAD TEST DATA
# ============================================================

print("Loading test dataset...")

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_PATH,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# CLASS INFORMATION
# ============================================================

class_names = train_ds.class_names

print("\n" + "=" * 70)
print("CLASS INFORMATION")
print("=" * 70)

print(f"Number of classes: {len(class_names)}")

for i, class_name in enumerate(class_names):
    print(f"{i:2d} : {class_name}")


# ============================================================
# CHECK DATASET SIZE
# ============================================================

train_batches = tf.data.experimental.cardinality(train_ds).numpy()
val_batches = tf.data.experimental.cardinality(val_ds).numpy()
test_batches = tf.data.experimental.cardinality(test_ds).numpy()

print("\n" + "=" * 70)
print("DATASET INFORMATION")
print("=" * 70)

print(f"Training batches   : {train_batches}")
print(f"Validation batches : {val_batches}")
print(f"Test batches       : {test_batches}")

print(f"Image size         : {IMAGE_SIZE}")
print(f"Batch size         : {BATCH_SIZE}")


# ============================================================
# CALCULATE CLASS WEIGHTS
# ============================================================

print("\n" + "=" * 70)
print("CALCULATING CLASS WEIGHTS")
print("=" * 70)

class_counts = np.zeros(len(class_names), dtype=int)

for images, labels in train_ds:

    labels_numpy = labels.numpy()

    for label in labels_numpy:
        class_counts[label] += 1


class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(len(class_names)),
    y=np.repeat(
        np.arange(len(class_names)),
        class_counts
    )
)

class_weights = {
    i: float(weight)
    for i, weight in enumerate(class_weights_array)
}


print("\nClass weights:")

for i, class_name in enumerate(class_names):

    print(
        f"{class_name:55s} "
        f"Images: {class_counts[i]:5d} "
        f"Weight: {class_weights[i]:.3f}"
    )


# ============================================================
# DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip(
            "horizontal"
        ),

        tf.keras.layers.RandomRotation(
            0.1
        ),

        tf.keras.layers.RandomZoom(
            0.1
        ),

        tf.keras.layers.RandomTranslation(
            height_factor=0.1,
            width_factor=0.1
        )
    ],
    name="data_augmentation"
)


# ============================================================
# EFFICIENTNETB0 PREPROCESSING
# ============================================================

# EfficientNetB0 in tf.keras includes its input
# preprocessing internally.

print("\nEfficientNetB0 preprocessing:")
print("Handled by the EfficientNetB0 model.")


# ============================================================
# PERFORMANCE OPTIMIZATION
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)


# ============================================================
# TEST ONE BATCH
# ============================================================

print("\n" + "=" * 70)
print("TESTING DATA LOADER")
print("=" * 70)

for images, labels in train_ds.take(1):

    print(f"Image batch shape : {images.shape}")
    print(f"Label batch shape : {labels.shape}")
    print(f"Image data type   : {images.dtype}")
    print(f"Label data type   : {labels.dtype}")


# ============================================================
# SAVE CLASS NAMES
# ============================================================

class_names_path = r"D:\project\AgriShield-AI\class_names.txt"

with open(class_names_path, "w", encoding="utf-8") as f:

    for class_name in class_names:
        f.write(class_name + "\n")


print("\nClass names saved to:")
print(class_names_path)


print("\n" + "=" * 70)
print("DATA PREPARATION COMPLETE")
print("=" * 70)