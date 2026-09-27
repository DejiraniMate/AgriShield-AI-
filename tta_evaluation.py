import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report

# ============================================================
# PATHS
# ============================================================

TEST_PATH = r"D:\project\AgriShield-AI\PlantVillage_21_Split\test"
MODEL_PATH = r"D:\project\AgriShield-AI\models\efficientnetb0_finetuned.keras"

# ============================================================
# SETTINGS
# ============================================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")

# ============================================================
# LOAD TEST DATA
# ============================================================

print("\nLoading test dataset...")

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_PATH,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = test_ds.class_names

print("\nNumber of classes:", len(class_names))

# ============================================================
# PREDICTION STORAGE
# ============================================================

all_true = []

original_probs = []
horizontal_probs = []
vertical_probs = []
rotated_probs = []

print("\nRunning multi-TTA evaluation...")

# ============================================================
# TTA
# ============================================================

for images, labels in test_ds:

    # Original
    p_original = model.predict(
        images,
        verbose=0
    )

    # Horizontal flip
    horizontal_images = tf.image.flip_left_right(images)

    p_horizontal = model.predict(
        horizontal_images,
        verbose=0
    )

    # Vertical flip
    vertical_images = tf.image.flip_up_down(images)

    p_vertical = model.predict(
        vertical_images,
        verbose=0
    )

    # Small 90-degree rotation
    rotated_images = tf.image.rot90(images, k=1)

    p_rotated = model.predict(
        rotated_images,
        verbose=0
    )

    all_true.extend(labels.numpy())

    original_probs.append(p_original)
    horizontal_probs.append(p_horizontal)
    vertical_probs.append(p_vertical)
    rotated_probs.append(p_rotated)

# ============================================================
# COMBINE
# ============================================================

y_true = np.array(all_true)

original_probs = np.concatenate(original_probs)
horizontal_probs = np.concatenate(horizontal_probs)
vertical_probs = np.concatenate(vertical_probs)
rotated_probs = np.concatenate(rotated_probs)

# Different TTA combinations

horizontal_tta = (
    original_probs +
    horizontal_probs
) / 2

multi_tta = (
    original_probs +
    horizontal_probs +
    vertical_probs +
    rotated_probs
) / 4

# Predictions

original_predictions = np.argmax(
    original_probs,
    axis=1
)

horizontal_predictions = np.argmax(
    horizontal_tta,
    axis=1
)

multi_predictions = np.argmax(
    multi_tta,
    axis=1
)

# ============================================================
# ACCURACY
# ============================================================

original_accuracy = accuracy_score(
    y_true,
    original_predictions
)

horizontal_accuracy = accuracy_score(
    y_true,
    horizontal_predictions
)

multi_accuracy = accuracy_score(
    y_true,
    multi_predictions
)

# ============================================================
# RESULTS
# ============================================================

print("\n==============================================")
print("MULTI-TTA RESULTS")
print("==============================================")

print(
    f"\nOriginal Accuracy       : "
    f"{original_accuracy * 100:.2f}%"
)

print(
    f"Horizontal Flip TTA    : "
    f"{horizontal_accuracy * 100:.2f}%"
)

print(
    f"Multi-TTA Accuracy     : "
    f"{multi_accuracy * 100:.2f}%"
)

print(
    f"\nMulti-TTA Change       : "
    f"{(multi_accuracy - original_accuracy) * 100:+.2f}%"
)

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n==============================================")
print("MULTI-TTA CLASSIFICATION REPORT")
print("==============================================\n")

print(
    classification_report(
        y_true,
        multi_predictions,
        target_names=class_names,
        digits=4
    )
)

# ============================================================
# SAVE RESULTS
# ============================================================

OUTPUT_FILE = (
    r"D:\project\AgriShield-AI\models\evaluation"
    r"\multi_tta_results.txt"
)

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write("MULTI-TTA RESULTS\n")
    f.write("=" * 60 + "\n\n")

    f.write(
        f"Original Accuracy: "
        f"{original_accuracy * 100:.2f}%\n"
    )

    f.write(
        f"Horizontal Flip TTA: "
        f"{horizontal_accuracy * 100:.2f}%\n"
    )

    f.write(
        f"Multi-TTA Accuracy: "
        f"{multi_accuracy * 100:.2f}%\n"
    )

    f.write(
        f"Multi-TTA Change: "
        f"{(multi_accuracy - original_accuracy) * 100:+.2f}%\n\n"
    )

    f.write("Classification Report\n")
    f.write("=" * 60 + "\n\n")

    f.write(
        classification_report(
            y_true,
            multi_predictions,
            target_names=class_names,
            digits=4
        )
    )

print("\nResults saved to:")
print(OUTPUT_FILE)