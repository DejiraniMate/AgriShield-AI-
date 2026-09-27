import os
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

BASE_PATH = r"D:\project\AgriShield-AI\PlantVillage_21_Split"

TEST_PATH = os.path.join(
    BASE_PATH,
    "test"
)

MODEL_PATH = r"D:\project\AgriShield-AI\models\efficientnetb0_finetuned.keras"

OUTPUT_DIR = r"D:\project\AgriShield-AI\models\evaluation"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32


print("\n" + "=" * 70)
print("AGRI SHIELD AI - FINAL MODEL EVALUATION")
print("=" * 70)


# ============================================================
# LOAD TEST DATA
# ============================================================

print("\nLoading test dataset...")

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_PATH,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = test_ds.class_names

print(f"\nNumber of classes: {len(class_names)}")
print(f"Test batches: {tf.data.experimental.cardinality(test_ds).numpy()}")


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading fine-tuned EfficientNetB0...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully.")


# ============================================================
# GET TRUE LABELS AND PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

y_true = []
y_pred = []


for images, labels in test_ds:

    predictions = model.predict(
        images,
        verbose=0
    )

    predicted_classes = np.argmax(
        predictions,
        axis=1
    )

    y_true.extend(
        labels.numpy()
    )

    y_pred.extend(
        predicted_classes
    )


y_true = np.array(y_true)
y_pred = np.array(y_pred)


# ============================================================
# ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

print("\n" + "=" * 70)
print("FINAL TEST ACCURACY")
print("=" * 70)

print(
    f"\nTest Accuracy: {accuracy * 100:.2f}%"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4
)

print("\n")
print(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\nGenerating confusion matrix...")

cm = confusion_matrix(
    y_true,
    y_pred
)


# ============================================================
# PLOT CONFUSION MATRIX
# ============================================================

plt.figure(
    figsize=(18, 16)
)

plt.imshow(
    cm,
    interpolation="nearest"
)

plt.title(
    "EfficientNetB0 - 21 Class Confusion Matrix"
)

plt.colorbar()

tick_marks = np.arange(
    len(class_names)
)

plt.xticks(
    tick_marks,
    class_names,
    rotation=90,
    fontsize=8
)

plt.yticks(
    tick_marks,
    class_names,
    fontsize=8
)

plt.xlabel(
    "Predicted Label"
)

plt.ylabel(
    "True Label"
)

plt.tight_layout()


confusion_matrix_path = os.path.join(
    OUTPUT_DIR,
    "confusion_matrix.png"
)

plt.savefig(
    confusion_matrix_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# SAVE REPORT
# ============================================================

report_path = os.path.join(
    OUTPUT_DIR,
    "classification_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "AGRI SHIELD AI\n"
    )

    f.write(
        "EfficientNetB0 Final Evaluation\n\n"
    )

    f.write(
        f"Test Accuracy: {accuracy * 100:.2f}%\n\n"
    )

    f.write(
        report
    )


print("\n" + "=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)

print(
    f"\nTest Accuracy: {accuracy * 100:.2f}%"
)

print(
    f"\nConfusion matrix saved to:"
)

print(
    confusion_matrix_path
)

print(
    f"\nClassification report saved to:"
)

print(
    report_path
)