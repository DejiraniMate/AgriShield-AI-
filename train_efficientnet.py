import os
import json
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

# ============================================================
# SETTINGS
# ============================================================

BASE_PATH = r"D:\project\AgriShield-AI\PlantVillage_21_Split"

TRAIN_PATH = os.path.join(BASE_PATH, "train")
VAL_PATH = os.path.join(BASE_PATH, "validation")
TEST_PATH = os.path.join(BASE_PATH, "test")

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 21
SEED = 42

INITIAL_EPOCHS = 1

MODEL_DIR = r"D:\project\AgriShield-AI\models"

os.makedirs(MODEL_DIR, exist_ok=True)

print("\n" + "=" * 70)
print("AGRI SHIELD AI - EFFICIENTNETB0 TRAINING")
print("=" * 70)


# ============================================================
# LOAD DATASETS
# ============================================================

print("\nLoading training dataset...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_PATH,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)

print("\nLoading validation dataset...")

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_PATH,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# CLASS NAMES
# ============================================================

class_names = train_ds.class_names

print("\nClasses:")
for i, name in enumerate(class_names):
    print(f"{i:2d}: {name}")

print(f"\nNumber of classes: {len(class_names)}")


# ============================================================
# CLASS WEIGHTS
# ============================================================

print("\nCalculating class weights...")

class_counts = np.zeros(len(class_names), dtype=int)

for _, labels in train_ds:

    for label in labels.numpy():
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

for i, name in enumerate(class_names):

    print(
        f"{i:2d} | "
        f"{name:55s} | "
        f"Images: {class_counts[i]:4d} | "
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
# DATA PIPELINE
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)


# ============================================================
# BUILD EFFICIENTNETB0
# ============================================================

print("\n" + "=" * 70)
print("BUILDING EFFICIENTNETB0")
print("=" * 70)

base_model = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3)
)

# Freeze pretrained model
base_model.trainable = False


# ============================================================
# BUILD CLASSIFICATION MODEL
# ============================================================

inputs = tf.keras.Input(
    shape=(224, 224, 3),
    name="input_image"
)

x = data_augmentation(inputs)

x = base_model(
    x,
    training=False
)

x = tf.keras.layers.GlobalAveragePooling2D()(x)

x = tf.keras.layers.Dropout(
    0.3
)(x)

outputs = tf.keras.layers.Dense(
    NUM_CLASSES,
    activation="softmax",
    name="disease_prediction"
)(x)

model = tf.keras.Model(
    inputs,
    outputs,
    name="AgriShield_EfficientNetB0"
)


# ============================================================
# COMPILE MODEL
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# MODEL SUMMARY
# ============================================================

model.summary()


# ============================================================
# CALLBACKS
# ============================================================

checkpoint_path = os.path.join(
    MODEL_DIR,
    "efficientnetb0_best.keras"
)

callbacks = [

    tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )
]


# ============================================================
# TRAIN
# ============================================================

print("\n" + "=" * 70)
print("STARTING TRANSFER LEARNING")
print("=" * 70)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=INITIAL_EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks
)


# ============================================================
# SAVE INITIAL MODEL
# ============================================================

initial_model_path = os.path.join(
    MODEL_DIR,
    "efficientnetb0_transfer_learning.keras"
)

model.save(initial_model_path)

print("\nInitial model saved to:")
print(initial_model_path)


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

history_path = os.path.join(
    MODEL_DIR,
    "training_history.json"
)

history_data = {
    key: [float(value) for value in values]
    for key, values in history.history.items()
}

with open(history_path, "w") as f:
    json.dump(history_data, f, indent=4)

print("\nTraining history saved to:")
print(history_path)


print("\n" + "=" * 70)
print("TRANSFER LEARNING COMPLETE")
print("=" * 70)