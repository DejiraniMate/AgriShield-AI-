import os
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

# ============================================================
# PATHS
# ============================================================

BASE_PATH = r"D:\project\AgriShield-AI\PlantVillage_21_Split"

TRAIN_PATH = os.path.join(BASE_PATH, "train")
VAL_PATH = os.path.join(BASE_PATH, "validation")

MODEL_PATH = r"D:\project\AgriShield-AI\models\efficientnetb0_best.keras"

SAVE_PATH = r"D:\project\AgriShield-AI\models\efficientnetb0_finetuned.keras"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

FINE_TUNE_EPOCHS = 3


# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("AGRI SHIELD AI - EFFICIENTNETB0 FINE-TUNING")
print("=" * 70)

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

class_names = train_ds.class_names

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


# ============================================================
# LOAD BEST MODEL
# ============================================================

print("\nLoading best transfer-learning model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")


# ============================================================
# FIND EFFICIENTNET BASE MODEL
# ============================================================

base_model = None

for layer in model.layers:

    if "efficientnetb0" in layer.name.lower():

        base_model = layer
        break

if base_model is None:

    raise ValueError(
        "EfficientNetB0 base model was not found."
    )

print("\nEfficientNetB0 base model found.")


# ============================================================
# UNFREEZE DEEPER LAYERS
# ============================================================

print("\nConfiguring fine-tuning...")

base_model.trainable = True

# Freeze the early layers.
# Only the deeper layers will be fine-tuned.

fine_tune_from = 150

for layer in base_model.layers[:fine_tune_from]:

    layer.trainable = False

for layer in base_model.layers[fine_tune_from:]:

    layer.trainable = True


# Keep BatchNormalization layers frozen.
# This provides more stable fine-tuning on a relatively
# small dataset and prevents their statistics from changing.

for layer in base_model.layers:

    if isinstance(
        layer,
        tf.keras.layers.BatchNormalization
    ):

        layer.trainable = False


# ============================================================
# COUNT TRAINABLE PARAMETERS
# ============================================================

trainable_params = np.sum([
    np.prod(variable.shape)
    for variable in model.trainable_variables
])

print(f"\nTrainable parameters: {trainable_params:,}")


# ============================================================
# RECOMPILE WITH LOW LEARNING RATE
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-5
    ),

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]
)


# ============================================================
# CALLBACKS
# ============================================================

callbacks = [

    tf.keras.callbacks.ModelCheckpoint(

        SAVE_PATH,

        monitor="val_accuracy",

        save_best_only=True,

        mode="max",

        verbose=1
    ),

    tf.keras.callbacks.EarlyStopping(

        monitor="val_loss",

        patience=2,

        restore_best_weights=True,

        verbose=1
    ),

    tf.keras.callbacks.ReduceLROnPlateau(

        monitor="val_loss",

        factor=0.2,

        patience=1,

        min_lr=1e-7,

        verbose=1
    )
]


# ============================================================
# FINE-TUNE
# ============================================================

print("\n" + "=" * 70)
print("STARTING FINE-TUNING")
print("=" * 70)

history = model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=FINE_TUNE_EPOCHS,

    class_weight=class_weights,

    callbacks=callbacks
)


# ============================================================
# SAVE FINAL FINE-TUNED MODEL
# ============================================================

model.save(SAVE_PATH)

print("\nFine-tuned model saved to:")

print(SAVE_PATH)


print("\n" + "=" * 70)
print("FINE-TUNING COMPLETE")
print("=" * 70)