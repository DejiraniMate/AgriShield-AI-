import sys
import numpy as np
import tensorflow as tf

MODEL_PATH = "models/efficientnetb0_finetuned.keras"
CLASS_NAMES_PATH = "class_names.txt"
IMAGE_SIZE = (224, 224)


def load_model_and_classes():
    model = tf.keras.models.load_model(MODEL_PATH)

    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        class_names = [line.strip() for line in f if line.strip()]

    return model, class_names


def predict_image(image_path):
    model, class_names = load_model_and_classes()

    image = tf.keras.utils.load_img(
        image_path,
        target_size=IMAGE_SIZE
    )

    image_array = tf.keras.utils.img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)

    # Original image prediction
    original_prediction = model.predict(image_array, verbose=0)

    # Horizontal flip prediction
    flipped_image = tf.image.flip_left_right(image_array)
    flipped_prediction = model.predict(flipped_image, verbose=0)

    # Average predictions
    final_prediction = (
        original_prediction + flipped_prediction
    ) / 2.0

    predicted_index = np.argmax(final_prediction[0])
    predicted_class = class_names[predicted_index]
    confidence = final_prediction[0][predicted_index] * 100

    print("\nPrediction Result")
    print("-----------------")
    print(f"Disease/Class : {predicted_class}")
    print(f"Confidence    : {confidence:.2f}%")


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage:")
        print("python scripts/predict.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    try:
        predict_image(image_path)
    except Exception as e:
        print(f"Error: {e}")