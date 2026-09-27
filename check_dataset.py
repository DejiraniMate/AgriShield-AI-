
import os
from PIL import Image

DATASET_PATH = r"D:\project\AgriShield-AI\PlantVillage-Dataset-master"

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

total_images = 0
valid_images = 0
corrupt_images = []

print("\n" + "=" * 70)
print("AGRI SHIELD AI - DATASET INTEGRITY CHECK")
print("=" * 70)

class_folders = [
    folder
    for folder in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, folder))
]

class_folders.sort()

print(f"\nNumber of class folders found: {len(class_folders)}")

for class_name in class_folders:

    class_path = os.path.join(DATASET_PATH, class_name)

    class_total = 0
    class_valid = 0
    class_corrupt = 0

    print("\nChecking:", class_name)

    for filename in os.listdir(class_path):

        if not filename.lower().endswith(IMAGE_EXTENSIONS):
            continue

        total_images += 1
        class_total += 1

        image_path = os.path.join(class_path, filename)

        try:
            with Image.open(image_path) as img:
                img.verify()

            with Image.open(image_path) as img:
                img.convert("RGB").load()

            valid_images += 1
            class_valid += 1

        except Exception as error:

            class_corrupt += 1

            corrupt_images.append({
                "class": class_name,
                "filename": filename,
                "path": image_path,
                "error": str(error)
            })

    print(
        f"Total: {class_total} | "
        f"Valid: {class_valid} | "
        f"Corrupt: {class_corrupt}"
    )

print("\n" + "=" * 70)
print("DATASET INTEGRITY RESULT")
print("=" * 70)

print(f"\nTotal images checked : {total_images}")
print(f"Valid images         : {valid_images}")
print(f"Corrupt images       : {len(corrupt_images)}")

if len(corrupt_images) == 0:

    print("\nNO CORRUPT IMAGES FOUND.")

else:

    print("\nCORRUPT IMAGE DETAILS")

    for image in corrupt_images:
        print("\nClass :", image["class"])
        print("File  :", image["filename"])
        print("Path  :", image["path"])
        print("Error :", image["error"])

report_path = os.path.join(
    DATASET_PATH,
    "corrupt_images_report.txt"
)

with open(report_path, "w", encoding="utf-8") as file:

    file.write("AGRI SHIELD AI - CORRUPT IMAGE REPORT\n")
    file.write("=" * 70 + "\n\n")

    file.write(f"Total images checked : {total_images}\n")
    file.write(f"Valid images         : {valid_images}\n")
    file.write(f"Corrupt images       : {len(corrupt_images)}\n\n")

    for image in corrupt_images:

        file.write(f"Class: {image['class']}\n")
        file.write(f"File: {image['filename']}\n")
        file.write(f"Path: {image['path']}\n")
        file.write(f"Error: {image['error']}\n")
        file.write("-" * 70 + "\n")

print("\nReport saved to:")
print(report_path)

print("\n" + "=" * 70)
print("DATASET CHECK COMPLETE")
print("=" * 70)
