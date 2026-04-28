import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from PIL import Image, UnidentifiedImageError
import os
import shutil

dataset_path = "dataset"

IMG_SIZE = 224
BATCH = 16
EPOCHS = 12

# Quick dataset sanitizer: move unreadable/corrupt images to dataset_corrupt
def sanitize_dataset(base_dir):
    corrupt_dir = os.path.join(os.path.dirname(__file__), "dataset_corrupt")
    os.makedirs(corrupt_dir, exist_ok=True)
    moved = 0

    for root, dirs, files in os.walk(base_dir):
        for fname in files:
            src = os.path.join(root, fname)
            # ignore directories that are already the corrupt folder
            if corrupt_dir in src:
                continue

            try:
                if os.path.getsize(src) == 0:
                    raise UnidentifiedImageError("empty file")
            except OSError:
                # if we can't stat the file, treat it as corrupt
                dest_dir = os.path.join(corrupt_dir, os.path.relpath(root, base_dir))
                os.makedirs(dest_dir, exist_ok=True)
                shutil.move(src, os.path.join(dest_dir, fname))
                moved += 1
                continue

            try:
                with Image.open(src) as im:
                    im.verify()
            except (UnidentifiedImageError, OSError, ValueError):
                dest_dir = os.path.join(corrupt_dir, os.path.relpath(root, base_dir))
                os.makedirs(dest_dir, exist_ok=True)
                shutil.move(src, os.path.join(dest_dir, fname))
                moved += 1

    return moved


print("Checking dataset for corrupt images...")
moved_count = sanitize_dataset(os.path.join(os.path.dirname(__file__), dataset_path))
if moved_count:
    print(f"Moved {moved_count} corrupt/unreadable files to dataset_corrupt/")
else:
    print("No corrupt images found.")

datagen = ImageDataGenerator(
    rescale=1/255.0,
    validation_split=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
)

train_gen = datagen.flow_from_directory(
    dataset_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH,
    class_mode="categorical",
    subset="training",
)

val_gen = datagen.flow_from_directory(
    dataset_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH,
    class_mode="categorical",
    subset="validation",
)

# Ensure consistent class ordering and save mapping for inference
# Ensure consistent class ordering and save mapping for inference
EXPECTED_CLASSES = ["Foam water", "Oil Spill Water", "algae_present", "clean_water", "polluted_water"]


train_gen = datagen.flow_from_directory(
    dataset_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH,
    class_mode="categorical",
    subset="training",
    classes=EXPECTED_CLASSES,
)

val_gen = datagen.flow_from_directory(
    dataset_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH,
    class_mode="categorical",
    subset="validation",
    classes=EXPECTED_CLASSES,
)

# persist class mapping to disk so the Flask server can use it
import json
with open(os.path.join(os.path.dirname(__file__), "class_map.json"), "w") as fh:
    json.dump(EXPECTED_CLASSES, fh)

model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    BatchNormalization(),
    MaxPooling2D(),

    Conv2D(64, (3, 3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D(),

    Conv2D(128, (3, 3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D(),

    Flatten(),
    Dense(256, activation="relu"),
    Dropout(0.4),

    Dense(len(EXPECTED_CLASSES), activation="softmax"),  # Dynamic number of classes

])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

history = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)

model.save("model.h5")
print("\n✔ Training complete! Model saved as model.h5")
