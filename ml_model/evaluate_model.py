import os
import json
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report

BASE = os.path.dirname(__file__)
DATASET = os.path.join(BASE, 'dataset')
MODEL_PATH = os.path.join(BASE, 'model.h5')
MAP_PATH = os.path.join(BASE, 'class_map.json')

if not os.path.exists(MODEL_PATH):
    print('Model file not found:', MODEL_PATH)
    raise SystemExit(1)

if os.path.exists(MAP_PATH):
    with open(MAP_PATH,'r') as fh:
        labels = json.load(fh)
else:
    labels = ["clean_water","polluted_water","algae_present"]

IMG_SIZE = 224
BATCH = 16

datagen = ImageDataGenerator(rescale=1/255.0, validation_split=0.2)

val_gen = datagen.flow_from_directory(DATASET, target_size=(IMG_SIZE,IMG_SIZE), batch_size=BATCH, class_mode='categorical', subset='validation', classes=labels, shuffle=False)

model = tf.keras.models.load_model(MODEL_PATH)

preds = model.predict(val_gen, verbose=1)
y_pred = np.argmax(preds, axis=1)
y_true = val_gen.classes

print('Labels:', labels)
print('\nClassification report:\n')
print(classification_report(y_true, y_pred, target_names=labels))

cm = confusion_matrix(y_true, y_pred)
print('\nConfusion Matrix:')
print(cm)
