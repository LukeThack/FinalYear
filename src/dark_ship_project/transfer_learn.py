import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, models
import cv2
import glob
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications import ResNet50

resnet_model = ResNet50(weights="imagenet", include_top=False, input_shape=(
    224, 224, 3))  # doesnt include classification layer
resnet_model.trainable = False  # small dataset, freeze trainable layers
data_augmentation = keras.Sequential([layers.RandomFlip("horizontal"), layers.RandomZoom(
    0.3), layers.RandomTranslation(0.2, 0.2), layers.RandomContrast(0.3),])

model = models.Sequential([data_augmentation, resnet_model, layers.GlobalAveragePooling2D(
), layers.Dense(64, activation="relu"), layers.Dropout(0.5), layers.Dense(1, activation="sigmoid")])
# optimiser set for transfer learning, binary_crossentropy, best loss function for binary outputs.
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4),
              loss="binary_crossentropy", metrics=["accuracy"])

image_directory = "SHIP_CATEGORISATION_IMAGES/dataset/images/"
label_directory = "SHIP_CATEGORISATION_IMAGES/dataset/labels/"
image_paths = glob.glob(image_directory+"*.jpg")

images = []
labels = []

for image in image_paths:
    img = cv2.imread(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = preprocess_input(img)
    img = img.astype(np.float32)

    label_path = os.path.basename(image).replace("jpg", "txt")
    with open(os.path.join(label_directory, label_path)) as label_file:
        label = int(label_file.read().strip())
        if label in [0, 1]:
            labels.append(label)
            images.append(img)

images = np.array(images)
labels = np.array(labels)
image_train, image_test, label_train, label_test = train_test_split(
    images, labels, random_state=0, test_size=0.2, stratify=labels)

callbacks = [keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True)]

first_model = model.fit(image_train, label_train, validation_data=(
    image_test, label_test), epochs=20, callbacks=callbacks, batch_size=16)

resnet_model.trainable = True

for layer in resnet_model.layers[:-30]:
    layer.trainable = False

model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-5),
              loss="binary_crossentropy", metrics=["accuracy"])

ship_model = model.fit(image_train, label_train, validation_data=(
    image_test, label_test), epochs=20, batch_size=16, callbacks=callbacks)

model.save("ship_classifier_final.keras")

loss, accuracy = model.evaluate(images, labels)

print("Test Loss:", loss)
print("Test Accuracy:", accuracy)
