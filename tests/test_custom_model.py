from tensorflow import keras
import glob
import cv2
import os
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input
ship_model = keras.models.load_model("ship_classifier_final.keras")
image_directory = "SHIP_CATEGORISATION_IMAGES/dataset/images/"
label_directory = "SHIP_CATEGORISATION_IMAGES/dataset/labels/"
image_paths = glob.glob(os.path.join(image_directory, "*.jpg"))


def test_custom_model_accuracy():
    images = []
    labels = []

    for image in image_paths:
        img = cv2.imread(image)
        if img is None:
            print(f"Couldnt read image: {image}")
            continue
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

    _, accuracy = ship_model.evaluate(images, labels)
    assert(accuracy>0.8)

