import cv2
import numpy as np
import os
from glob import glob
'''
Preprocess and save image that can train a YOLO model, processing matching that of the SSDD dataset.

Parameters:
    path(string): where the image to process is
    save_path(string): where the image to process should be saved
'''


def preprocess_SSDD(path, save_path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("didnt read")
        return
    img = np.log1p(img)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    img_3_channel = cv2.merge([img, img, img])
    cv2.imwrite(save_path, img_3_channel)


'''
Applies the preprocess_SSDD function to every image in a folder.

Parameters:
    folder_path(string): path to the folder to process.
    output_path(string): path to where to save the processed images.
'''


def process_SSDD_folder(folder_path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    SSDD_images_list = glob(os.path.join(folder_path, "*.jpg"))
    for image in SSDD_images_list:
        filename = os.path.basename(image)
        save_path = os.path.join(output_path, filename)
        preprocess_SSDD(image, save_path)


base_path = "SSDD_YOLO_IMAGES/dataset/images"
output_path = "SSDD_YOLO_IMAGES/dataset/images_processed"


process_SSDD_folder(os.path.join(base_path, "train"),
                    os.path.join(output_path, "train"))
process_SSDD_folder(os.path.join(base_path, "test"),
                    os.path.join(output_path, "val"))
