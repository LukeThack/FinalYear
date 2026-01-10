from detect_dark_ships import find_dark_ships
from esa_snappy import ProductIO
from readAIS import read_AIS_data
import numpy
import cv2
import json
import matplotlib.pyplot as plt
import pandas as pd
from getting_shp_vectors import get_coastline_vectors
import math

_,confrimed_ships,_=find_dark_ships("2023-06-03 18:03:00","2023-06-03 18:10:00","20230603","subset_1_of_mosaic_msk.dim",0.0001516640332, 0.04868127104362205,250,5)

def show_image(key,low,high,ship,band,file_name,size,ais_folder):
    x=int(ship[1])
    y=int(ship[2])
    type=key
    if x<size:
        start_x=0
    else:
        start_x=int(x-size//2)
    if y<size:
        start_y=0
    else:
        start_y=int(y-size//2)
    data=numpy.zeros((size,size))
    band.readPixels(start_x,start_y,size,size,data)
    data=numpy.clip(data,low,high)
    img=cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    img_3_channel=cv2.merge([img,img,img])
    plt.imshow(img_3_channel)
    plt.title(type)
    plt.axis('off')
    plt.show()
    category=0
    while category not in ["0","1","2","3"]:
        category=input("Enter category for ship (0- Container Ship, 1- Oil Tanker, 2- Fishing,  3-Other): \n")
    category=int(category)
    acceptable=2
    while acceptable not in ["0","1"]:
        acceptable=input("Is this image acceptable? (1- Yes, 0- No): \n")
    acceptable=int(acceptable)
    if acceptable:
        cv2.imwrite("SSDD_YOLO_IMAGES/dataset/images/train/"+file_name, img_3_channel)

    return category,acceptable




def label_ships(image_path,low,high,confirmed_ships,json_file,max_image_id,ais_folder):
    product = ProductIO.readProduct(image_path)
    band=product.getBand("Sigma0_VH")
    all_ship_keys=confirmed_ships.keys()

    for i,key in enumerate(all_ship_keys):
        ship=confirmed_ships[key]
        file_name="luke_ship_{}.jpg".format(i+max_image_id)
        category,acceptable=show_image(key,low,high,ship,band,file_name,100,ais_folder)
            
        if not acceptable:
            print("Label not accepted, skipping...")
            continue

        new_image={
            "file_name":file_name,
            "height":100,
            "width":100,
            "id":max_image_id+i,
        }

        new_annotation={
            "iscrowd":0,
            "image_id":max_image_id+i,
            "category_id":category,
            "id":max_image_id+i,
        }

        with open(json_file,"r") as f:
            data=json.load(f)

        if "images" not in data or not data["images"]:
            data["images"] = []
        if "annotations" not in data or not data["annotations"]:
            data["annotations"] = []

        data["images"].append(new_image)
        data["annotations"].append(new_annotation)

        if "categories" not in data or not data["categories"]:
            data["categories"] = [
                {"id": 0, "name": "Ship"},
                {"id": 1, "name": "Infrastructure"},
                {"id": 2, "name": "Ambiguous"},
            ]
        with open(json_file,"w") as f:
            json.dump(data,f,indent=4) 




label_ships("subset_1_of_mosaic_msk.dim",0.0001516640332, 0.04868127104362205,confrimed_ships,"SSDD_YOLO_IMAGES/dataset/annotations/instances_train.json",1161,2042)
