from detect_dark_ships import find_dark_ships
from esa_snappy import ProductIO
from readAIS import read_AIS_data
import numpy
import cv2
import json
import matplotlib.pyplot as plt
import pandas as pd
from getting_ship_vectors import get_coastline_vectors
import math
from find_rotated_boundary_boxes import find_rotated_boundary_box

_,confrimed_ships,_=find_dark_ships("2023-06-03 18:03:00","2023-06-03 18:10:00","20230603","mosaic_msk.dim",0.0001516640332, 0.04868127104362205,250,5)

ship_type_dict={
    "HSC": "High Speed Craft",
    "MER": "Merchant",
    "TNK": "Tanker",
    "FSH": "Fishing",
    "PLS": "Pleasure Craft",
    "TUG": "Tug",
    "PLT": "Pilot Vessel",
    "PRT": "Port Tender",
    "WIG": "Wing In Ground",
    "NAV": "Navigation Aid",
    "AIR": "Aircraft",
    "SAR": "Search and Rescue",
    "LAW": "Law Enforcement",
    "MIL": "Military",
}

def show_image(key,low,high,ship,band,file_name,ais_df):
    ship_type=ais_df[ais_df["mmsi"]==int(key)]["ship_type"].values[0]
    display_ship_type=ship_type_dict.get(str(ship_type),"Other")
    x1=int(ship[3])
    y1=int(ship[4])
    x2=int(ship[5])
    y2=int(ship[6])
    box_width=x2-x1
    box_height=y2-y1

    find_rotated_boundary_box(ship,band,250)

    flat_array=numpy.zeros(box_width*box_height,dtype=numpy.float32) #has to be a flat array for read pixels of small values.
    band.readPixels(x1, y1, box_width, box_height, flat_array)
    data = flat_array.reshape((box_height, box_width))

    data=numpy.clip(data,low,high)
    img=cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    img_3_channel=cv2.merge([img,img,img])
    plt.imshow(img_3_channel,interpolation="nearest")
    plt.title(display_ship_type)
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
        cv2.imwrite("SHIP_CATEGORISATION_IMAGES/dataset/images/train/"+file_name, img_3_channel)

    return category,acceptable




def label_ships(image_path,low,high,confirmed_ships,json_file,max_image_id,ais_folder):
    ais_df=read_AIS_data(ais_folder)
    product = ProductIO.readProduct(image_path)
    band=product.getBand("Sigma0_VH")
    all_ship_keys=confirmed_ships.keys()

    for i,key in enumerate(all_ship_keys):
        ship=confirmed_ships[key]
        file_name="luke_ship_{}.jpg".format(i+max_image_id)
        category,acceptable=show_image(key,low,high,ship,band,file_name,ais_df)
            
        if not acceptable:
            print("Label not accepted, skipping...")
            continue
        
        x1=int(ship[3])
        y1=int(ship[4])
        x2=int(ship[5])
        y2=int(ship[6])
        box_width=x2-x1
        box_height=y2-y1
        area=(box_width)*(box_height)

        new_image={
            "file_name":file_name,
            "height":box_height,
            "width":box_width,
            "id":max_image_id+i,
        }


        new_annotation={
            "area":area,
            "iscrowd":0,
            "image_id":max_image_id+i,
            "bbox":[x1,y1,box_width,box_height],
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


label_ships("mosaic_msk.dim",0.0001516640332, 0.04868127104362205,confrimed_ships,"SHIP_CATEGORISATION_IMAGES/dataset/annotations/instances_train.json",0,"20230603")
