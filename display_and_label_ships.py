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
from ultralytics import YOLO
yolo_model=YOLO("runs/obb/train/weights/best.pt")

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

    bbox=ship.rbbox
    x1,y1,x2,y2,x3,y3,x4,y4=bbox
    max_x=x1
    min_x=x1
    max_y=y1
    min_y=y1

    for i in range(0,len(bbox),2):
        x=bbox[i]
        y=bbox[i+1]
        if x<min_x:
            min_x=x
        elif x>max_x:
            max_x=x
        if y<min_y:
            min_y=y
        elif y>max_y:
            max_y=y
    height=max_y-min_y
    width=max_x-min_x


    flat_array=numpy.zeros(width*height,dtype=numpy.float32) #has to be a flat array for read pixels of small values.
    band.readPixels(min_x, min_y, width, height, flat_array)
    data = flat_array.reshape((height, width))
    data=numpy.clip(data,low,high)

    points=numpy.array([[x1,y1],[x2,y2],[x3,y3],[x4,y4]],dtype=numpy.float32)
    points[:,0]-=min_x
    points[:,1]-=min_y
    centre,(rect_width,rect_height),angle=cv2.minAreaRect(points)
    if angle<-45:
        angle+=90

    rotation_matrix=cv2.getRotationMatrix2D(centre,angle,1.0)

    rad_angle=math.radians(angle)
    sin=abs(math.sin(rad_angle))
    cos=abs(math.cos(rad_angle))
    new_width=int((height*sin)+(width*cos)) #need to be adjusted after rotation to prevent cut-off
    new_height=int((height*cos)+(width*sin))

    centre_offset_x=(new_width/2)-centre[0]
    centre_offset_y=(new_height/2)-centre[1]
    rotation_matrix[0,2]+=centre_offset_x #move matrix to new centre, 2x2 rotation, 2x1 translation to make 2x3 matrix.
    rotation_matrix[1,2]+=centre_offset_y

    rotated_image=cv2.warpAffine(data,rotation_matrix,(new_width, new_height))

    centre[0]

    h,w=rotated_image.shape[:2]
    rotated_image = rotated_image[int((h-rect_height)/2):int((h+rect_height)/2),int((w-rect_width)/2):int((w+rect_width)/2)] #rotated about centre of bounding box - centre of new image is centre of bounding box
    if rect_height<rect_width:
        rotated_image=cv2.rotate(rotated_image, cv2.ROTATE_90_CLOCKWISE) #orientate vertically




    img=cv2.normalize(rotated_image, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    img_3_channel=cv2.merge([img,img,img])
    data2=cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    data3=cv2.merge([data2,data2,data2])


    fig,axes=plt.subplots(1,2,figsize=(10,5))
    axes[0].imshow(img_3_channel, interpolation="nearest")
    axes[0].set_title(display_ship_type)
    axes[0].axis('off')
    axes[1].imshow(data3, interpolation="nearest")
    axes[1].set_title("Second image")
    axes[1].axis('off')

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
''' 
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
'''
label_ships("mosaic_msk.dim",0.0001516640332, 0.04868127104362205,confrimed_ships,"SHIP_CATEGORISATION_IMAGES/dataset/annotations/instances_train.json",0,"20230603")

