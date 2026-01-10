from detect_dark_ships import find_dark_ships
from esa_snappy import ProductIO
import numpy
import cv2
import json
import matplotlib.pyplot as plt
import pandas as pd
from getting_shp_vectors import get_coastline_vectors
import math
from ultralytics import YOLO
yolo_model=YOLO("runs/detect/ssdd_offshore/weights/best.pt")

found_dark_ships,_,_=find_dark_ships("2023-06-03 18:03:00","2023-06-03 18:10:00","20230603","mosaic_msk.dim",0.0001516640332, 0.04868127104362205,250,5)

dark_ships=[]
for ship in found_dark_ships:
    dark_ships.append([ship[0].lat,ship[0].lon])


coastline=get_coastline_vectors("coastlines-split-4326/coastlines-split-4326/lines.shp")
i=0
new_dark_ships=[]
for lat,lon in dark_ships[:]:
    min_ship_long=math.floor(lon*100)/100 #position recorded wont be perfect
    min_ship_lat=math.floor(lat*100)/100
    lat_filter=coastline[(coastline["latitude"]>min_ship_lat-0.01)&(coastline["latitude"]<min_ship_lat+0.01)] #within 0.035, about 2.2km from a coastline, ignore result.
    final_filter=lat_filter[(lat_filter["longitude"]>min_ship_long-0.01)&(lat_filter["longitude"]<min_ship_long+0.01)]

    if len(final_filter)>0 or lat<50.4: #50.4 is minimum latitude for irish sea, remove if finding is close enough to land.
        pass
    else:
        new_dark_ships.append(found_dark_ships[i])
    i+=1


def show_image(low,high,ship,band,file_name,size):
    x=int(ship[1])
    y=int(ship[2])
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
    img_with_boxes=img_3_channel.copy()
    results=yolo_model(img_3_channel)
    if len(results[0].boxes)==0:
        return results,3,0
    else:
        for box in results[0].boxes.xyxy:
            x1=int(box[0])
            y1=int(box[1])
            x2=int(box[2])
            y2=int(box[3])
            cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 1)

    plt.imshow(img_with_boxes)
    plt.title("Ship "+file_name)
    plt.axis('off')
    plt.show()
    category=0
    while category not in ["0","1","2"]:
        category=input("Enter category for ship (0- Ship, 1- Likely Infrastructure, 2- Ambiguous): \n")
    category=int(category)
    acceptable=2
    while acceptable not in ["0","1"]:
        acceptable=input("Is this box/s acceptable? (1- Yes, 0- No): \n")
    acceptable=int(acceptable)
    if acceptable:
        cv2.imwrite("SSDD_YOLO_IMAGES/dataset/images/train/"+file_name, img_3_channel)

    return results,category,acceptable




def label_ships(image_path,low,high,found_dark_ships,json_file,max_image_id,max_annotation_id):
    product = ProductIO.readProduct(image_path)
    band=product.getBand("Sigma0_VH")

    for i,ship in enumerate(found_dark_ships):
        file_name="luke_ship_{}.jpg".format(i+max_image_id)
        results,category,acceptable=show_image(low,high,ship,band,file_name,300)
        if(len(results[0].boxes)==0):
            print("No Ship detected by YOLO, skipping...")
            continue
            
        if not acceptable:
            print("Label not accepted, skipping...")
            continue
        new_image={
            "file_name":file_name,
            "height":300,
            "width":300,
            "id":max_image_id+i,
        }
        new_annotations=[]
        for box in results[0].boxes.xyxy:
            x1=float(box[0])
            y1=float(box[1])
            x2=float(box[2])
            y2=float(box[3])
            area=(x2 - x1) * (y2 - y1)

            new_annotation={
                "area":area,
                "iscrowd":0,
                "image_id":max_image_id+i,
                "bbox":[x1,y1,x2-x1,y2-y1],
                "category_id":category,
                "id":max_annotation_id+i,
            }
            max_annotation_id+=1
            new_annotations.append(new_annotation)

        with open(json_file,"r") as f:
            data=json.load(f)
        data["images"].append(new_image)
        data["annotations"].extend(new_annotations)
        if "categories" not in data or not data["categories"]:
            data["categories"] = [
                {"id": 0, "name": "Ship"},
                {"id": 1, "name": "Infrastructure"},
                {"id": 2, "name": "Ambiguous"},
            ]
        with open(json_file,"w") as f:
            json.dump(data,f,indent=4) 
print(len(new_dark_ships))
label_ships("mosaic_msk.dim",0.0001516640332, 0.04868127104362205,new_dark_ships,"SSDD_YOLO_IMAGES/dataset/annotations/instances_train.json",1161,2042)
