from detect_dark_ships import find_dark_ships
from esa_snappy import ProductIO
import numpy
import cv2
import json
import matplotlib.pyplot as plt
import pandas as pd
from getting_ship_vectors import get_coastline_vectors
import math
import os
from ultralytics import YOLO
yolo_model=YOLO("runs/obb/train/weights/best.pt")

found_dark_ships,_,_=find_dark_ships("2023-06-03 18:03:00","2023-06-03 18:10:00","20230603","mosaic_msk.dim",0.0001516640332, 0.04868127104362205,250,5)

dark_ships=[]
for ship in found_dark_ships:
    dark_ships.append([ship.geo_centre[0],ship.geo_centre[1]])


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
    x=int(ship.pixel_centre[0])
    y=int(ship.pixel_centre[1])
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
    if len(results[0].obb)==0:
        return results,3,0
    else:
        width,height=300,300
        for box in results[0].obb.xyxyxyxy:

            x1=int(box[0][0])
            y1=int(box[0][1])
            x2=int(box[1][0])
            y2=int(box[1][1])
            x3=int(box[2][0])
            y3=int(box[2][1])
            x4=int(box[3][0])
            y4=int(box[3][1])

            cv2.line(img_with_boxes,(x1,y1),(x2,y2),(0,255,0),1)
            cv2.line(img_with_boxes,(x2,y2),(x3,y3),(0,255,0),1)
            cv2.line(img_with_boxes,(x3,y3),(x4,y4),(0,255,0),1)
            cv2.line(img_with_boxes,(x4,y4),(x1,y1),(0,255,0),1)

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
        cv2.imwrite("SSDD_RBOX_IMAGES/dataset/images/train/"+file_name, img_3_channel)

    return results,category,acceptable




def label_ships(image_path,low,high,found_dark_ships,output_dir,max_image_id):
    product = ProductIO.readProduct(image_path)
    band=product.getBand("Sigma0_VH")

    for i,ship in enumerate(found_dark_ships):
        file_name="luke_ship_{}.jpg".format(i+max_image_id)
        results,category,acceptable=show_image(low,high,ship,band,file_name,300)
        if(len(results[0].obb)==0):
            print("No Ship detected by YOLO, skipping...")
            continue
            
        if not acceptable:
            print("Label not accepted, skipping...")
            continue


        width,height=300,300
        output_lines=[]
        for box in results[0].obb.xyxyxyxy:
            x1=float(box[1][0])/width
            y1=float(box[0][1])/height
            x2=float(box[1][0])/width
            y2=float(box[1][1])/height
            x3=float(box[2][0])/width
            y3=float(box[2][1])/height
            x4=float(box[3][0])/width
            y4=float(box[3][1])/height

            output_line="{:} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(category,x1,y1,x2,y2,x3,y3,x4,y4)

            output_lines.append(output_line)

        output_file_path=os.path.join(output_dir,os.path.basename(file_name).replace(".jpg",".txt"))
        with open(output_file_path,"a") as output_file:
            for output_line in output_lines:
                output_file.write(output_line)



label_ships("mosaic_msk.dim",0.0001516640332, 0.04868127104362205,new_dark_ships,"SSDD_RBOX_IMAGES/dataset/labels/train/",1161)
