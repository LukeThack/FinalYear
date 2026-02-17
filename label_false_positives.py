from detect_dark_ships import find_dark_ships
from readSARdata import get_low_high
from esa_snappy import ProductIO
import numpy
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from getting_ship_vectors import get_coastline_vectors
import math
import os
from ultralytics import YOLO
yolo_model=YOLO("runs/obb/train/weights/best.pt")

def show_image(ship,band,file_name,size):
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
    low,high=get_low_high(band)
    data=numpy.clip(data,low,high)
    img=cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    img_3_channel=cv2.merge([img,img,img])
    img_with_boxes=img_3_channel.copy()
    results=yolo_model(img_3_channel)
    if len(results[0].obb)==0:
        return results,3,0
    else:
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




def label_ships(image_path,found_dark_ships,output_dir,max_image_id):
    product = ProductIO.readProduct(image_path)
    band=product.getBand("Gamma0_VV_ocean")

    for i,ship in enumerate(found_dark_ships):
        file_name="luke_ship_{}.jpg".format(i+max_image_id)
        results,category,acceptable=show_image(ship,band,file_name,300)
        if(len(results[0].obb)==0):
            print("No Ship detected by YOLO, skipping...")
            continue
            
        if not acceptable:
            print("Label not accepted, skipping...")
            continue


        width,height=300,300
        output_lines=[]
        for box in results[0].obb.xyxyxyxy:
            x1=float(box[0][0])/width
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


found_dark_ships,_,_=find_dark_ships("202306","satelite/image3.dim",250,5)

label_ships("satelite/image3.dim",found_dark_ships,"SSDD_RBOX_IMAGES/dataset/labels/train/",1500)

