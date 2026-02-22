from detect_dark_ships import find_dark_ships
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
    data=numpy.nan_to_num(data, nan=1e-6) #replace nans with small value. make normalise less aggressive.
    data=numpy.log1p(data)
    img=cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    img_3_channel=cv2.merge([img,img,img])
    img_with_boxes=img_3_channel.copy()

    box=ship.rbbox

    x1=int(box[0]-start_x)
    y1=int(box[1]-start_y)
    x2=int(box[2]-start_x)
    y2=int(box[3]-start_y)
    x3=int(box[4]-start_x)
    y3=int(box[5]-start_y)
    x4=int(box[6]-start_x)
    y4=int(box[7]-start_y)

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

    return category,acceptable




def label_ships(image_path,found_dark_ships,output_dir,max_image_id):
    product = ProductIO.readProduct(image_path)
    band=product.getBand("Gamma0_VV_ocean")

    for i,ship in enumerate(found_dark_ships):
        file_name="luke_ship_{}.jpg".format(i+max_image_id)
        category,acceptable=show_image(ship,band,file_name,300)

        if not acceptable:
            print("Label not accepted, skipping...")
            continue


        width,height=300,300
        output_lines=[]
        box=ship.rbbox
        x1=int(box[0])
        y1=int(box[1])
        x2=int(box[2])
        y2=int(box[3])
        x3=int(box[4])
        y3=int(box[5])
        x4=int(box[6])
        y4=int(box[7])
        output_line="{:} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(category,x1,y1,x2,y2,x3,y3,x4,y4)

        output_lines.append(output_line)

        output_file_path=os.path.join(output_dir,os.path.basename(file_name).replace(".jpg",".txt"))
        with open(output_file_path,"a") as output_file:
            for output_line in output_lines:
                output_file.write(output_line)


found_dark_ships,found_confirmed_ships,multi_ships=find_dark_ships("202306","satelite/image3.dim",250,50)
print(len(found_dark_ships))

ships=[]
for key in found_confirmed_ships.keys():
    ship=found_confirmed_ships[key]
    ships.append(ship)

label_ships("satelite/image3.dim",ships,"SSDD_RBOX_IMAGES/dataset/labels/train/",1500)

