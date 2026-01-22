from detect_dark_ships import find_dark_ships
from esa_snappy import ProductIO
from readAIS import read_AIS_data
import numpy
import cv2
import matplotlib.pyplot as plt
from getting_ship_vectors import get_coastline_vectors
import math
from ultralytics import YOLO
import os
yolo_model=YOLO("runs/obb/train/weights/best.pt")

_,found_confirmed_ships,_=find_dark_ships("2023-06-03 18:03:00","2023-06-03 18:10:00","20230603","mosaic_msk.dim",0.0001516640332, 0.04868127104362205,250,5)

pos_confirmed_ship=[]
for key in found_confirmed_ships.keys():
    ship=found_confirmed_ships[key]
    pos_confirmed_ship.append([ship.geo_centre[0],ship.geo_centre[1]])


coastline=get_coastline_vectors("coastlines-split-4326/coastlines-split-4326/lines.shp")
i=0
confirmed_ships={}

for lat,lon in pos_confirmed_ship[:]:
    min_ship_long=math.floor(lon*100)/100 #position recorded wont be perfect
    min_ship_lat=math.floor(lat*100)/100
    lat_filter=coastline[(coastline["latitude"]>min_ship_lat-0.01)&(coastline["latitude"]<min_ship_lat+0.01)] #within 0.035, about 2.2km from a coastline, ignore result.
    final_filter=lat_filter[(lat_filter["longitude"]>min_ship_long-0.01)&(lat_filter["longitude"]<min_ship_long+0.01)]

    if len(final_filter)>0 or lat<50.4: #50.4 is minimum latitude for irish sea, remove if finding is close enough to land.
        pass
    else:
        ship_key=list(found_confirmed_ships.keys())[i]
        confirmed_ships[ship_key]=found_confirmed_ships[ship_key]
    i+=1






ship_type_dict={
    "MER": "Merchant",
    "TNK": "Tanker",
    "FSH": "Fishing",
    "TUG": "Tug",
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

    if rect_height<rect_width: #rotate image to be vertical.
        angle+=90
        rect_width,rect_height=rect_height,rect_width

    rotation_matrix=cv2.getRotationMatrix2D(centre,angle,1.0)

    rad_angle=math.radians(angle)
    sin=abs(math.sin(rad_angle))
    cos=abs(math.cos(rad_angle))
    new_width=int((height*sin)+(width*cos)) #need to be adjusted after rotation to prevent cut-off
    new_height=int((height*cos)+(width*sin))

    centre_offset_x=(new_width/2)-centre[0]
    centre_offset_y=(new_height/2)-centre[1]
    rotation_matrix[0,2]+=centre_offset_x #move image to centre of new canvas, 2x2 rotation, 2x1 translation to make 2x3 matrix.
    rotation_matrix[1,2]+=centre_offset_y
    rotated_image=cv2.warpAffine(data,rotation_matrix,(new_width, new_height))

    h,w=rotated_image.shape[:2]
    max_x_crop=max(0,int((w-rect_width)/2)) #avoid negatives
    min_x_crop=min(w,int((w+rect_width)/2))
    max_y_crop=max(0,int((h-rect_height)/2))
    min_y_crop=min(h,int((h+rect_height)/2))
    rotated_image = rotated_image[y1:y2, x1:x2]


    img=cv2.normalize(rotated_image, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    img_3_channel=cv2.merge([img,img,img])
    data2=cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    data3=cv2.merge([data2,data2,data2])

    _,axes=plt.subplots(1,2,figsize=(10,5))
    axes[0].imshow(img_3_channel, interpolation="nearest")
    axes[0].set_title(display_ship_type)
    axes[0].axis('off')
    axes[1].imshow(data3, interpolation="nearest")
    axes[1].set_title("Second image")
    axes[1].axis('off')
    plt.show()

    category=0
    while category not in ["0","1","2","3","4","5"]:
        category=input("Enter category for ship (0-Merchant, 1-Tanker, 2-Fishing, 3-Military, 4-Tug, 5-Other): \n")
    category=int(category)
    acceptable=2
    while acceptable not in ["0","1"]:
        acceptable=input("Is this image acceptable? (1- Yes, 0- No): \n")
    acceptable=int(acceptable)
    if acceptable:
        cv2.imwrite("SHIP_CATEGORISATION_IMAGES/dataset/images/train/"+file_name, img_3_channel)

    return category,acceptable


def label_ships(image_path,low,high,confirmed_ships,output_dir,max_image_id,ais_folder):
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
    
        output_file_path=os.path.join(output_dir,os.path.basename(file_name).replace(".jpg",".txt"))
        with open(output_file_path,"w") as output_file:
            output_file.write(category)



label_ships("mosaic_msk.dim",0.0001516640332, 0.04868127104362205,confirmed_ships,"SHIP_CATEGORISATION_IMAGES/dataset/labels/train",0,"20230603")

